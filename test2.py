"""
Qwen2.5-VL模型封装类
提供多模态文本生成和embedding提取功能
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import GenerationConfig
from qwen_vl_utils import process_vision_info

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类"""
    model_path: str
    device: str = "cuda:0"
    torch_dtype: torch.dtype = torch.float16
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.05


class Qwen2_5VLWrapper:
    """
    Qwen2.5-VL模型封装类

    提供以下功能：
    1. 多模态文本生成
    2. 句子embedding提取
    3. 相似度计算
    """

    def __init__(self, config: ModelConfig):
        """
        初始化模型

        Args:
            config: 模型配置对象
        """
        self.config = config
        self.device = config.device

        try:
            logger.info(f"正在加载模型: {config.model_path}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_path,
                torch_dtype=config.torch_dtype
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(config.model_path)
            logger.info("模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            pooling_mask: Optional[torch.LongTensor] = None,
            output_hidden_states: bool = True,
            **kwargs
    ) -> torch.Tensor:
        """
        前向传播，提取embedding

        Args:
            各种输入参数
            output_hidden_states: 是否输出hidden states

        Returns:
            embedding张量
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=output_hidden_states,
                **kwargs
            )

        # 获取最后一层hidden state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        elif hasattr(outputs, "encoder_last_hidden_state"):
            last_hidden_state = outputs.encoder_last_hidden_state
        else:
            last_hidden_state = outputs.last_hidden_state

        # 使用pooling_mask或attention_mask进行pooling
        mask = pooling_mask if pooling_mask is not None else attention_mask
        if mask is None:
            raise ValueError("需要提供attention_mask或pooling_mask")

        # 计算embedding
        left_padding = (mask[:, -1].sum() == mask.shape[0])
        if left_padding:
            embeddings = last_hidden_state[:, -1]
        else:
            sequence_lengths = mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            embeddings = last_hidden_state[torch.arange(
                batch_size, device=last_hidden_state.device
            ), sequence_lengths]

        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def prepare_inputs(self, messages: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        准备模型输入

        Args:
            messages: 消息列表，包含文本、图像、视频等信息

        Returns:
            处理后的输入字典
        """
        try:
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 处理视觉信息
            vision_result = process_vision_info(messages)
            if vision_result is None:
                image_inputs, video_inputs = None, None
            elif len(vision_result) == 2:
                image_inputs, video_inputs = vision_result
            else:
                image_inputs, video_inputs, _ = vision_result

            # 准备输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # 移动到指定设备
            return {k: v.to(self.device) for k, v in inputs.items()}

        except Exception as e:
            logger.error(f"输入准备失败: {e}")
            raise

    def get_sentence_embedding(self, **inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取句子embedding

        Args:
            **inputs: 模型输入参数

        Returns:
            (last_token_embedding, mean_pooling_embedding)
        """
        # 使用forward方法获取embedding
        last_token_emb = self.forward(**inputs)

        # 计算mean pooling embedding
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("inputs 中没有 attention_mask")

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = mask.sum(1)
        mean_pool_emb = sum_embeddings / sum_mask

        return last_token_emb, mean_pool_emb

    def generate_text(self, messages: List[Dict], gen_config: Optional[Dict] = None) -> str:
        """
        生成文本

        Args:
            messages: 输入消息
            gen_config: 生成配置字典

        Returns:
            生成的文本
        """
        if gen_config is None:
            gen_config = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": True
            }

        try:
            inputs = self.prepare_inputs(messages)

            # 使用transformers的GenerationConfig
            generation_config = GenerationConfig(**gen_config)

            generated_ids = self.model.generate(
                # input_ids=inputs['input_ids'],
                # attention_mask=inputs['attention_mask'],
                **inputs,
                generation_config=generation_config
            )

            # 提取新生成的部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return output_text[0] if len(output_text) == 1 else output_text

        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise

    def calculate_similarity(self, query_messages: List[Dict], candidate_messages_list: List[List[Dict]]) -> List[
        float]:
        """
        计算相似度

        Args:
            query_messages: 查询消息
            candidate_messages_list: 候选消息列表

        Returns:
            相似度列表
        """
        try:
            # 获取查询embedding
            query_inputs = self.prepare_inputs(query_messages)
            query_last_token_emb, query_mean_pool_emb = self.get_sentence_embedding(**query_inputs)

            similarities = []

            for candidate_messages in candidate_messages_list:
                candidate_inputs = self.prepare_inputs(candidate_messages)
                candidate_last_token_emb, candidate_mean_pool_emb = self.get_sentence_embedding(**candidate_inputs)

                # 计算余弦相似度
                sim = F.cosine_similarity(query_last_token_emb, candidate_last_token_emb)
                similarities.append(sim.item())

            return similarities

        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            raise

    def encode(self, messages: List[Dict], **kwargs) -> torch.Tensor:
        """
        编码接口，提取embedding

        Args:
            messages: 输入消息
            **kwargs: 其他参数

        Returns:
            embedding张量
        """
        inputs = self.prepare_inputs(messages)
        # 只传递模型需要的参数
        model_inputs = {
            'input_ids': inputs.get('input_ids'),
            'attention_mask': inputs.get('attention_mask'),
            'pixel_values': inputs.get('pixel_values'),
            'image_grid_thw': inputs.get('image_grid_thw')
        }
        return self.forward(**model_inputs)


class MultiModalExperiment:
    """多模态实验类"""

    def __init__(self, model_config: ModelConfig):
        self.wrapper = Qwen2_5VLWrapper(model_config)

    def run_weather_classification_experiment(self) -> Dict[str, Union[str, List[float], List[str]]]:
        """
        运行天气分类实验

        Returns:
            包含生成文本和相似度的字典
        """
        # 定义候选选项
        weather_options = [
            "A:晴朗",
            "B:多云",
            "C:雨雪天",
            "D:阴天"
        ]

        # 准备查询消息
        query_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图片中的天气是什么？ 请直接选择,A:晴朗, B:多云, C:雨雪天, D:阴天"},
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                ],
            }
        ]

        # 准备候选消息
        candidate_messages_list = [
            [{"role": "user", "content": [{"type": "text", "text": option}]}]
            for option in weather_options
        ]

        # 生成文本
        generated_text = self.wrapper.generate_text(query_messages)

        # 计算相似度
        similarities = self.wrapper.calculate_similarity(query_messages, candidate_messages_list)

        # 打印结果
        print(f"生成的文本: {generated_text}")
        print("\n相似度结果:")
        for option, sim in zip(weather_options, similarities):
            print(f"{option}: {sim:.4f}")

        return {
            "generated_text": generated_text,
            "similarities": similarities,
            "options": weather_options
        }


def main():
    """主函数"""
    # 配置模型
    model_config = ModelConfig(
        model_path="/mnt/data/ai-ground/models/Qwen2.5-VL-7B-Instruct",
        device="cuda:0"
    )

    # 创建实验实例
    experiment = MultiModalExperiment(model_config)

    # 运行实验
    try:
        results = experiment.run_weather_classification_experiment()
        logger.info("实验完成")
        return results
    except Exception as e:
        logger.error(f"实验失败: {e}")
        raise


if __name__ == "__main__":
    main()
