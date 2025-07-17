from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import inspect


class Qwen2VLWrapper:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def prepare_inputs(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_sentence_embedding(self, **inputs):
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        elif hasattr(outputs, "encoder_last_hidden_state"):
            last_hidden_state = outputs.encoder_last_hidden_state
        else:
            raise AttributeError("模型输出不包含hidden_states或encoder_last_hidden_state")

        # 获取 attention_mask
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            raise ValueError("inputs 中没有 attention_mask，无法定位有效token")
        # 对于每个样本，找到最后一个有效token的位置
        seq_lens = attention_mask.sum(dim=1) - 1  # shape: [batch_size]
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        # 取出每个样本最后一个有效token的向量
        last_token_emb = last_hidden_state[batch_indices, seq_lens]  # [batch, hidden_size]

        # mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = mask.sum(1)
        mean_pool_emb = sum_embeddings / sum_mask  # [batch, hidden_size]

        # 计算两者的余弦相似度
        import torch.nn.functional as F
        sim = F.cosine_similarity(last_token_emb, mean_pool_emb)
        print("最后一个token embedding与mean pooling embedding的余弦相似度：", sim)

        # 返回两种embedding，便于主程序对比
        return last_token_emb, mean_pool_emb


# default: Load the model on the available device(s)
device = "cuda:0"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
wrapper = Qwen2VLWrapper("/mnt/data/ai-ground/models/Qwen/Qwen2-VL-2B-Instruct", device)
# with open("model_structure.txt", "w", encoding="utf-8") as f:
#     f.write(str(wrapper.model))

# import inspect
# with open("model_forward_source.txt", "w", encoding="utf-8") as f:
#     f.write(inspect.getsource(wrapper.model.forward))

inputs = wrapper.prepare_inputs(messages)

# 获取两种embedding
last_token_emb, mean_pool_emb = wrapper.get_sentence_embedding(**inputs)
print("last_token_emb shape:", last_token_emb.shape)
print("mean_pool_emb shape:", mean_pool_emb.shape)

# # ========== 新增：一张图片与多个文本的相似度对比实验 ==========
# texts = [
#     "女孩与狗在海边玩耍",
#     "一只猫在沙发上睡觉",
#     "海边有一只狗和一个小女孩",
#     "城市夜景灯火辉煌"
# ]

# # 图片输入
# image_messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#         ],
#     }
# ]

# # 批量准备文本输入
# text_messages_list = [
#     [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": t},
#             ],
#         }
#     ] for t in texts
# ]

# # 获取图片embedding
# image_inputs = wrapper.prepare_inputs(image_messages)
# image_last_token_emb, image_mean_pool_emb = wrapper.get_sentence_embedding(**image_inputs)

# # 批量获取文本embedding
# text_last_token_embs = []
# text_mean_pool_embs = []
# for text_messages in text_messages_list:
#     text_inputs = wrapper.prepare_inputs(text_messages)
#     last_token_emb, mean_pool_emb = wrapper.get_sentence_embedding(**text_inputs)
#     text_last_token_embs.append(last_token_emb)
#     text_mean_pool_embs.append(mean_pool_emb)

# import torch
# text_last_token_embs = torch.cat(text_last_token_embs, dim=0)
# text_mean_pool_embs = torch.cat(text_mean_pool_embs, dim=0)

# # 计算相似度
# import torch.nn.functional as F
# image_last_token_emb_expand = image_last_token_emb.expand(text_last_token_embs.shape[0], -1)
# image_mean_pool_emb_expand = image_mean_pool_emb.expand(text_mean_pool_embs.shape[0], -1)

# sim_last_token = F.cosine_similarity(text_last_token_embs, image_last_token_emb_expand)
# sim_mean_pool = F.cosine_similarity(text_mean_pool_embs, image_mean_pool_emb_expand)

# print("\n一张图片与多个文本的相似度对比：")
# for i, t in enumerate(texts):
#     print(f"文本: {t}")
#     print(f"  最后一个token embedding相似度: {sim_last_token[i].item():.4f}")
#     print(f"  mean pooling embedding相似度: {sim_mean_pool[i].item():.4f}")
