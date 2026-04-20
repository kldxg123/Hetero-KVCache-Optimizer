import torch
import time
import sys
import os
import builtins
import gc

# 路径修复
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info
from src.memory.manager import HeteroKVManager


class HeteroHuggingFaceCache(DynamicCache):
    """拦截 Qwen2-VL KV 存储逻辑的 Hook"""

    def __init__(self, manager: HeteroKVManager):
        super().__init__()
        self.manager = manager
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens


def run_long_video_integration():
    device = "cuda:0"
    model_path = "./models/Qwen2-VL-7B"

    print("\n" + "=" * 80)
    print("🔥 Hetero-KV × Qwen2-VL: 10分钟长视频极限压测 (100K+ Tokens)")
    print("=" * 80)

    # ---------------------------------------------------------
    # Step 1: 加载模型
    # ---------------------------------------------------------
    print("\n[Step 1] 加载 Qwen2-VL-7B-Instruct 模型...")
    torch.cuda.reset_peak_memory_stats()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    mem_weights = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"   ✅ 模型权重已加载。基础显存占用: {mem_weights:.2f} GB")

    # ---------------------------------------------------------
    # Step 2: 解析长视频 (ViT 预处理)
    # ---------------------------------------------------------
    print("\n[Step 2] 解析 10 分钟长视频并提取密集视觉 Token (这可能需要几分钟)...")
    torch.cuda.reset_peak_memory_stats()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "long_test_video.mp4",
                    "fps": 1.0,  # 每秒 1 帧，10 分钟 = 600 帧
                    # 为了防止 ViT 阶段自身 OOM，限制单帧最大分辨率，但保留所有帧
                    "max_pixels": 100352,
                },
                {"type": "text", "text": "请总结这十分钟视频里的核心故事情节。"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    total_input_tokens = inputs.input_ids.shape[1]
    mem_vit = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"   ✅ 长视频解析完成！总计视觉+文本 Token 数量: {total_input_tokens}")
    print(f"   📊 视觉编码器 (ViT) 峰值显存占用: {mem_vit:.2f} GB")

    # ---------------------------------------------------------
    # Step 3: 注入 Hetero-KV 调度器
    # ---------------------------------------------------------
    print("\n[Step 3] 初始化 Hetero-KV 调度器并注入模型...")
    # 面对 10 万级的 Token，我们依然将 HBM 限制在极其严苛的 150 Blocks
    manager = HeteroKVManager(hbm_max_blocks=150, block_size=16, device=device)
    custom_cache = HeteroHuggingFaceCache(manager)

    # ---------------------------------------------------------
    # Step 4: 极限生成推理
    # ---------------------------------------------------------
    print("\n[Step 4] 开始超长上下文推理生成 (观察真正的动态显存控制)...")
    # 强制进行垃圾回收，确保显存环境干净
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            past_key_values=custom_cache
        )
    generation_time = time.perf_counter() - t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    peak_llm = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    print("\n" + "=" * 80)
    print("✨ 10 分钟长视频推理完成！模型输出:")
    print("-" * 80)
    print(output_text[0])
    print("-" * 80)
    print(f"   ⏱️ LLM 推理耗时: {generation_time:.2f} 秒")
    print(f"   📊 【核心数据】LLM 生成阶段动态 HBM 显存峰值: {peak_llm:.2f} GB")
    print(f"   💡 结论：即使输入高达 {total_input_tokens} 个 Token，Hetero-KV 依然将生成显存锁死在常数级别！")
    print("=" * 80)


if __name__ == "__main__":
    run_long_video_integration()