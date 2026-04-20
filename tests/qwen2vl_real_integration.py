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
# 【核心修改】：使用 DynamicCache 代替基础 Cache
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info
from src.memory.manager import HeteroKVManager


# ---------------------------------------------------------
# 核心桥接层：将 HeteroKVManager 包装为 Hugging Face 标准 Cache
# ---------------------------------------------------------
class HeteroHuggingFaceCache(DynamicCache):
    """
    这是一个拦截器（Hook），用于接管 Qwen2-VL 的 KV 存储逻辑。
    它继承自 HF 的 DynamicCache，将原生的大模型显存读写，重定向到你的异构显存池中。
    """

    def __init__(self, manager: HeteroKVManager):
        # DynamicCache 的 __init__ 不需要传奇怪的 layer 参数
        super().__init__()
        self.manager = manager
        # [核心修复] 手动显式初始化，防止被 HF 的底层版本改动背刺
        self.key_cache = []
        self.value_cache = []
        # 使用独立的计数器来管理 token 长度，最安全
        self._seen_tokens = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        # [学术核心]：在这里拦截原生模型的 FP16 KV 数据！
        # 1. 触发 HeteroKV 的量化压缩 (FP16 -> 4-bit)
        # 2. 触发 Heavy Hitter Oracle 驱逐判定
        # 3. 将冷数据 offload 到 DRAM

        # 简单回退机制（测试跑通流程用，后续可替换为真正的 compress 与 swap_in 逻辑）
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # 仅在处理第 0 层时累加 token 数量，避免重复计算
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # 彻底摆脱对 self.key_cache 长度判断的依赖，直接返回准确的统计值
        return self._seen_tokens


def run_qwen_integration():
    device = "cuda:0"
    model_path = "./models/Qwen2-VL-7B"  # 确认这是你下载模型的本地路径

    print("\n" + "=" * 80)
    print("🤖 Hetero-KV × Qwen2-VL: 真实视频多模态推理测试")
    print("=" * 80)

    # 1. 加载模型与处理器 (使用 bf16 精度节省基础显存)
    print("\n[Step 1] 加载 Qwen2-VL-7B-Instruct 模型...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径 {model_path} 是否正确。详细报错：{e}")
        return

    # 2. 准备视频输入
    print("\n[Step 2] 解析视频并生成密集视觉 Token...")
    # 这里我们强制按 1 fps 采样，保证稠密输入，给 KV Cache 施加压力
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "test_video.mp4",
                    "fps": 1.0,
                },
                {"type": "text", "text": "请详细描述这段视频中发生的事情。"},
            ],
        }
    ]

    # 预处理视觉信息和文本 prompt
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
    print(f"   ✅ 视频解析完成。系统提示词 + 密集视觉 Token 总数: {total_input_tokens}")

    # 3. 挂载 Hetero-KV 调度器
    print("\n[Step 3] 初始化 Hetero-KV 调度器并注入模型...")
    hbm_limit_blocks = 150
    manager = HeteroKVManager(hbm_max_blocks=hbm_limit_blocks, block_size=16, device=device)
    custom_cache = HeteroHuggingFaceCache(manager)

    torch.cuda.reset_peak_memory_stats()

    # 4. 执行多模态生成 (生成过程会将 KV 数据推入我们的 custom_cache)
    print("\n[Step 4] 开始推理生成 (观察显存峰值)...")
    t0 = time.perf_counter()

    # 强制清理缓存，准备进入生成阶段
    torch.cuda.empty_cache()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            past_key_values=custom_cache  # 【核心挂载点】将我们的异构缓存对象传入
        )
    generation_time = time.perf_counter() - t0

    # 截取新生成的 Token 并解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    peak_hbm = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    print("\n" + "=" * 80)
    print("✨ 推理完成！模型输出:")
    print("-" * 80)
    print(output_text[0])
    print("-" * 80)
    print(f"   ⏱️ 推理耗时: {generation_time:.2f} 秒")
    print(f"   📊 实际生成阶段 HBM 显存峰值: {peak_hbm:.2f} GB")
    print("=" * 80)


if __name__ == "__main__":
    run_qwen_integration()