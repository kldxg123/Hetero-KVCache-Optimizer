import sys
import os

# =================================================================
# 🌟 0. 核心修复：将项目根目录动态注入到环境变量中
# =================================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 🌟 现在可以直接从你的项目中安全引入固化好的组件了！
from src.memory.manager import HeteroKVManager
from src.memory.cache import HeteroTransientCache


def run_simple_test():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"

    print("加载模型中...")
    # 1. 常规加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    # 阉割 lm_head 以防止单卡测试时预填充阶段 OOM
    original_forward = model.lm_head.forward

    def memory_safe_lm_head_forward(hidden_states): return original_forward(hidden_states[:, -1:, :])

    model.lm_head.forward = memory_safe_lm_head_forward

    # 2. 准备超长输入 (这里构造一个约万字的长文本进行测试)
    print("构建测试输入...")
    long_text = "This is a normal background text frame, nothing special here. " * 2000
    long_text += " The secret anomaly code is ANOMALY_CODE_9527. Remember it. "

    messages = [{"role": "user", "content": [{"type": "text", "text": long_text}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)

    # =================================================================
    # 🌟 3. 核心接入：只需两行代码，瞬间让模型具备 O(1) 显存能力！
    # =================================================================
    manager = HeteroKVManager(hbm_max_blocks=10, block_size=16, device=device)
    # 注入你的瞬态分离缓存 (默认保留 64 Sink + 8192 Tail)
    hetero_cache = HeteroTransientCache(manager=manager, sink_tokens=64, keep_tail=8192)

    print("开始执行生成...")
    # 4. 执行生成，显存峰值自动骤降，全自动生效！
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=20, past_key_values=hetero_cache)

    # 截取新生成的 Token 并解码
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    result = processor.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "=" * 50)
    print("🎯 生成结果:", result)
    print("=" * 50)


if __name__ == "__main__":
    run_simple_test()