#!/usr/bin/env python3
"""
消融测试：定位准确率下降原因
逐步排除法：
  A. Baseline (标准KV cache, 短上下文)
  B. Baseline (标准KV cache, 长上下文) - 排除测试设计问题
  C. HeteroKV 无patch (不调用patch_model_for_fused_attention)
  D. HeteroKV 有patch
  E. 逐步开启功能: quant / triton / self_healing
"""

import torch
import time
import sys
import os
from PIL import Image
import numpy as np

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from transformers import AutoProcessor, LlavaForConditionalGeneration

# ═══════════════════════════════════════════════════════════════════════════════
# 加载模型
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 75)
print("消融测试：定位 HeteroKV 准确率下降原因")
print("=" * 75)

print("\n[加载模型] LLaVA-1.5-7B...")

model_path = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snapshots = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
latest = sorted(snapshots)[-1]
model_path = os.path.join(model_path, latest)

processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()
print(f"   ✅ 模型已加载 ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")

# 创建测试图像
img_arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
y, x = np.ogrid[:224, :224]
mask = (x - 112)**2 + (y - 112)**2 <= 50**2
img_arr[mask] = [255, 0, 0]
test_image = Image.fromarray(img_arr)

# QA对 - 多个问题，每个都有明确的预期关键词
qa_pairs = [
    ("What color is the circle in the image?", "red"),
    ("What shape is shown in the image?", "circle"),
    ("What is the background color of the image?", "white"),
    ("Is the circle red or blue?", "red"),
    ("How many shapes are in the image?", "one"),
]

def normalize(text):
    return text.lower().strip().replace(".", "").replace(",", "").replace("!", "")

def check_answer(response, expected):
    """检查答案是否包含预期关键词"""
    norm_resp = normalize(response)
    norm_exp = normalize(expected)
    return norm_exp in norm_resp

def build_prompt(question, context_len):
    """构建带上下文的prompt"""
    # 用有意义的重复文本填充上下文
    padding = "The quick brown fox jumps over the lazy dog. " * max(1, context_len // 12)
    return f"USER: <image>\n{padding}\n{question}\nASSISTANT:"

def run_test(label, question, expected, context_len, use_heterokv=False,
             use_patch=False, enable_quant=True, enable_triton=True,
             self_healing=False, adaptive=False):
    """运行单个测试"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    prompt = build_prompt(question, context_len)
    inputs = processor(text=prompt, images=test_image, return_tensors='pt').to('cuda')
    num_tokens = inputs.input_ids.shape[-1]

    cache = None
    ctx_manager = None

    if use_heterokv:
        from core.engine_wrapper import FusedHeteroCache
        cache = FusedHeteroCache(
            num_layers=32, sink_tokens=64, keep_tail=2048,
            chunk_size=2048, device='cuda',
            enable_quant=enable_quant, enable_triton=enable_triton,
            self_healing=self_healing, adaptive_self_healing=adaptive,
        )

    start = time.time()
    try:
        if use_patch and cache is not None:
            from core.fused_attention_patch import patch_model_for_fused_attention
            ctx_manager = patch_model_for_fused_attention(model, cache, enable_fused=True)
            ctx_manager.__enter__()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                do_sample=False,
                past_key_values=cache,
            )

        elapsed = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip()
        correct = check_answer(answer, expected)

        result = {
            'label': label,
            'tokens': num_tokens,
            'peak_mb': peak_mem,
            'time': elapsed,
            'correct': correct,
            'answer': answer[:80],
            'error': None,
        }

    except Exception as e:
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        result = {
            'label': label,
            'tokens': num_tokens,
            'peak_mb': peak_mem,
            'time': 0,
            'correct': False,
            'answer': '',
            'error': str(e)[:120],
        }

    finally:
        if ctx_manager is not None:
            try:
                ctx_manager.__exit__(None, None, None)
            except:
                pass
        del inputs
        if cache is not None:
            del cache
        torch.cuda.empty_cache()

    return result

def print_result(r):
    status = "✅" if r['correct'] else "❌"
    err = f" | ERR: {r['error']}" if r['error'] else ""
    ans = f" → {r['answer']}" if r['correct'] or not r['error'] else ""
    print(f"  {status} {r['label']:<35} {r['tokens']:>6} tokens | {r['peak_mb']:>8.0f} MB | {r['time']:>5.1f}s{err}")
    if not r['correct'] and not r['error']:
        print(f"     答案: {r['answer']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 消融测试 A: Baseline - 不同上下文长度
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("测试 A: Baseline (标准KV cache) - 排除测试设计问题")
print("   目的：确认LLaVA模型本身在不同上下文长度下准确率是否稳定")
print("=" * 75)

q, exp = qa_pairs[0]  # What color is the circle?

baseline_results = []
for ctx in [0, 500, 1000, 2000, 4000, 8000]:
    label = f"Baseline ctx={ctx}"
    r = run_test(label, q, exp, ctx, use_heterokv=False)
    baseline_results.append(r)
    print_result(r)

baseline_acc = sum(1 for r in baseline_results if r['correct']) / len(baseline_results) * 100
print(f"\n  Baseline准确率: {baseline_acc:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 消融测试 B: HeteroKV 无patch - 不同上下文长度
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("测试 B: HeteroKV 无patch - 排除patch问题")
print("   目的：确认HeteroKV本身（不经过patch）是否影响准确率")
print("=" * 75)

q, exp = qa_pairs[0]

no_patch_results = []
for ctx in [0, 500, 1000, 2000, 4000]:
    label = f"NoPatch ctx={ctx}"
    r = run_test(label, q, exp, ctx,
                 use_heterokv=True, use_patch=False,
                 enable_quant=True, enable_triton=False,
                 self_healing=False, adaptive=False)
    no_patch_results.append(r)
    print_result(r)

no_patch_acc = sum(1 for r in no_patch_results if r['correct']) / len(no_patch_results) * 100
print(f"\n  NoPatch准确率: {no_patch_acc:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 消融测试 C: HeteroKV 有patch - 不同上下文长度
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("测试 C: HeteroKV 有patch - 确认patch影响")
print("   目的：确认patch_model_for_fused_attention是否导致准确率下降")
print("=" * 75)

q, exp = qa_pairs[0]

with_patch_results = []
for ctx in [0, 500, 1000, 2000, 4000]:
    label = f"WithPatch ctx={ctx}"
    r = run_test(label, q, exp, ctx,
                 use_heterokv=True, use_patch=True,
                 enable_quant=True, enable_triton=True,
                 self_healing=False, adaptive=False)
    with_patch_results.append(r)
    print_result(r)

with_patch_acc = sum(1 for r in with_patch_results if r['correct']) / len(with_patch_results) * 100
print(f"\n  WithPatch准确率: {with_patch_acc:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 消融测试 D: 多问题测试 - 排除问题本身的问题
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("测试 D: 多问题测试 - 排除特定问题导致的准确率问题")
print("   目的：确认准确率问题是否是问题相关的")
print("=" * 75)

ctx = 500  # 固定中等上下文

multi_q_results = []
for q, exp in qa_pairs:
    # Baseline
    r0 = run_test(f"Base: {q[:30]}", q, exp, ctx, use_heterokv=False)
    print_result(r0)

    # HeteroKV
    r1 = run_test(f"HK:   {q[:30]}", q, exp, ctx,
                  use_heterokv=True, use_patch=False,
                  enable_quant=True, self_healing=False)
    print_result(r1)

    multi_q_results.append((r0, r1))

# ═══════════════════════════════════════════════════════════════════════════════
# 消融测试 E: 逐步开启功能
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("测试 E: 逐步开启功能 - 定位具体问题模块")
print("   目的：逐个开启quant/triton/self_healing，找出哪个导致准确率下降")
print("=" * 75)

q, exp = qa_pairs[0]
ctx = 2000

configs = [
    ("仅HeteroKV(无quant)",      dict(use_heterokv=True, use_patch=False, enable_quant=False, enable_triton=False, self_healing=False, adaptive=False)),
    ("HeteroKV + quant",         dict(use_heterokv=True, use_patch=False, enable_quant=True,  enable_triton=False, self_healing=False, adaptive=False)),
    ("HeteroKV + quant + patch", dict(use_heterokv=True, use_patch=True,  enable_quant=True,  enable_triton=True,  self_healing=False, adaptive=False)),
    ("全部 + self_healing",      dict(use_heterokv=True, use_patch=True,  enable_quant=True,  enable_triton=True,  self_healing=True,  adaptive=False)),
    ("全部 + adaptive",          dict(use_heterokv=True, use_patch=True,  enable_quant=True,  enable_triton=True,  self_healing=True,  adaptive=True)),
]

feature_results = []
for label, cfg in configs:
    r = run_test(label, q, exp, ctx, **cfg)
    feature_results.append(r)
    print_result(r)

# ═══════════════════════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("📊 消融测试总结")
print("=" * 75)

print(f"\n  A. Baseline (标准KV):      {baseline_acc:.0f}% 准确率")
print(f"  B. HeteroKV 无patch:       {no_patch_acc:.0f}% 准确率")
print(f"  C. HeteroKV 有patch:       {with_patch_acc:.0f}% 准确率")

base_multi = sum(1 for r0, r1 in multi_q_results if r0['correct']) / len(multi_q_results) * 100
hk_multi = sum(1 for r0, r1 in multi_q_results if r1['correct']) / len(multi_q_results) * 100
print(f"  D. 多问题 Baseline:        {base_multi:.0f}% 准确率")
print(f"  D. 多问题 HeteroKV:        {hk_multi:.0f}% 准确率")

print(f"\n  E. 功能逐步开启:")
for r in feature_results:
    status = "✅" if r['correct'] else "❌"
    err = f" (ERR: {r['error'][:60]})" if r['error'] else ""
    print(f"     {status} {r['label']:<30} {r['peak_mb']:>8.0f} MB{err}")

# 诊断结论
print(f"\n{'='*75}")
print("🔍 诊断结论:")

# 找出准确率开始下降的配置
if baseline_acc == 100 and no_patch_acc < 100:
    print("  ❌ 问题出在: HeteroKV本身（即使不使用patch）")
    print("     → 可能是prefill_update或decode_update截断KV导致信息丢失")
elif no_patch_acc == 100 and with_patch_acc < 100:
    print("  ❌ 问题出在: patch_model_for_fused_attention")
    print("     → patch修改了attention计算逻辑，导致结果变化")
elif with_patch_acc == 100:
    print("  ✅ 所有配置准确率正常!")
    print("     → 之前的准确率问题可能是测试设计问题")

# 检查feature逐个开启
first_fail = None
for r in feature_results:
    if not r['correct'] and r['error'] is None:
        first_fail = r['label']
        break
if first_fail:
    print(f"  ⚠️  功能 '{first_fail}' 首次导致准确率下降")

# 检查错误
errors = [r for r in feature_results if r['error']]
if errors:
    print(f"  ⚠️  有 {len(errors)} 个配置出现运行错误:")
    for r in errors:
        print(f"     - {r['label']}: {r['error'][:80]}")

print("=" * 75)
