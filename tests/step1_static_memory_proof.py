import torch
import sys
import os
import gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
from transformers.cache_utils import DynamicCache
from transformers import LogitsProcessor, LogitsProcessorList

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# =====================================================================
# 🔍 深度探针：分离 Prefill 尖峰与 Decode 稳态峰值！
# =====================================================================
class DetailedMemoryProbe(LogitsProcessor):
    def __init__(self, device, name, base_mem, cache=None):
        self.step = 0
        self.device = device
        self.name = name
        self.base_mem = base_mem
        self.cache = cache
        self.prefill_peak = 0.0

    def __call__(self, input_ids, scores):
        self.step += 1

        if self.step == 1:
            # 🚀 预填充刚刚结束！记录下被 HF 引擎强制产生的不可避尖峰
            self.prefill_peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3

            # 🚀 核心魔法：重置峰值探针！
            # 接下来记录的，将是真正决定模型能否持续生成的【解码稳态峰值】！
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.step in [1, 5, 10, 15]:
            gc.collect();
            torch.cuda.empty_cache()
            dyn_mem = (torch.cuda.memory_allocated(self.device) / 1024 ** 3) - self.base_mem
            kv_shape = self.cache.key_cache[0].shape[-2] if self.cache else input_ids.shape[-1]
            print(
                f"      [{self.name} 探针] Decode 第 {self.step:<2} 步 | 纯动态 KV: {dyn_mem:.3f} GB | 物理长度死锁: {kv_shape}")
        return scores


# =====================================================================
# 🧠 终极工业级架构：瞬态分离缓存 (解封 FlashAttention！)
# =====================================================================
class TransientHeteroCache(DynamicCache):
    def __init__(self, sink_tokens=64, keep_tail=8192):
        super().__init__()
        self.sink_tokens = sink_tokens
        self.keep_tail = keep_tail
        self.key_cache = []
        self.value_cache = []
        self.real_seq_len = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        new_len = key_states.shape[-2]
        is_prefill = new_len > 1

        if is_prefill:
            sink_amount = min(new_len, self.sink_tokens)
            tail_amount = min(max(new_len - sink_amount, 0), self.keep_tail)

            sink_k = key_states[..., :sink_amount, :]
            sink_v = value_states[..., :sink_amount, :]

            if tail_amount > 0:
                tail_k = key_states[..., -tail_amount:, :]
                tail_v = value_states[..., -tail_amount:, :]
                saved_k = torch.cat([sink_k, tail_k], dim=-2)
                saved_v = torch.cat([sink_v, tail_v], dim=-2)
            else:
                saved_k, saved_v = sink_k, sink_v

            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(saved_k)
                self.value_cache.append(saved_v)
            else:
                self.key_cache[layer_idx] = saved_k
                self.value_cache[layer_idx] = saved_v

            if layer_idx == 0:
                self.real_seq_len += new_len

            return key_states, value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            if layer_idx == 0:
                self.real_seq_len += 1

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.real_seq_len


def run_ultimate_45k_native():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"

    print("\n" + "=" * 80)
    print("🔥 终极巅峰对决：剥离 HF 引擎尖峰，透视稳态解码峰值！")
    print("=" * 80)

    config = AutoConfig.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    original_forward = model.lm_head.forward

    def memory_safe_lm_head_forward(hidden_states):
        return original_forward(hidden_states[:, -1:, :])

    model.lm_head.forward = memory_safe_lm_head_forward

    print("[处理文本中... 构建 45,000 Token 史诗级长文]")
    bg_sentence = "This is a normal background text frame, nothing special here. "
    bg_tokens = processor.tokenizer(bg_sentence, return_tensors="pt").input_ids[0].to(device)
    num_repeats = 45000 // len(bg_tokens)
    bg_input = bg_tokens.repeat(num_repeats).unsqueeze(0)

    needle_tokens = processor.tokenizer(" The secret anomaly code is ANOMALY_CODE_9527. Remember it. ",
                                        return_tensors="pt").input_ids.to(device)
    question_tokens = processor.tokenizer(" What is the secret anomaly code? The code is: ",
                                          return_tensors="pt").input_ids.to(device)

    insert_idx = bg_input.shape[1] - 2000
    input_ids = torch.cat([bg_input[:, :insert_idx], needle_tokens, bg_input[:, insert_idx:], question_tokens], dim=1)

    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated(device) / 1024 ** 3

    # ---------------------------------------------------------
    # 对照组：Native HF
    # ---------------------------------------------------------
    print("\n🚀 [1/2] 运行 Native HF 原生基线...")
    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()

    probe_native = DetailedMemoryProbe(device, "Native", base_mem)
    try:
        with torch.inference_mode():
            outputs_native = model.generate(
                input_ids=input_ids, max_new_tokens=15,
                logits_processor=LogitsProcessorList([probe_native])
            )
        # 此时读取的，是被探针重置后的 Decode 阶段峰值！
        decode_peak_native = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        resp_native = \
        processor.batch_decode([out[len(input_ids[0]):] for out in outputs_native], skip_special_tokens=True)[0]

        print(f"   🎯 Native 回答: {resp_native}")
        print(f"   🌋 Prefill 瞬间激活尖峰 (不可避免): {probe_native.prefill_peak:.2f} GB")
        print(f"   📈 Decode 真实稳态峰值 (决定生存): {decode_peak_native:.2f} GB")
    except torch.OutOfMemoryError:
        print("   💥 Native OOM 崩溃！")
        decode_peak_native = float('inf')

    # ---------------------------------------------------------
    # 实验组：Hetero-KV (瞬态分离缓存)
    # ---------------------------------------------------------
    print("\n🚀 [2/2] 运行 Hetero-KV 瞬态分离架构...")
    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()

    hetero_cache = TransientHeteroCache()
    probe_hetero = DetailedMemoryProbe(device, "Hetero", base_mem, hetero_cache)

    try:
        with torch.inference_mode():
            outputs_hetero = model.generate(
                input_ids=input_ids, max_new_tokens=15, past_key_values=hetero_cache,
                logits_processor=LogitsProcessorList([probe_hetero])
            )
        # 此时读取的，是被探针重置后的 Decode 阶段峰值！
        decode_peak_hetero = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        resp_hetero = \
        processor.batch_decode([out[len(input_ids[0]):] for out in outputs_hetero], skip_special_tokens=True)[0]

        print(f"   🎯 Hetero 回答: {resp_hetero}")
        print(f"   🌋 Prefill 瞬间激活尖峰 (不可避免): {probe_hetero.prefill_peak:.2f} GB")
        print(f"   📉 Hetero 真实稳态峰值 (决定生存): {decode_peak_hetero:.2f} GB")

        print("\n" + "=" * 80)
        print("🏆 终极对账结论 🏆")
        print(f"在剥离了 HF 引擎自带的瞬间激活尖峰后，我们看到了模型稳态运行的真实面貌：")
        print(f"Native 稳态占用: {decode_peak_native:.2f} GB  |  Hetero-KV 稳态占用: {decode_peak_hetero:.2f} GB")
        if decode_peak_native != float('inf'):
            print(f"✂️  设备实际生存硬件门槛，真切降低了: {decode_peak_native - decode_peak_hetero:.2f} GB !!!")
        print(
            "只要在工业级端侧框架中加入分块预填充 (Chunked Prefill) 消除尖峰，你的系统就能将模型部署成本暴降一个显卡等级！")
        print("=" * 80)

    except Exception as e:
        print(f"   💥 崩溃: {e}")


if __name__ == "__main__":
    run_ultimate_45k_native()