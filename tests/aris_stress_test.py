import torch
import time
import sys
import os
import gc
import json
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig, LogitsProcessor, LogitsProcessorList

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from src.memory.cache import HeteroTransientCache


class MemoryProbe(LogitsProcessor):
    def __init__(self, device, name, base_mem, cache=None):
        self.step = 0
        self.device = device
        self.name = name
        self.base_mem = base_mem
        self.cache = cache
        self.prefill_peak = 0.0
        self.decode_peaks = []
        self.ttft = None
        self.tpot_list = []
        self.start_time = None

    def __call__(self, input_ids, scores):
        current_time = time.perf_counter()
        self.step += 1

        if self.step == 1:
            self.prefill_peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.start_time:
                self.ttft = current_time - self.start_time

        dyn_mem = (torch.cuda.memory_allocated(self.device) / 1024 ** 3) - self.base_mem
        self.decode_peaks.append(dyn_mem)

        if self.step > 1 and len(self.tpot_list) < self.step - 1:
            self.tpot_list.append(current_time - self.last_step_time)
        self.last_step_time = current_time

        return scores

    @property
    def avg_tpot(self):
        return sum(self.tpot_list) / len(self.tpot_list) if self.tpot_list else 0


def run_stress_test(input_len=32000, sink_tokens=64, keep_tail=8192, device="cuda:0", model_path="./models/Qwen2-VL-7B"):
    print(f"\n{'=' * 80}")
    print(f"🚀 Stress Test: input_len={input_len}, sink={sink_tokens}, tail={keep_tail}")
    print(f"{'=' * 80}")

    try:
        config = AutoConfig.from_pretrained(model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)

        original_forward = model.lm_head.forward

        def memory_safe_lm_head_forward(hidden_states):
            return original_forward(hidden_states[:, -1:, :])

        model.lm_head.forward = memory_safe_lm_head_forward

        print(f"[构建输入] 目标: {input_len} tokens")
        bg_sentence = "Background sentence for long context testing. "
        bg_tokens = processor.tokenizer(bg_sentence, return_tensors="pt").input_ids[0].to(device)
        num_repeats = input_len // len(bg_tokens)
        bg_input = bg_tokens.repeat(num_repeats).unsqueeze(0)
        actual_len = bg_input.shape[1]

        needle_tokens = processor.tokenizer(" Secret code: HETERO_KV_2025. ", return_tensors="pt").input_ids.to(device)
        question_tokens = processor.tokenizer(" What is the secret code? The code is: ", return_tensors="pt").input_ids.to(device)

        insert_idx = max(0, actual_len - 2000)
        input_ids = torch.cat([bg_input[:, :insert_idx], needle_tokens, bg_input[:, insert_idx:], question_tokens], dim=1)
        final_input_len = input_ids.shape[1]
        print(f"[输入就绪] 实际长度: {final_input_len} tokens")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated(device) / 1024 ** 3
        print(f"[基准显存] 模型加载后: {base_mem:.2f} GB")

        cache = HeteroTransientCache(sink_tokens=sink_tokens, keep_tail=keep_tail)
        probe = MemoryProbe(device, "Hetero", base_mem, cache)
        probe.start_time = time.perf_counter()

        try:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=15,
                    past_key_values=cache,
                    logits_processor=LogitsProcessorList([probe])
                )
            decode_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            resp = processor.batch_decode([out[final_input_len:] for out in outputs], skip_special_tokens=True)[0]

            success = "HETERO_KV" in resp or "2025" in resp

            result = {
                "input_len": final_input_len,
                "success": success,
                "prefill_peak_gb": float(probe.prefill_peak),
                "decode_peak_gb": float(decode_peak),
                "base_mem_gb": float(base_mem),
                "ttft_s": float(probe.ttft) if probe.ttft else None,
                "avg_tpot_ms": float(probe.avg_tpot * 1000) if probe.avg_tpot else None,
                "response": resp,
                "sink_tokens": sink_tokens,
                "keep_tail": keep_tail,
                "timestamp": datetime.now().isoformat()
            }

            print(f"\n✅ 测试成功!")
            print(f"   回答: {resp}")
            print(f"   Prefill 尖峰: {probe.prefill_peak:.2f} GB")
            print(f"   Decode 峰值: {decode_peak:.2f} GB")
            print(f"   纯动态 KV: {decode_peak - base_mem:.2f} GB")
            if probe.ttft:
                print(f"   TTFT: {probe.ttft:.3f} s")
            if probe.avg_tpot:
                print(f"   TPOT: {probe.avg_tpot * 1000:.2f} ms/token")

            return result

        except torch.OutOfMemoryError as e:
            print(f"\n💥 OOM 崩溃 at input_len={input_len}")
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(f"   崩溃前峰值: {peak:.2f} GB")
            return {
                "input_len": input_len,
                "success": False,
                "error": "OOM",
                "peak_gb": float(peak),
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        print(f"\n💥 异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            "input_len": input_len,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def main():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"

    results = []
    target_lens = [32000, 37000, 42000, 45000, 50000]

    print("\n" + "=" * 80)
    print("🏆 ARIS Mission - 16GB VRAM Frontier Stress Test")
    print("=" * 80)

    for input_len in target_lens:
        result = run_stress_test(
            input_len=input_len,
            sink_tokens=64,
            keep_tail=8192,
            device=device,
            model_path=model_path
        )
        results.append(result)

        if not result.get("success", False):
            if result.get("error") == "OOM":
                print(f"\n⚠️  检测到 OOM，尝试优化...")
                print("   策略: 减小 keep_tail 10%")
                keep_tail = int(8192 * 0.9)
                result_retry = run_stress_test(
                    input_len=input_len,
                    sink_tokens=64,
                    keep_tail=keep_tail,
                    device=device,
                    model_path=model_path
                )
                result_retry["note"] = f"Retry with keep_tail={keep_tail}"
                results.append(result_retry)
                if result_retry.get("success", False):
                    continue
            break

    print("\n" + "=" * 80)
    print("📊 实验报告")
    print("=" * 80)

    for r in results:
        status = "✅" if r.get("success", False) else "💥"
        ilen = r.get("input_len", 0)
        note = r.get("note", "")
        if r.get("success", False):
            decode = r.get("decode_peak_gb", 0)
            print(f"{status} {ilen:>6} tokens | Decode: {decode:.2f} GB {note}")
        else:
            err = r.get("error", "Unknown")
            print(f"{status} {ilen:>6} tokens | Error: {err} {note}")

    log_file = os.path.join(project_root, "experiments", "log.md")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {datetime.now().isoformat()}\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
        f.write("\n```\n")
    print(f"\n📝 详细日志已保存至: {log_file}")


if __name__ == "__main__":
    main()
