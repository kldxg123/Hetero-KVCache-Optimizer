import torch
import time
import sys
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig, LogitsProcessorList
from tests.run_final_showcase import ShowcaseMemoryProbe
from transformers import LogitsProcessorList

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.cache import HeteroTransientCache


def profile_transient_cache_effect(device, model_path):
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    # Prepare test input of 45k tokens
    bg_sentence = "Background sentence for long input testing. "
    bg_tokens = processor.tokenizer(bg_sentence, return_tensors="pt").input_ids[0].to(device)
    num_repeats = 45000 // len(bg_tokens)
    bg_input = bg_tokens.repeat(num_repeats).unsqueeze(0)

    needle_tokens = processor.tokenizer(" Code: ABC_123_CODE.", return_tensors="pt").input_ids.to(device)
    question_tokens = processor.tokenizer(" What is the code? The code is: ", return_tensors="pt").input_ids.to(device)

    insert_idx = bg_input.shape[1] - 2000
    input_ids = torch.cat([bg_input[:, :insert_idx], needle_tokens, bg_input[:, insert_idx:], question_tokens], dim=1)


    for use_cache in [True, False]:
        print("\n[With Transient Cache]" if use_cache else "\n[Without Transient Cache]")
        cache = HeteroTransientCache(sink_tokens=64, keep_tail=8192) if use_cache else None
        probe = ShowcaseMemoryProbe(device, "Hetero" if use_cache else "Native", base_mem=None, cache=cache)

        try:
            with torch.inference_mode():
                start_time = time.perf_counter()
                outputs = model.generate(
                    input_ids=input_ids, max_new_tokens=15, past_key_values=cache,
                    logits_processor=LogitsProcessorList([probe])
                )
                probe.calculate_latency(start_time)
            decode_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            resp = processor.batch_decode([out[len(input_ids[0]):] for out in outputs], skip_special_tokens=True)[0]
            print(f"   - 首字延迟 (TTFT): {probe.ttft:.3f} s")
            print(f"   - 吞吐性能 (TPOT - Time Per Output Token): {probe.avg_tpot * 1000:.2f} ms/token")
            print(f"   - 解码峰值 VRAM: {decode_peak:.2f} GB")
            print(f"   - 输出: {resp}")
        except Exception as e:
            print(f"   💥 {('Hetero' if use_cache else 'Native')} 崩溃: {e}")
            print(f"   - 解码峰值 VRAM: float('inf') GB")
            print(f"   - 输出: CRASHED")

        cache = HeteroTransientCache(sink_tokens=64, keep_tail=8192) if use_cache else None
        cache = HeteroTransientCache(sink_tokens=64, keep_tail=8192) if use_cache else None
        probe = ShowcaseMemoryProbe(device, "Hetero" if use_cache else "Native", base_mem=None, cache=cache)
profile_transient_cache_effect("./models/Qwen2-VL-7B", device="cuda:0")

try:
            with torch.inference_mode():
                start_time = time.perf_counter()
                outputs = model.generate(
                    input_ids=input_ids, max_new_tokens=15, past_key_values=cache,
                    logits_processor=LogitsProcessorList([probe])
                )
                probe.calculate_latency(start_time)
            decode_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            resp = processor.batch_decode([out[len(input_ids[0]):] for out in outputs], skip_special_tokens=True)[0]
            
            print("\n[With Transient Cache]" if use_cache else "\n[Without Transient Cache]")
            print(f"   - Peak VRAM: {decode_peak:.2f} GB")
            print(f"   - TTFT: {probe.ttft:.3f} s")
            print(f"   - TPOT: {probe.avg_tpot * 1000:.2f} ms/token")
            print(f"   - Output: {resp}")
        except Exception as e:
            print(f"   💥 Error: {e}")



if __name__ == "__main__":
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"
    profile_transient_cache_effect(device, model_path)

