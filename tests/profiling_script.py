import torch
import os
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig, LogitsProcessorList
from tests.run_final_showcase import ShowcaseMemoryProbe

from src.memory.cache import HeteroTransientCache

def profile_throughput_and_latency(model_path, device="cuda:0"):
    def display_profile_results(probe, decode_peak, resp_str):
        print(f"   - 首字延迟 (TTFT): {probe.ttft:.3f} s")
        print(f"   - 吞吐性能 (TPOT - Time Per Output Token): {probe.avg_tpot * 1000:.2f} ms/token")
        print(f"   - 解码峰值 VRAM: {decode_peak:.2f} GB")
        print(f"   - 输出: {resp_str}")

    # Preparing environment
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    config = AutoConfig.from_pretrained(model_path)
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

    def display_profile_results(probe, decode_peak, resp_str):
        print(f"   - 首字延迟 (TTFT): {probe.ttft:.3f} s")
        print(f"   - 吞吐性能 (TPOT - Time Per Output Token): {probe.avg_tpot * 1000:.2f} ms/token")
        print(f"   - 解码峰值 VRAM: {decode_peak:.2f} GB")
        print(f"   - 输出: {resp_str}")

    # Preparing environment
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    config = AutoConfig.from_pretrained(model_path)
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

    # Profile with different configurations
    for use_cache in [True, False]:
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
            display_profile_results(probe, decode_peak, resp)
        except Exception as e:
            print(f"   💥 {('Hetero' if use_cache else 'Native')} 崩溃: {e}")
            display_profile_results(probe, float('inf'), "CRASHED")

if __name__ == "__main__":
    profile_throughput_and_latency("./models/Qwen2-VL-7B", device="cuda:3")

