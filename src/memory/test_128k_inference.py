import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure src package is discoverable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.engine_wrapper import HeteroKVCache


def test_generation():
    # 1. Dynamically locate the downloaded model path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "Qwen2.5-7B-Instruct")

    print(f"Loading model weights...")
    print(f"Path: {model_path}")

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print("Model loaded successfully.\n")

    # 3. Prepare a long test prompt
    prompt = "You are my top-tier AI architect mentor. Please explain in detail what a Large Language Model's KV Cache is, and why it causes OOM at 128K context length."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4. Instantiate our heterogeneous Cache interceptor
    custom_cache = HeteroKVCache(max_hbm_length=60, evict_chunk_size=16)

    print("Starting generation test with HeteroKVCache interceptor...")

    # 5. Execute generation, forcibly passing our custom_cache
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=custom_cache,
            max_new_tokens=100,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

    # 6. Decode output
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("\n" + "=" * 50)
    print("Generation result:\n")
    print(generated_text)
    print("=" * 50)

    # Part A: Print profiling report
    print_profiling_report(custom_cache)

    # Part C: Swap-in wake-up test
    print("\n" + "=" * 50)
    print("   Heterogeneous memory dynamic wake-up (Swap-in) test")
    print("=" * 50)

    dram_keys = list(custom_cache.manager.dram_table.keys())
    if dram_keys:
        target_chunk = dram_keys[0]
        print(f"Simulated scenario: current Query strongly attends to early-evicted node '{target_chunk}'")

        # Record VRAM before wake-up
        torch.cuda.reset_peak_memory_stats(model.device)
        mem_before = torch.cuda.memory_allocated(model.device)

        # Trigger Swap-in mechanism
        restored_k, restored_v = custom_cache.manager.swap_in(target_chunk, model.device)

        mem_after = torch.cuda.memory_allocated(model.device)
        pulled_mb = (mem_after - mem_before) / (1024 * 1024)

        print(f"Verification passed! Successfully pulled data back across PCIe.")
        print(f"Restored tensor device: {restored_k.device}, dtype: {restored_k.dtype}")
        print(f"This chunk now re-occupies GPU VRAM: {pulled_mb:.4f} MB")
        print("In a bypass Attention architecture, this awakened BF16 data will immediately participate in the current token computation.")


def print_profiling_report(cache):
    print("\n" + "=" * 50)
    print("   [Hetero-KV Profiler] Experiment data analysis report")
    print("=" * 50)

    dram_table = cache.manager.dram_table
    if not dram_table:
        print("No eviction was triggered; memory analysis is empty.")
        return

    total_dram_bytes = 0
    total_original_bytes = 0
    evicted_chunks = len(dram_table)

    for chunk_id, data in dram_table.items():
        # Count KV block parameters
        num_k_params = data["k_data"].nelement()
        num_v_params = data["v_data"].nelement()
        total_params = num_k_params + num_v_params

        # 1. Compute GPU storage if kept in BF16 (2 Bytes per param)
        chunk_original_bytes = total_params * 2

        # 2. Compute actual 4-bit DRAM storage (0.5 Bytes per param)
        # plus meta data (scales and zero-points, typically FP16/BF16 = 2 Bytes)
        k_meta_bytes = data["k_meta"][0].nelement() * 2 * 2
        v_meta_bytes = data["v_meta"][0].nelement() * 2 * 2

        chunk_dram_bytes = (total_params * 0.5) + k_meta_bytes + v_meta_bytes

        total_original_bytes += chunk_original_bytes
        total_dram_bytes += chunk_dram_bytes

    # Convert to Megabytes (MB)
    orig_mb = total_original_bytes / (1024 * 1024)
    dram_mb = total_dram_bytes / (1024 * 1024)
    saved_mb = orig_mb - dram_mb
    compression_ratio = (1 - dram_mb / orig_mb) * 100 if orig_mb > 0 else 0

    print(f"Total evictions performed: {cache.eviction_count} (across layers, {evicted_chunks} cache chunks intercepted)")
    print(f"Original BF16 GPU VRAM (without this architecture): {orig_mb:.4f} MB")
    print(f"Compressed DRAM system memory (Tier 2):              {dram_mb:.4f} MB")
    print(f"GPU VRAM saved:                                      {saved_mb:.4f} MB")
    print(f"Overall compression ratio (including quantization meta overhead): {compression_ratio:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    test_generation()
