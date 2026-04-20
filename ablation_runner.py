#!/usr/bin/env python3
"""
ablation_runner.py
==================
Hetero-KV 消融实验脚本 (Ablation Study) — 无人值守版
"""

import gc
import sys
import os
import time
import csv
import traceback
import torch
from typing import Optional, Tuple
from transformers.cache_utils import DynamicCache

project_root = "."
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.engine_wrapper import FusedHeteroCache, ChunkedPrefillEngine
from src.policy.heavy_hitter import HeavyHitterOracle

# ---------------------------------------------------------------------------
# 消融 Cache 子类
# ---------------------------------------------------------------------------

class AblationHeteroCache(FusedHeteroCache):
    def __init__(
        self,
        num_layers: int = 4,
        sink_tokens: int = 64,
        keep_tail: int = 8192,
        chunk_size: int = 2048,
        device: str = "cuda:0",
        group_size: int = 128,
        enable_quant: bool = True,
        enable_dram_offload: bool = False,
        use_hh: bool = False,
        enable_swapin: bool = True,
        enable_prefetch: bool = True,
        enable_triton: bool = True,
    ):
        super().__init__(
            num_layers=num_layers,
            sink_tokens=sink_tokens,
            keep_tail=keep_tail,
            chunk_size=chunk_size,
            device=device,
            group_size=group_size,
            enable_quant=enable_quant,
            enable_prefetch=enable_prefetch,
            enable_triton=enable_triton,
        )
        self.enable_dram_offload = enable_dram_offload
        self.use_hh = use_hh
        self.enable_swapin = enable_swapin
        self.retained_positions: list[Optional[torch.Tensor]] = []
        self.evicted_positions: dict[str, torch.Tensor] = {}
        self._dram_table: dict = {}
        self._eviction_counter = 0
        from src.quantization.kv_compressor import KVCompressor
        self.compressor = KVCompressor(group_size=group_size, bits=4)
        self._prefetcher = None
        # DynamicCache in newer transformers no longer exposes key_cache/value_cache lists;
        # we maintain our own for the ablation path.
        self.key_cache: list = []
        self.value_cache: list = []

        if self.use_hh:
            self.hh_oracle = HeavyHitterOracle(
                block_size=1,
                sink_tokens=sink_tokens,
                local_window=keep_tail // 2,
            )

    @property
    def dram_table(self):
        return self._dram_table

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override FusedHeteroCache.update to route through our ablation logic."""
        new_len = key_states.shape[-2]
        mode = "prefill" if new_len > 1 else "decode"
        if mode == "prefill":
            out_k, out_v = self._prefill_update(key_states, value_states, layer_idx)
        else:
            out_k, out_v = self._decode_update(key_states, value_states, layer_idx)
        if layer_idx == 0:
            self.real_seq_len += new_len
        return out_k, out_v

    def _prefill_update(self, key_states, value_states, layer_idx):
        new_len = key_states.shape[-2]
        max_hbm = self.sink_tokens + self.keep_tail

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.retained_positions.append(None)

        prev_len = self.real_seq_len if self.key_cache[layer_idx] is not None else 0

        if self.key_cache[layer_idx] is None:
            sink_amt = min(new_len, self.sink_tokens)
            tail_amt = min(new_len - sink_amt, self.keep_tail)
            k_sink = key_states[..., :sink_amt, :]
            v_sink = value_states[..., :sink_amt, :]
            positions = torch.arange(min(new_len, max_hbm), device=key_states.device, dtype=torch.long)

            if tail_amt > 0:
                k_tail = key_states[..., -tail_amt:, :]
                v_tail = value_states[..., -tail_amt:, :]
                self.key_cache[layer_idx] = torch.cat([k_sink, k_tail], dim=-2)
                self.value_cache[layer_idx] = torch.cat([v_sink, v_tail], dim=-2)
            else:
                self.key_cache[layer_idx] = k_sink
                self.value_cache[layer_idx] = v_sink
            self.retained_positions[layer_idx] = positions
        else:
            new_k = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_v = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            new_positions = torch.cat([
                self.retained_positions[layer_idx],
                torch.arange(prev_len, prev_len + new_len, device=key_states.device, dtype=torch.long),
            ])
            cur_len = new_k.shape[-2]

            if cur_len > max_hbm:
                overflow = cur_len - max_hbm

                if (self.use_hh and self.hh_oracle.token_scores is not None
                        and self.hh_oracle.token_scores.shape[0] >= cur_len):
                    evict_candidates = self.hh_oracle.get_eviction_candidates(cur_len, evict_num_blocks=overflow)
                    evict_indices = evict_candidates.to(device=new_k.device, dtype=torch.long)
                    keep_mask = torch.ones(cur_len, dtype=torch.bool, device=new_k.device)
                    keep_mask[evict_indices] = False
                    all_indices = torch.arange(cur_len, device=new_k.device)[keep_mask]
                else:
                    evict_start = self.sink_tokens
                    evict_end = evict_start + overflow
                    evict_indices = torch.arange(evict_start, evict_end, device=new_k.device)
                    all_indices = torch.cat([
                        torch.arange(self.sink_tokens, device=new_k.device),
                        torch.arange(cur_len - self.keep_tail, cur_len, device=new_k.device),
                    ])

                if evict_indices.numel() > 0:
                    k_evict = new_k[..., evict_indices, :]
                    v_evict = new_v[..., evict_indices, :]
                    pos_evict = new_positions[evict_indices]
                    if self.enable_quant:
                        self._evict_to_dram_with_pos(layer_idx, k_evict, v_evict, pos_evict)
                    elif self.enable_dram_offload:
                        self._evict_to_dram_fp16(layer_idx, k_evict, v_evict, pos_evict)

                self.key_cache[layer_idx] = new_k[..., all_indices, :]
                self.value_cache[layer_idx] = new_v[..., all_indices, :]
                self.retained_positions[layer_idx] = new_positions[all_indices]
            else:
                self.key_cache[layer_idx] = new_k
                self.value_cache[layer_idx] = new_v
                self.retained_positions[layer_idx] = new_positions

            del new_k, new_v

        if layer_idx == 0:
            self.real_seq_len += new_len

        torch.cuda.empty_cache()
        gc.collect()
        return key_states, value_states

    def _decode_update(self, key_states, value_states, layer_idx):
        max_hbm = self.sink_tokens + self.keep_tail
        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]
        new_pos = torch.tensor([self.real_seq_len], device=key_states.device, dtype=torch.long)

        full_k = torch.cat([k_cache, key_states], dim=-2)
        full_v = torch.cat([v_cache, value_states], dim=-2)
        full_positions = torch.cat([self.retained_positions[layer_idx], new_pos])

        if full_k.shape[-2] > max_hbm:
            total_len = full_k.shape[-2]
            if self.use_hh and self.hh_oracle.token_scores is not None:
                candidates = self.hh_oracle.get_eviction_candidates(total_len, 1)
                if candidates.numel() > 0:
                    evict_idx = candidates[0]
                    keep_mask = torch.ones(total_len, dtype=torch.bool, device=full_k.device)
                    keep_mask[evict_idx] = False
                    self.key_cache[layer_idx] = full_k[..., keep_mask, :]
                    self.value_cache[layer_idx] = full_v[..., keep_mask, :]
                    self.retained_positions[layer_idx] = full_positions[keep_mask]
                else:
                    self._fifo_decode_commit(full_k, full_v, full_positions, layer_idx)
            else:
                self._fifo_decode_commit(full_k, full_v, full_positions, layer_idx)
        else:
            self.key_cache[layer_idx] = full_k
            self.value_cache[layer_idx] = full_v
            self.retained_positions[layer_idx] = full_positions

        if layer_idx == 0:
            self.real_seq_len += 1
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _fifo_decode_commit(self, full_k, full_v, full_positions, layer_idx):
        sink_k = full_k[..., :self.sink_tokens, :]
        sink_v = full_v[..., :self.sink_tokens, :]
        old_tail_k = full_k[..., self.sink_tokens + 1:-1, :]
        old_tail_v = full_v[..., self.sink_tokens + 1:-1, :]
        self.key_cache[layer_idx] = torch.cat([sink_k, old_tail_k, full_k[..., -1:, :]], dim=-2)
        self.value_cache[layer_idx] = torch.cat([sink_v, old_tail_v, full_v[..., -1:, :]], dim=-2)
        self.retained_positions[layer_idx] = torch.cat([
            full_positions[:self.sink_tokens],
            full_positions[self.sink_tokens + 1:],
        ])

    def _evict_to_dram_with_pos(self, layer_idx, k_chunk, v_chunk, positions):
        chunk_key = f"l{layer_idx}_e{self._eviction_counter}"
        q_k, k_scales, k_zps = self.compressor.compress(k_chunk)
        q_v, v_scales, v_zps = self.compressor.compress(v_chunk)
        self.dram_table[chunk_key] = {
            "k_data": q_k.cpu().pin_memory(),
            "k_scales": k_scales.cpu().pin_memory(),
            "k_zps": k_zps.cpu().pin_memory(),
            "v_data": q_v.cpu().pin_memory(),
            "v_scales": v_scales.cpu().pin_memory(),
            "v_zps": v_zps.cpu().pin_memory(),
        }
        self.evicted_positions[chunk_key] = positions.cpu()
        if layer_idx == 0:
            self._eviction_counter += 1

    def _evict_to_dram_fp16(self, layer_idx, k_chunk, v_chunk, positions):
        chunk_key = f"l{layer_idx}_e{self._eviction_counter}"
        self.dram_table[chunk_key] = {
            "k_data": k_chunk.cpu().pin_memory(),
            "v_data": v_chunk.cpu().pin_memory(),
            "fp16": True,
        }
        self.evicted_positions[chunk_key] = positions.cpu()
        if layer_idx == 0:
            self._eviction_counter += 1

    def swap_in_chunk(self, chunk_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if chunk_key not in self.dram_table:
            return None
        entry = self.dram_table.pop(chunk_key)
        self.evicted_positions.pop(chunk_key, None)
        if entry.get("fp16"):
            k = entry["k_data"].to(self.device, non_blocking=True)
            v = entry["v_data"].to(self.device, non_blocking=True)
            torch.cuda.synchronize(self.device)
            return k, v
        if self._prefetcher is not None:
            result = self._prefetcher.fetch_if_ready(chunk_key)
            if result is not None:
                restored_k, _ = result
                q_v = entry["v_data"].to(self.device, non_blocking=True)
                s_v = entry["v_scales"].to(self.device, non_blocking=True)
                z_v = entry["v_zps"].to(self.device, non_blocking=True)
                torch.cuda.synchronize(self.device)
                restored_v = self.compressor.decompress(q_v, s_v, z_v).to(torch.bfloat16)
                return restored_k, restored_v
        q_k = entry["k_data"].to(self.device, non_blocking=True)
        s_k = entry["k_scales"].to(self.device, non_blocking=True)
        z_k = entry["k_zps"].to(self.device, non_blocking=True)
        q_v = entry["v_data"].to(self.device, non_blocking=True)
        s_v = entry["v_scales"].to(self.device, non_blocking=True)
        z_v = entry["v_zps"].to(self.device, non_blocking=True)
        torch.cuda.synchronize(self.device)
        restored_k = self.compressor.decompress(q_k, s_k, z_k).to(torch.bfloat16)
        restored_v = self.compressor.decompress(q_v, s_v, z_v).to(torch.bfloat16)
        return restored_k, restored_v


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM(torch.nn.Module):
    def __init__(self, num_layers: int = 4, num_heads: int = 8, head_dim: int = 128, hidden_dim: int = 1024):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        hidden_states = torch.randn(batch, seq_len, self.hidden_dim, dtype=torch.bfloat16, device=device)
        for layer_idx in range(self.num_layers):
            key_states = torch.randn(batch, self.num_heads, seq_len, self.head_dim, dtype=torch.bfloat16, device=device)
            value_states = torch.randn(batch, self.num_heads, seq_len, self.head_dim, dtype=torch.bfloat16, device=device)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)
        if use_cache:
            out = type("Output", (), {})()
            out.last_hidden_state = hidden_states
            out.past_key_values = past_key_values
            return out
        return hidden_states


# ---------------------------------------------------------------------------
# Native Baseline 的 Prefill（绕过 ChunkedPrefillEngine 的 dram_table 日志）
# ---------------------------------------------------------------------------

def native_prefill(model, input_ids, cache, chunk_size):
    total_len = input_ids.shape[-1]
    device = input_ids.device
    print(f"[NativePrefill] total_tokens={total_len} chunk_size={chunk_size}")
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        chunk_ids = input_ids[:, start:end]
        chunk_pos = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)
        model(input_ids=chunk_ids, past_key_values=cache, use_cache=True, position_ids=chunk_pos)
        mem_gb = torch.cuda.memory_allocated(device) / 1024 ** 3
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f"  chunk [{start:>6}:{end:>6}] current={mem_gb:.2f}GB peak={peak_gb:.2f}GB")
    print(f"[NativePrefill] 完成. seq_len={total_len}")


# ---------------------------------------------------------------------------
# 消融实验主控
# ---------------------------------------------------------------------------

class AblationStudy:
    def __init__(
        self,
        seq_len: int = 128_000,
        chunk_size: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        head_dim: int = 128,
        device: str = "cuda",
    ):
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.model = MockLLM(num_layers, num_heads, head_dim).to(device)

    def _make_native_cache(self):
        return DynamicCache()

    def _make_wo_swapin_cache(self):
        return AblationHeteroCache(
            num_layers=self.num_layers,
            sink_tokens=64, keep_tail=8192, chunk_size=self.chunk_size, device=self.device,
            enable_quant=True, use_hh=True, enable_swapin=False, enable_prefetch=False,
        )

    def _make_wo_quant_cache(self):
        return AblationHeteroCache(
            num_layers=self.num_layers,
            sink_tokens=64, keep_tail=8192, chunk_size=self.chunk_size, device=self.device,
            enable_quant=False, enable_dram_offload=True, use_hh=True, enable_swapin=True, enable_prefetch=False,
        )

    def _make_wo_hh_cache(self):
        return AblationHeteroCache(
            num_layers=self.num_layers,
            sink_tokens=64, keep_tail=8192, chunk_size=self.chunk_size, device=self.device,
            enable_quant=True, use_hh=False, enable_swapin=True, enable_prefetch=False,
        )

    def _make_full_cache(self):
        return AblationHeteroCache(
            num_layers=self.num_layers,
            sink_tokens=64, keep_tail=8192, chunk_size=self.chunk_size, device=self.device,
            enable_quant=True, use_hh=True, enable_swapin=True, enable_prefetch=True,
        )

    def _run_single(self, name: str, cache_factory):
        print(f"\n{'=' * 70}")
        print(f"  Configuration: {name}")
        print(f"{'=' * 70}")

        result = {
            "Configuration": name,
            "Peak VRAM (GB)": "Error/OOM",
            "Prefill Latency (s)": "Error/OOM",
            "TTFT (s)": "Error/OOM",
            "NIAH Recall (%)": "Error/OOM",
        }

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            cache = cache_factory()

            needle_pos = int(self.seq_len * 0.75)
            input_ids = torch.randint(0, 32000, (1, self.seq_len), device=self.device)

            if hasattr(cache, "hh_oracle") and cache.hh_oracle is not None:
                scores = torch.full((self.seq_len,), 0.05, device=self.device)
                scores[needle_pos] = 10.0
                cache.hh_oracle.token_scores = scores

            # Prefill
            t0 = time.time()
            if isinstance(cache, DynamicCache) and not isinstance(cache, FusedHeteroCache):
                native_prefill(self.model, input_ids, cache, self.chunk_size)
            else:
                engine = ChunkedPrefillEngine(self.model, cache, chunk_size=self.chunk_size)
                engine.prefill(input_ids)
            prefill_latency = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

            # Decode TTFT
            decode_input = torch.randint(0, 32000, (1, 1), device=self.device)
            t0 = time.time()
            with torch.no_grad():
                self.model(input_ids=decode_input, past_key_values=cache, use_cache=True)
            ttft = time.time() - t0

            recall = self._check_niah_recall(cache, needle_pos)

            result.update({
                "Peak VRAM (GB)": round(peak_vram, 3),
                "Prefill Latency (s)": round(prefill_latency, 2),
                "TTFT (s)": round(ttft, 4),
                "NIAH Recall (%)": round(recall, 1),
            })

            del cache, input_ids, decode_input
            torch.cuda.empty_cache()
            gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"[ERROR] OOM in {name}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Exception in {name}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

        return result

    def _check_niah_recall(self, cache, needle_pos: int) -> float:
        if isinstance(cache, DynamicCache) and not isinstance(cache, FusedHeteroCache):
            return 100.0
        if not isinstance(cache, AblationHeteroCache):
            return 0.0
        for layer_idx in range(self.num_layers):
            pos_tensor = cache.retained_positions[layer_idx]
            if pos_tensor is not None and needle_pos in pos_tensor.tolist():
                return 100.0
        for chunk_key, pos_tensor in cache.evicted_positions.items():
            if needle_pos in pos_tensor.tolist():
                return 100.0 if cache.enable_swapin else 0.0
        return 0.0

    def run(self):
        configs = [
            ("Full", self._make_full_cache),
            ("w/o_Quant", self._make_wo_quant_cache),
            ("w/o_HH", self._make_wo_hh_cache),
            ("w/o_SwapIn", self._make_wo_swapin_cache),
            ("Baseline_Native", self._make_native_cache),
        ]

        results = []
        for name, factory in configs:
            results.append(self._run_single(name, factory))

        self._print_markdown_table(results)
        self._save_report(results)
        return results

    def _print_markdown_table(self, results):
        headers = ["Configuration", "Peak VRAM (GB)", "Prefill Latency (s)", "TTFT (s)", "NIAH Recall (%)"]
        col_widths = [max(len(h), max(len(str(r[h])) for r in results)) for h in headers]

        def row_str(cells):
            return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)) + " |"

        sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

        print("\n" + "=" * 70)
        print("  Ablation Study Results")
        print("=" * 70 + "\n")
        print(row_str(headers))
        print(sep)
        for r in results:
            print(row_str([r[h] for h in headers]))
        print()

        print("\n[显存/延迟趋势分析]")
        try:
            valid = [r for r in results if str(r["Peak VRAM (GB)"]) != "Error/OOM"]
            if valid:
                by_vram = sorted(valid, key=lambda x: float(x["Peak VRAM (GB)"]))
                print(f"  - 最低 Peak VRAM: {by_vram[0]['Configuration']} ({by_vram[0]['Peak VRAM (GB)']} GB)")
                print(f"  - 最高 Peak VRAM: {by_vram[-1]['Configuration']} ({by_vram[-1]['Peak VRAM (GB)']} GB)")

                by_lat = sorted(valid, key=lambda x: float(x["Prefill Latency (s)"]))
                print(f"  - 最短 Prefill:   {by_lat[0]['Configuration']} ({by_lat[0]['Prefill Latency (s)']} s)")
                print(f"  - 最长 Prefill:   {by_lat[-1]['Configuration']} ({by_lat[-1]['Prefill Latency (s)']} s)")

                by_ttft = sorted(valid, key=lambda x: float(x["TTFT (s)"]))
                print(f"  - 最短 TTFT:      {by_ttft[0]['Configuration']} ({by_ttft[0]['TTFT (s)']} s)")
                print(f"  - 最长 TTFT:      {by_ttft[-1]['Configuration']} ({by_ttft[-1]['TTFT (s)']} s)")

                recall_100 = [r["Configuration"] for r in valid if float(r["NIAH Recall (%)"]) >= 99.9]
                recall_fail = [r["Configuration"] for r in valid if float(r["NIAH Recall (%)"]) < 99.9]
                if recall_100:
                    print(f"  - 100% NIAH Recall: {', '.join(recall_100)}")
                if recall_fail:
                    print(f"  - NIAH Recall <100%: {', '.join(recall_fail)}")
        except Exception as e:
            print(f"  [趋势分析跳过] {e}")
        print()

    def _save_report(self, results):
        report_path = "ablation_report.csv"
        try:
            with open(report_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "Configuration", "Peak VRAM (GB)", "Prefill Latency (s)", "TTFT (s)", "NIAH Recall (%)"
                ])
                writer.writeheader()
                writer.writerows(results)
            print(f"[报告已保存] {report_path}")
        except Exception as e:
            print(f"[报告保存失败] {e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = AblationStudy(
        seq_len=128_000,
        chunk_size=2048,
        num_layers=4,
        num_heads=8,
        head_dim=128,
        device=device,
    )
    study.run()


if __name__ == "__main__":
    main()
