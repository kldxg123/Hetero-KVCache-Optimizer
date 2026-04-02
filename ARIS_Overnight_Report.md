# ARIS Overnight Research Report: Hetero-KVCache-Optimizer

**Mission Date:** 2024-07-29
**Objective:** Conduct robustness review, performance stress-testing, and edge-case analysis of the `HeteroTransientCache` implementation.

---

## 1. Weakness Analysis & Code Fixes

### 1.1. Identified Weakness: Unbounded Cache Growth in Decoding

A critical vulnerability was identified in `src/memory/cache.py`. The original implementation of `HeteroTransientCache.update` only handled the prefill stage correctly. During the decoding stage (when `new_len == 1`), the logic simply appended new tokens to the cache using `torch.cat` without any size checks.

