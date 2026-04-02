import torch
from transformers.cache_utils import DynamicCache


class HeteroTransientCache(DynamicCache):
    """
    Hetero-KVCache-Optimizer 核心缓存控制器 (瞬态分离架构)

    设计理念：
    在大模型的长文本 Prefill（预填充）阶段，如果直接切断 KV Cache 会导致
    FlashAttention 算子因维度不匹配而降级为 O(N^2) 的 Math 矩阵乘法，从而引发
    数十 GB 的不可避免的显存 OOM。

    本架构采用“瞬态抽离”机制：
    1. 预填充时：拦截并保存首尾关键 Token (Sink + Tail)，但向原生引擎全量退还张量，
       骗过并解封 FlashAttention，计算完毕后依靠 Python GC 自动回收巨量废弃激活值。
    2. 解码时：只依靠存活下来的极小物理池进行拼接，将稳态峰值显存锁定为常数 O(1)。
    """

    def __init__(self, manager=None, sink_tokens=64, keep_tail=8192):
        super().__init__()
        self.manager = manager  # 预留给你的 C++ HBM 调度器接口
        self.sink_tokens = sink_tokens
        self.keep_tail = keep_tail
        self.key_cache = []
        self.value_cache = []
        self.real_seq_len = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        new_len = key_states.shape[-2]
        is_prefill = new_len > 1

        if is_prefill:
            # 🚀 瞬态抽离：保存地基 (Sink) 与 近期上下文 (Tail)
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

            # 写入内部物理池
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(saved_k)
                self.value_cache.append(saved_v)
            else:
                self.key_cache[layer_idx] = saved_k
                self.value_cache[layer_idx] = saved_v

            if layer_idx == 0:
                self.real_seq_len += new_len

            # 瞒天过海：全量退还给 HF，保持 FlashAttention 满血运行
            return key_states, value_states
        else:
            # 🚀 解码驻留：新字拼接入物理池，并执行原地滚动 (In-place Rolling)
            k_cache = self.key_cache[layer_idx]
            v_cache = self.value_cache[layer_idx]

            # 拼接新 Token
            new_k = torch.cat([k_cache, key_states], dim=-2)
            new_v = torch.cat([v_cache, value_states], dim=-2)

            # 检查尾部是否超出容量，如果超出则执行滚动
            max_len = self.sink_tokens + self.keep_tail
            if new_k.shape[-2] > max_len:
                # Ensure slices are handled without exceeding bounds.
                sink_k = new_k[..., :self.sink_tokens, :]
                tail_k = new_k[..., self.sink_tokens:, :]
                rolled_tail_k = tail_k[..., -self.keep_tail:, :] if tail_k.shape[-2] > self.keep_tail else tail_k
                new_k = torch.cat([sink_k, rolled_tail_k], dim=-2)

                sink_v = new_v[..., :self.sink_tokens, :]
                tail_v = new_v[..., self.sink_tokens:, :]
                rolled_tail_v = tail_v[..., -self.keep_tail:, :] if tail_v.shape[-2] > self.keep_tail else tail_v
                new_v = torch.cat([sink_v, rolled_tail_v], dim=-2)

            self.key_cache[layer_idx] = new_k
            self.value_cache[layer_idx] = new_v

            if layer_idx == 0:
                self.real_seq_len += 1

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        # 欺骗原生引擎的 RoPE 编码器，保持长序列相对位置的数学对齐
        return self.real_seq_len
