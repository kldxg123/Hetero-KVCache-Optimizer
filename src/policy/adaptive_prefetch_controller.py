"""
src/policy/adaptive_prefetch_controller.py
==========================================
AdaptivePrefetchController: dynamically adjusts the prefetch window w
based on real-time decode-sequence attention heat volatility.

Core Formula:
  w_t = w_min + clip( (σ(A_t) / σ_ref - 1) · α, -Δ_max, Δ_max )
       + β · miss_rate_t

  where:
    σ(A_t)       = std of the most recent attention weight vector (heat volatility)
    σ_ref        = running EMA of σ(A) (reference baseline)
    α            = volatility sensitivity coefficient
    β            = cache-miss penalty coefficient
    w_min        = minimum prefetch window (always prefetch at least this many)
    Δ_max        = max single-step adjustment cap (smooth ramp)
    miss_rate_t  = fraction of swap-in requests that were cache misses (recent window)

Design rationale:
  When attention heat is highly volatile (σ(A_t) >> σ_ref), the query token is
  attending broadly, suggesting a regime change. A wider prefetch window improves
  hit rate under such uncertainty. Conversely, when σ(A_t) ≈ σ_ref (stable),
  a narrower window suffices, reducing wasted PCIe bandwidth.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


class AdaptivePrefetchController:
    """
    Dynamically adjusts the prefetch window w based on attention heat volatility
    and cache-miss feedback to optimize compute-communication overlap.

    Integrates with the existing PredictivePrefetchScheduler by overriding
    its lookahead_window parameter at each decode step.
    """

    def __init__(
        self,
        w_min: int = 2,
        w_max: int = 8,
        alpha: float = 1.5,
        beta: float = 0.5,
        delta_max: float = 2.0,
        ema_decay: float = 0.9,
        miss_window_size: int = 16,
    ):
        """
        Args:
            w_min: minimum prefetch window (safety floor).
            w_max: maximum prefetch window (PCIe bandwidth cap).
            alpha: volatility sensitivity — how aggressively w tracks σ deviation.
            beta: cache-miss penalty — how much miss rate inflates w.
            delta_max: max single-step adjustment magnitude (smoothness).
            ema_decay: decay factor for the EMA reference σ_ref.
            miss_window_size: number of recent decode steps to track for miss rate.
        """
        self.w_min = w_min
        self.w_max = w_max
        self.alpha = alpha
        self.beta = beta
        self.delta_max = delta_max
        self.ema_decay = ema_decay
        self.miss_window_size = miss_window_size

        # State
        self._current_w: float = float(w_min)
        self._sigma_ref: float = 0.0
        self._sigma_ref_initialized: bool = False
        self._miss_history: List[bool] = []  # True = cache miss
        self._step: int = 0

    def compute_window(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        cache_miss: bool = False,
    ) -> int:
        """
        Compute the adaptive prefetch window for the current decode step.

        Args:
            attention_weights: [seq_len] attention weights from the latest query.
            cache_miss: whether the current decode step suffered a swap-in cache miss.

        Returns:
            The integer prefetch window w to use for this step.
        """
        self._step += 1

        # Track cache misses
        self._miss_history.append(cache_miss)
        if len(self._miss_history) > self.miss_window_size:
            self._miss_history.pop(0)

        miss_rate = sum(self._miss_history) / len(self._miss_history) if self._miss_history else 0.0

        # Compute volatility adjustment
        volatility_delta = 0.0
        if attention_weights is not None and attention_weights.numel() > 1:
            sigma_t = float(attention_weights.detach().cpu().std())

            if not self._sigma_ref_initialized:
                self._sigma_ref = sigma_t
                self._sigma_ref_initialized = True
            else:
                # EMA update of reference baseline
                self._sigma_ref = self.ema_decay * self._sigma_ref + (1 - self.ema_decay) * sigma_t

            # Normalized volatility deviation
            if self._sigma_ref > 1e-8:
                normalized_deviation = (sigma_t / self._sigma_ref - 1.0)
            else:
                normalized_deviation = 0.0

            # Smooth clamped adjustment
            volatility_delta = max(-self.delta_max, min(self.delta_max, normalized_deviation * self.alpha))

        # Combine: base = w_min + volatility + miss penalty
        raw_w = self.w_min + volatility_delta + self.beta * miss_rate * self.w_max

        # Clamp to [w_min, w_max] and convert to integer
        w_new = max(self.w_min, min(self.w_max, round(raw_w)))

        # Smooth the transition (don't jump more than 1 per step)
        if abs(w_new - self._current_w) > 1:
            w_new = int(self._current_w + math.copysign(1, w_new - self._current_w))

        self._current_w = float(w_new)
        return int(w_new)

    @property
    def stats(self) -> Dict[str, float]:
        miss_rate = sum(self._miss_history) / len(self._miss_history) if self._miss_history else 0.0
        return {
            "current_w": self._current_w,
            "sigma_ref": self._sigma_ref,
            "miss_rate": miss_rate,
            "step": self._step,
        }
