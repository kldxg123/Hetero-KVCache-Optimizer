"""
src/simulation/pcie_throttle_sim.py
=====================================
PCIe Bandwidth Throttle Simulator for Edge GPU Evaluation.

Provides deterministic bandwidth-delay injection for HBM <-> DRAM transfers,
allowing high-end server GPUs (e.g. A100) to approximate the PCIe transfer
latency of consumer edge cards (e.g. RTX 4060 Ti on PCIe Gen4 x16).

Usage:
    from src.simulation.pcie_throttle_sim import BandwidthLimiter
    limiter = BandwidthLimiter(max_bandwidth_gbps=32.0)  # ~PCIe Gen4 x8
    manager = HeteroKVManager(..., bandwidth_limiter=limiter)
"""

import time
import torch
from typing import Optional


class BandwidthLimiter:
    """
    Inject synthetic delays proportional to tensor transfer sizes to simulate
    a PCIe bandwidth bottleneck.
    """

    def __init__(self, max_bandwidth_gbps: float = 64.0):
        """
        Args:
            max_bandwidth_gbps: Simulated bidirectional PCIe bandwidth in GB/s.
                                Typical values:
                                  - A100 PCIe Gen4 x16: ~64 GB/s
                                  - RTX 4090 PCIe Gen4 x16: ~32 GB/s
                                  - RTX 4060 Ti PCIe Gen4 x8: ~16 GB/s
                                  - Laptop/mobile edge: ~8 GB/s
        """
        self.max_bandwidth_gbps = max_bandwidth_gbps

    def simulate_transfer(self, tensor: torch.Tensor) -> None:
        """
        Sleep for the amount of time it would take to move `tensor` across
        a link of capacity `max_bandwidth_gbps`.
        """
        bytes_moved = tensor.element_size() * tensor.numel()
        seconds = bytes_moved / (self.max_bandwidth_gbps * 1e9)
        if seconds > 0:
            time.sleep(seconds)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pass-through that injects delay. Returns the tensor unchanged.
        Useful for inline throttling: tensor = limiter(tensor.to(device)).
        """
        self.simulate_transfer(tensor)
        return tensor


class NVIDIAPowerLimiter:
    """
    Hardware-level power capping via nvidia-smi.
    This is a coarse-grained proxy for throttling that reduces boost clocks
    and can indirectly limit effective memory bandwidth.
    """

    def __init__(self, target_watts: int):
        self.target_watts = target_watts
        self.original_watts: Optional[int] = None

    def _get_current_pl(self, gpu_id: int = 0) -> Optional[int]:
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                text=True,
            )
            return int(float(out.strip()))
        except Exception:
            return None

    def apply(self, gpu_id: int = 0) -> bool:
        import subprocess
        cur = self._get_current_pl(gpu_id)
        if cur is None:
            print("[NVIDIAPowerLimiter] nvidia-smi not available or insufficient privileges.")
            return False
        self.original_watts = cur
        try:
            subprocess.check_call(
                ["nvidia-smi", "-i", str(gpu_id), "-pl", str(self.target_watts)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[NVIDIAPowerLimiter] GPU {gpu_id} power limit set to {self.target_watts}W (was {cur}W).")
            return True
        except subprocess.CalledProcessError:
            print("[NVIDIAPowerLimiter] Failed to set power limit (requires root or elevated privileges).")
            return False

    def restore(self, gpu_id: int = 0) -> bool:
        if self.original_watts is None:
            return False
        import subprocess
        try:
            subprocess.check_call(
                ["nvidia-smi", "-i", str(gpu_id), "-pl", str(self.original_watts)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[NVIDIAPowerLimiter] GPU {gpu_id} power limit restored to {self.original_watts}W.")
            return True
        except subprocess.CalledProcessError:
            return False


def demo():
    """Quick sanity check."""
    limiter = BandwidthLimiter(max_bandwidth_gbps=16.0)
    dummy = torch.randn(1024, 1024, dtype=torch.float16)
    print(f"Simulating transfer of {dummy.numel() * 2 / 1e6:.2f} MB at 16 GB/s...")
    t0 = time.time()
    limiter.simulate_transfer(dummy)
    print(f"Injected delay: {time.time() - t0:.4f}s")


if __name__ == "__main__":
    demo()
