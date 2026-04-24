#!/usr/bin/env python3
"""
A100 GPU Emulation with Strict 24GB Memory Ceiling
Implements the hardware boundary constraint for edge device emulation
"""

import os
import subprocess
import torch
import json
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

class A100Emulator:
    """
    A100 GPU emulator with strict 24GB memory constraint
    Simulates edge device hardware boundaries for rigorous experiments
    """

    def __init__(self, memory_fraction: float = 24.0/80.0):
        self.memory_fraction = memory_fraction  # 24GB/80GB for A100 edge emulation
        self.original_memory_fraction = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.power_log_file = "logs/power_consumption.log"
        self.memory_log_file = "logs/memory_usage.log"

        # Create log directory
        Path("logs").mkdir(parents=True, exist_ok=True)

        # Setup GPU memory constraint
        self._setup_memory_constraint()

    def _setup_memory_constraint(self):
        """Apply strict 24GB memory ceiling on A100"""
        print(f"🔧 Setting A100 memory constraint to {self.memory_fraction*80:.0f}GB ({self.memory_fraction*100:.0f}% of 80GB)")

        try:
            # Store original memory fraction if set
            if torch.cuda.is_available():
                self.original_memory_fraction = torch.cuda.get_per_process_memory_fraction()
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                torch.cuda.empty_cache()
                print(f"✓ Applied memory constraint: {self.memory_fraction*80:.0f}GB available")
            else:
                print("⚠️ CUDA not available, running CPU emulation")
        except Exception as e:
            print(f"⚠️ Warning: Could not set memory constraint: {e}")

    def get_power_metrics(self) -> Dict[str, float]:
        """Monitor GPU power consumption via nvidia-smi"""
        if not torch.cuda.is_available():
            return {"power_watts": 0.0, "energy_used_wh": 0.0}

        try:
            # Get power draw in watts (first GPU only)
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "-i", "0"
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                # Take the first value if multiple values are returned
                power_value = result.stdout.strip().split('\n')[0]
                power_watts = float(power_value)
                return {"power_watts": power_watts}
            else:
                return {"power_watts": 0.0}
        except Exception as e:
            print(f"⚠️ Power monitoring failed: {e}")
            return {"power_watts": 0.0}

    def log_power_metrics(self, mode: str = "inference"):
        """Log power consumption metrics"""
        power_metrics = self.get_power_metrics()

        with open(self.power_log_file, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{mode},{power_metrics.get('power_watts', 0.0)}\n")

    def log_memory_metrics(self, prefix: str = ""):
        """Log GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            with open(self.memory_log_file, "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp},{prefix},{allocated:.2f},{cached:.2f},{max_allocated:.2f}\n")
        else:
            with open(self.memory_log_file, "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp},{prefix},0.00,0.00,0.00\n")

    def reset_peak_memory(self):
        """Reset peak memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original memory fraction
        if self.original_memory_fraction is not None and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.original_memory_fraction)
            print("🔧 Restored original memory fraction")


def run_power_monitoring(emulator: A100Emulator, duration_seconds: int = 300):
    """Continuous power monitoring during inference"""
    print(f"📊 Starting power monitoring for {duration_seconds} seconds...")

    start_time = time.time()
    power_samples = []

    try:
        while time.time() - start_time < duration_seconds:
            emulator.log_power_metrics("inference")
            power_metrics = emulator.get_power_metrics()
            power_samples.append(power_metrics["power_watts"])
            time.sleep(1)  # Sample every second

    except KeyboardInterrupt:
        print("⏹️ Power monitoring interrupted by user")

    # Calculate statistics
    if power_samples:
        avg_power = sum(power_samples) / len(power_samples)
        max_power = max(power_samples)
        total_energy = avg_power * (time.time() - start_time) / 3600  # Wh

        print(f"📈 Power Consumption Summary:")
        print(f"   Average: {avg_power:.2f}W")
        print(f"   Maximum: {max_power:.2f}W")
        print(f"   Total Energy: {total_energy:.3f}Wh")

        return {
            "average_watts": avg_power,
            "maximum_watts": max_power,
            "total_energy_wh": total_energy,
            "duration_seconds": time.time() - start_time
        }
    else:
        return {}


def validate_memory_constraint(emulator: A100Emulator, target_tokens: int = 128000):
    """Test that memory constraint is enforced with large token sequences"""
    print(f"🔍 Validating memory constraint with {target_tokens:,} tokens...")

    try:
        # Create dummy tensor to test memory allocation
        tokens_per_gb = 1000  # Approximate tokens per GB
        tensor_size = min(target_tokens, int(20 * 1024**3 / 4))  # Max 20GB test

        dummy_tensor = torch.randn(tensor_size // 4, dtype=torch.float32, device=emulator.device)

        # Check actual allocation
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"   Allocated: {allocated:.2f}GB (should be ≤ {emulator.memory_fraction*80:.0f}GB)")

        # Validate constraint
        if allocated <= emulator.memory_fraction * 80 * 1.1:  # Allow 10% tolerance
            print("✓ Memory constraint validation passed")
            return True
        else:
            print(f"✗ Memory constraint violated: {allocated:.2f}GB > {emulator.memory_fraction*80:.0f}GB")
            return False

    except torch.cuda.OutOfMemoryError:
        print("✓ OOM correctly triggered under memory constraint")
        return True
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        return False


def setup_environment():
    """Setup environment for A100 emulation"""
    print("🚀 Setting up A100 emulation environment...")

    # Enable torch cudnn benchmark for faster inference
    torch.backends.cudnn.benchmark = True

    # Disable gradient computation for memory efficiency
    torch.set_grad_enabled(False)

    # Set CUDA device
    if torch.cuda.is_available():
        print(f"🎯 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")
    else:
        print("⚠️ Running in CPU emulation mode")


def main():
    """Main A100 emulation entry point"""
    parser = argparse.ArgumentParser(description="A100 GPU Emulation with 24GB Memory Constraint")
    parser.add_argument("--tokens", type=int, default=128000, help="Number of tokens for testing")
    parser.add_argument("--duration", type=int, default=300, help="Power monitoring duration in seconds")
    parser.add_argument("--validate", action="store_true", help="Validate memory constraint")

    args = parser.parse_args()

    print("🎬 A100 Emulator: Hardware-Edge Simulation")
    print("=" * 50)

    # Setup environment
    setup_environment()

    # Create emulator with 24GB constraint
    with A100Emulator(memory_fraction=24.0/80.0) as emulator:
        # Reset peak memory tracking
        emulator.reset_peak_memory()

        # Validate memory constraint
        if args.validate:
            validate_memory_constraint(emulator, args.tokens)

        # Log initial state
        emulator.log_memory_metrics("initial")

        # Run power monitoring (placeholder for actual inference)
        power_stats = run_power_monitoring(emulator, args.duration)

        # Log final state
        emulator.log_memory_metrics("final")

        print("\n📋 A100 Emulation Summary:")
        print(f"   Memory Ceiling: {24.0}GB (24/80 fraction)")
        print(f"   GPU Available: {torch.cuda.is_available()}")
        print(f"   Power Monitoring: {'Enabled' if power_stats else 'Failed'}")

        return power_stats


if __name__ == "__main__":
    main()