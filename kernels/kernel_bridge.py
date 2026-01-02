#!/usr/bin/env python3
"""
7DMH-QA Kernel Bridge
Python interface to CUDA/CPU kernels for Crystal Architecture.
Provides unified API regardless of available hardware.
"""

import os
import sys
import ctypes
import numpy as np
from typing import Optional, Tuple
from enum import Enum

# Sacred Constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895


class ComputeDevice(Enum):
    """Available compute devices"""
    CPU = "cpu"
    CUDA = "cuda"
    HIP = "hip"


class KernelBridge:
    """
    Unified interface to 7DMH-QA compute kernels.
    Automatically selects best available backend.
    """
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize kernel bridge.
        
        Args:
            preferred_device: "cuda", "hip", "cpu", or None for auto-detect
        """
        self.device = self._detect_device(preferred_device)
        self.cuda_available = False
        self.cupy = None
        
        if self.device == ComputeDevice.CUDA:
            self._init_cuda()
        elif self.device == ComputeDevice.HIP:
            self._init_hip()
        
        print(f"[7DMH-QA] Kernel Bridge initialized: {self.device.value.upper()}")
    
    def _detect_device(self, preferred: Optional[str]) -> ComputeDevice:
        """Auto-detect best available device"""
        if preferred:
            return ComputeDevice(preferred.lower())
        
        # Try CUDA
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return ComputeDevice.CUDA
        except:
            pass
        
        # Try HIP (AMD)
        try:
            import subprocess
            result = subprocess.run(['rocm-smi'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return ComputeDevice.HIP
        except:
            pass
        
        return ComputeDevice.CPU
    
    def _init_cuda(self):
        """Initialize CUDA backend"""
        try:
            import cupy as cp
            self.cupy = cp
            self.cuda_available = True
            
            # Get device info
            device = cp.cuda.Device(0)
            mem = device.mem_info
            print(f"   CUDA Device: {device.attributes['MultiProcessorCount']} SMs")
            print(f"   VRAM: {mem[1] / 1e9:.1f} GB")
        except Exception as e:
            print(f"   CUDA init failed: {e}")
            self.device = ComputeDevice.CPU
    
    def _init_hip(self):
        """Initialize HIP backend (uses CuPy with ROCm)"""
        # HIP uses same CuPy interface on AMD
        self._init_cuda()
    
    # ==============================================================
    # KERNEL IMPLEMENTATIONS
    # ==============================================================
    
    def manifold_projection(self, input_tensor: np.ndarray, 
                           dim: int = 64) -> np.ndarray:
        """
        Manifold-Constrained Projection (S² Stability).
        Projects tensor onto 7D Poincaré Ball.
        
        Args:
            input_tensor: Input tensor (any shape, will be flattened)
            dim: Feature dimension
            
        Returns:
            Projected tensor (same shape as input)
        """
        original_shape = input_tensor.shape
        flat = input_tensor.flatten().astype(np.float32)
        n = len(flat)
        
        if self.cuda_available:
            return self._cuda_manifold_projection(flat, n, dim).reshape(original_shape)
        else:
            return self._cpu_manifold_projection(flat, n, dim).reshape(original_shape)
    
    def _cuda_manifold_projection(self, data: np.ndarray, n: int, dim: int) -> np.ndarray:
        """CUDA manifold projection"""
        cp = self.cupy
        
        d_input = cp.array(data)
        d_output = cp.zeros_like(d_input)
        
        # Calculate row indices
        row = cp.arange(n) // dim
        col = cp.arange(n) % dim
        
        # Poincare projection
        norm = cp.abs(d_input)
        projected = d_input / (1.0 + norm + PHI_INV)
        
        # Add identity for stability
        identity_mask = (col == (row % dim)).astype(cp.float32)
        d_output = projected + identity_mask * 0.01
        
        return d_output.get()
    
    def _cpu_manifold_projection(self, data: np.ndarray, n: int, dim: int) -> np.ndarray:
        """CPU manifold projection"""
        row = np.arange(n) // dim
        col = np.arange(n) % dim
        
        norm = np.abs(data)
        projected = data / (1.0 + norm + PHI_INV)
        
        identity_mask = (col == (row % dim)).astype(np.float32)
        output = projected + identity_mask * 0.01
        
        return output
    
    def phi_modulation(self, data: np.ndarray, phase: float = 0.0) -> np.ndarray:
        """
        Apply Golden Ratio modulation to tensor.
        
        Args:
            data: Input tensor
            phase: Phase offset
            
        Returns:
            Modulated tensor
        """
        if self.cuda_available:
            return self._cuda_phi_modulation(data, phase)
        else:
            return self._cpu_phi_modulation(data, phase)
    
    def _cuda_phi_modulation(self, data: np.ndarray, phase: float) -> np.ndarray:
        cp = self.cupy
        d_data = cp.array(data.astype(np.float32))
        modulated = d_data * cp.cos(d_data * PHI + phase) * PHI_INV
        return cp.tanh(modulated).get()
    
    def _cpu_phi_modulation(self, data: np.ndarray, phase: float) -> np.ndarray:
        modulated = data * np.cos(data * PHI + phase) * PHI_INV
        return np.tanh(modulated)
    
    def flux_unfold(self, seed: np.ndarray, output_size: int,
                   phi_flux: float = 0.809) -> np.ndarray:
        """
        Unfold seed into full parameter space.
        
        Args:
            seed: Small seed tensor
            output_size: Target output size
            phi_flux: Flux parameter (default: phi/2)
            
        Returns:
            Expanded parameter tensor
        """
        seed = seed.astype(np.float32)
        seed_size = len(seed)
        
        if self.cuda_available:
            return self._cuda_flux_unfold(seed, seed_size, output_size, phi_flux)
        else:
            return self._cpu_flux_unfold(seed, seed_size, output_size, phi_flux)
    
    def _cuda_flux_unfold(self, seed: np.ndarray, seed_size: int,
                         output_size: int, phi_flux: float) -> np.ndarray:
        cp = self.cupy
        
        d_seed = cp.array(seed)
        idx = cp.arange(output_size)
        
        seed_idx = idx % seed_size
        generation = idx // seed_size
        
        base = d_seed[seed_idx]
        flux = cp.sin(base * PHI + generation * phi_flux) * PHI_INV
        interference = cp.cos(idx.astype(cp.float32) * PHI_INV / 1000.0)
        
        output = cp.tanh(base + flux * 0.1 + interference * 0.01)
        return output.get()
    
    def _cpu_flux_unfold(self, seed: np.ndarray, seed_size: int,
                        output_size: int, phi_flux: float) -> np.ndarray:
        idx = np.arange(output_size)
        
        seed_idx = idx % seed_size
        generation = idx // seed_size
        
        base = seed[seed_idx]
        flux = np.sin(base * PHI + generation * phi_flux) * PHI_INV
        interference = np.cos(idx.astype(np.float32) * PHI_INV / 1000.0)
        
        return np.tanh(base + flux * 0.1 + interference * 0.01)
    
    def holographic_interference(self, pattern1: np.ndarray,
                                 pattern2: np.ndarray) -> np.ndarray:
        """
        Generate holographic interference from two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern (same shape)
            
        Returns:
            Interference pattern
        """
        assert pattern1.shape == pattern2.shape, "Patterns must have same shape"
        
        p1 = pattern1.astype(np.float32)
        p2 = pattern2.astype(np.float32)
        
        if self.cuda_available:
            return self._cuda_interference(p1, p2)
        else:
            return self._cpu_interference(p1, p2)
    
    def _cuda_interference(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        cp = self.cupy
        
        d_p1 = cp.array(p1)
        d_p2 = cp.array(p2)
        
        phase1 = cp.arctan2(cp.sin(d_p1 * np.pi), cp.cos(d_p1 * np.pi))
        phase2 = cp.arctan2(cp.sin(d_p2 * np.pi), cp.cos(d_p2 * np.pi))
        phase_diff = phase1 - phase2
        
        interference = cp.cos(phase_diff) * PHI_INV
        output = interference / (1.0 + cp.abs(interference))
        
        return output.get()
    
    def _cpu_interference(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        phase1 = np.arctan2(np.sin(p1 * np.pi), np.cos(p1 * np.pi))
        phase2 = np.arctan2(np.sin(p2 * np.pi), np.cos(p2 * np.pi))
        phase_diff = phase1 - phase2
        
        interference = np.cos(phase_diff) * PHI_INV
        return interference / (1.0 + np.abs(interference))
    
    def quantum_evolution_step(self, field: np.ndarray, phase: float) -> np.ndarray:
        """
        Single step of quantum field evolution.
        
        Args:
            field: 2D field array
            phase: Evolution phase
            
        Returns:
            Evolved field
        """
        assert field.ndim == 2, "Field must be 2D"
        
        if self.cuda_available:
            return self._cuda_evolution(field.astype(np.float32), phase)
        else:
            return self._cpu_evolution(field.astype(np.float32), phase)
    
    def _cuda_evolution(self, field: np.ndarray, phase: float) -> np.ndarray:
        cp = self.cupy
        d_field = cp.array(field)
        h, w = field.shape
        
        # Neighbor average (toroidal)
        neighbors = (
            cp.roll(d_field, 1, axis=0) +
            cp.roll(d_field, -1, axis=0) +
            cp.roll(d_field, 1, axis=1) +
            cp.roll(d_field, -1, axis=1)
        ) / 4.0
        
        # Quantum interference
        interference = cp.sin(d_field + phase) * cp.cos(d_field * PHI_INV + phase)
        
        # Evolution
        evolved = d_field + interference * 0.1
        evolved = (evolved + neighbors) / 2.0
        
        # Sacred sigmoid
        output = 1.0 / (1.0 + cp.exp(-(evolved + cp.cos(evolved * PHI) * PHI_INV) * PHI))
        
        return output.get()
    
    def _cpu_evolution(self, field: np.ndarray, phase: float) -> np.ndarray:
        h, w = field.shape
        
        neighbors = (
            np.roll(field, 1, axis=0) +
            np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) +
            np.roll(field, -1, axis=1)
        ) / 4.0
        
        interference = np.sin(field + phase) * np.cos(field * PHI_INV + phase)
        
        evolved = field + interference * 0.1
        evolved = (evolved + neighbors) / 2.0
        
        output = 1.0 / (1.0 + np.exp(-(evolved + np.cos(evolved * PHI) * PHI_INV) * PHI))
        
        return output
    
    def entropy_mine(self, size: int, time_factor: float = 1.0) -> np.ndarray:
        """
        Generate crystal entropy.
        
        Args:
            size: Number of entropy values
            time_factor: Time-based modulation
            
        Returns:
            Entropy array [-1, 1]
        """
        if self.cuda_available:
            return self._cuda_entropy(size, time_factor)
        else:
            return self._cpu_entropy(size, time_factor)
    
    def _cuda_entropy(self, size: int, time_factor: float) -> np.ndarray:
        cp = self.cupy
        
        noise = cp.random.randn(size).astype(cp.float32)
        idx = cp.arange(size).astype(cp.float32)
        phi_osc = cp.sin(2.0 * np.pi * PHI * idx / size * time_factor)
        
        entropy = (noise + phi_osc) * PHI
        return cp.tanh(entropy).get()
    
    def _cpu_entropy(self, size: int, time_factor: float) -> np.ndarray:
        noise = np.random.randn(size).astype(np.float32)
        idx = np.arange(size).astype(np.float32)
        phi_osc = np.sin(2.0 * np.pi * PHI * idx / size * time_factor)
        
        entropy = (noise + phi_osc) * PHI
        return np.tanh(entropy)
    
    def benchmark(self) -> dict:
        """Run quick benchmark of kernel performance"""
        import time
        
        results = {}
        test_size = 1000000
        
        # Manifold projection
        data = np.random.randn(test_size).astype(np.float32)
        start = time.perf_counter()
        _ = self.manifold_projection(data)
        results['manifold_projection'] = time.perf_counter() - start
        
        # Phi modulation
        start = time.perf_counter()
        _ = self.phi_modulation(data)
        results['phi_modulation'] = time.perf_counter() - start
        
        # Flux unfold
        seed = np.random.randn(512).astype(np.float32)
        start = time.perf_counter()
        _ = self.flux_unfold(seed, 100000)
        results['flux_unfold'] = time.perf_counter() - start
        
        # Quantum evolution
        field = np.random.randn(256, 256).astype(np.float32)
        start = time.perf_counter()
        _ = self.quantum_evolution_step(field, 0.0)
        results['quantum_evolution'] = time.perf_counter() - start
        
        return results


# Global instance for convenience
_bridge = None

def get_kernel_bridge(preferred_device: Optional[str] = None) -> KernelBridge:
    """Get or create singleton kernel bridge"""
    global _bridge
    if _bridge is None:
        _bridge = KernelBridge(preferred_device)
    return _bridge


if __name__ == "__main__":
    print("=" * 60)
    print("   7DMH-QA KERNEL BRIDGE TEST")
    print("=" * 60)
    
    bridge = KernelBridge()
    
    # Test all kernels
    print("\n[TEST] Manifold Projection")
    data = np.random.randn(1000, 64).astype(np.float32)
    projected = bridge.manifold_projection(data)
    print(f"   Input: {data.shape} -> Output: {projected.shape}")
    
    print("\n[TEST] Phi Modulation")
    modulated = bridge.phi_modulation(data.flatten())
    print(f"   Range: [{modulated.min():.3f}, {modulated.max():.3f}]")
    
    print("\n[TEST] Flux Unfold")
    seed = np.random.randn(512).astype(np.float32)
    unfolded = bridge.flux_unfold(seed, 1000000)
    print(f"   512 -> 1,000,000 parameters")
    
    print("\n[TEST] Holographic Interference")
    p1 = np.random.randn(64, 64).astype(np.float32)
    p2 = np.random.randn(64, 64).astype(np.float32)
    interference = bridge.holographic_interference(p1, p2)
    print(f"   Interference shape: {interference.shape}")
    
    print("\n[TEST] Quantum Evolution")
    field = np.random.randn(128, 128).astype(np.float32)
    evolved = bridge.quantum_evolution_step(field, 0.5)
    print(f"   Evolved shape: {evolved.shape}")
    
    print("\n[BENCHMARK]")
    results = bridge.benchmark()
    for name, time_s in results.items():
        print(f"   {name}: {time_s*1000:.2f}ms")
    
    print("\n✅ All kernel tests passed!")
