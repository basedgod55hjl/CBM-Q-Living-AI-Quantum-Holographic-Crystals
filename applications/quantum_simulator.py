#!/usr/bin/env python3
"""
7D mH-Q: Quantum Field Simulator
Simulates quantum field evolution on the Crystal Manifold.
"""

import sys
import os
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystal_patterns import CrystalPatternGenerator, CrystalEvolutionEngine

# Mathematical Constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895
PI = np.pi
TWO_PI = 2.0 * PI


@dataclass
class QuantumState:
    """Represents a quantum field state."""
    field: np.ndarray
    time: float
    energy: float
    entropy: float
    coherence: float


class QuantumFieldSimulator:
    """
    7D mH-Q Quantum Field Simulator
    
    Simulates quantum field evolution using:
    - Wave equation dynamics
    - Crystal lattice interactions
    - Golden ratio harmonic modulation
    - Holographic interference patterns
    """
    
    def __init__(self, field_size: Tuple[int, int] = (64, 64), 
                 dt: float = 0.01, 
                 use_gpu: bool = True):
        """
        Initialize quantum field simulator.
        
        Args:
            field_size: (height, width) of field
            dt: Time step for evolution
            use_gpu: Use GPU acceleration if available
        """
        self.field_size = field_size
        self.dt = dt
        self.use_gpu = use_gpu and self._check_gpu()
        
        self.pattern_gen = CrystalPatternGenerator(complexity=512)
        self.evolution_engine = CrystalEvolutionEngine(self.pattern_gen)
        
        # Initialize field
        self.field = None
        self.time = 0.0
        self.history: List[QuantumState] = []
        
        # Physical constants (in natural units)
        self.hbar = PHI_INV  # Reduced Planck constant (sacred)
        self.mass = 1.0
        self.potential_strength = PHI * 0.1
        
        print(f"[QUANTUM] Simulator initialized: {field_size[0]}x{field_size[1]}")
        print(f"[QUANTUM] GPU: {'Enabled' if self.use_gpu else 'Disabled (CPU)'}")
    
    def _check_gpu(self) -> bool:
        """Check GPU availability."""
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False
    
    def initialize_field(self, init_type: str = "gaussian", **kwargs) -> np.ndarray:
        """
        Initialize quantum field.
        
        Args:
            init_type: Initialization type
                - "gaussian": Gaussian wave packet
                - "plane_wave": Plane wave
                - "superposition": Multiple Gaussian superposition
                - "crystal": Crystal lattice pattern
                - "random": Random noise field
                
        Returns:
            Initialized field
        """
        height, width = self.field_size
        
        # Create coordinate grids
        y = np.linspace(-1, 1, height)
        x = np.linspace(-1, 1, width)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        if init_type == "gaussian":
            # Gaussian wave packet
            sigma = kwargs.get('sigma', 0.2)
            x0 = kwargs.get('x0', 0.0)
            y0 = kwargs.get('y0', 0.0)
            k0 = kwargs.get('k0', 5.0)  # Wave number
            
            envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
            phase = np.exp(1j * k0 * X)
            
            self.field = (envelope * phase).real
            
        elif init_type == "plane_wave":
            kx = kwargs.get('kx', 2 * PI)
            ky = kwargs.get('ky', 0.0)
            
            self.field = np.cos(kx * X + ky * Y)
            
        elif init_type == "superposition":
            # Superposition of multiple Gaussians
            n_packets = kwargs.get('n_packets', 3)
            self.field = np.zeros((height, width))
            
            for i in range(n_packets):
                angle = TWO_PI * i / n_packets
                x0 = 0.3 * np.cos(angle)
                y0 = 0.3 * np.sin(angle)
                sigma = 0.15
                k0 = 5.0
                
                envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
                phase = np.cos(k0 * (X * np.cos(angle) + Y * np.sin(angle)))
                
                self.field += envelope * phase
            
            self.field /= n_packets
            
        elif init_type == "crystal":
            # Crystal lattice pattern using sacred geometry
            flower = self.pattern_gen.generate_sacred_geometry("flower_of_life", num_rings=2)
            centers = flower['centers']
            
            self.field = np.zeros((height, width))
            for cx, cy in centers:
                cx_norm = cx / 3.0
                cy_norm = cy / 3.0
                dist = np.sqrt((X - cx_norm)**2 + (Y - cy_norm)**2)
                self.field += np.exp(-dist**2 / 0.02) * np.cos(dist * 10 * PHI)
            
            self.field /= np.max(np.abs(self.field))
            
        elif init_type == "random":
            self.field = np.random.randn(height, width)
            # Apply Gaussian smoothing
            from scipy.ndimage import gaussian_filter
            self.field = gaussian_filter(self.field, sigma=2)
            self.field /= np.max(np.abs(self.field))
        
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        # Normalize
        self.field = self.field.astype(np.float32)
        self.time = 0.0
        self.history = []
        
        # Record initial state
        self._record_state()
        
        print(f"[QUANTUM] Field initialized: {init_type}")
        return self.field
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute Laplacian using finite differences."""
        # 5-point stencil Laplacian
        laplacian = (
            np.roll(field, 1, axis=0) +
            np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) +
            np.roll(field, -1, axis=1) -
            4 * field
        )
        return laplacian
    
    def _compute_potential(self, field: np.ndarray) -> np.ndarray:
        """Compute potential energy contribution."""
        height, width = self.field_size
        y = np.linspace(-1, 1, height)
        x = np.linspace(-1, 1, width)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        # Harmonic potential with golden ratio modulation
        r2 = X**2 + Y**2
        potential = self.potential_strength * r2 * (1 + PHI_INV * np.sin(r2 * PHI * PI))
        
        return potential * field
    
    def evolve_step(self) -> np.ndarray:
        """
        Perform single evolution time step.
        
        Uses modified wave equation with crystal stability:
        ∂²ψ/∂t² = c² ∇²ψ - V(x)ψ + Crystal_modulation
        """
        if self.field is None:
            raise RuntimeError("Field not initialized")
        
        # Compute Laplacian (kinetic term)
        laplacian = self._compute_laplacian(self.field)
        
        # Compute potential term
        potential = self._compute_potential(self.field)
        
        # Crystal interference modulation
        phase = self.time * PHI
        interference = np.sin(self.field * PHI + phase) * np.cos(self.field * PHI_INV + phase)
        
        # Evolution equation
        # Using first-order approximation for stability
        d_field = self.dt * (
            self.hbar * laplacian -          # Diffusion
            potential +                       # Potential
            PHI_INV * interference * 0.1     # Crystal modulation
        )
        
        # Update field
        self.field = self.field + d_field
        
        # Sacred sigmoid stabilization
        self.field = 1.0 / (1.0 + np.exp(
            -(self.field + np.cos(self.field * PHI) * PHI_INV) * PHI
        ))
        
        # Renormalize to [-1, 1]
        self.field = 2 * self.field - 1
        
        self.time += self.dt
        
        return self.field
    
    def evolve(self, steps: int = 100, record_interval: int = 10) -> List[QuantumState]:
        """
        Evolve field for multiple steps.
        
        Args:
            steps: Number of evolution steps
            record_interval: Record state every N steps
            
        Returns:
            List of recorded quantum states
        """
        print(f"[QUANTUM] Evolving {steps} steps (dt={self.dt})...")
        
        start_time = time.time()
        
        for step in range(steps):
            self.evolve_step()
            
            if step % record_interval == 0:
                self._record_state()
                
            if step % (steps // 10) == 0:
                print(f"  Step {step}/{steps} | t={self.time:.4f}")
        
        elapsed = time.time() - start_time
        print(f"[QUANTUM] Evolution complete: {elapsed:.2f}s ({steps/elapsed:.1f} steps/s)")
        
        return self.history
    
    def _record_state(self):
        """Record current quantum state."""
        energy = self._compute_energy()
        entropy = self._compute_entropy()
        coherence = self._compute_coherence()
        
        state = QuantumState(
            field=self.field.copy(),
            time=self.time,
            energy=energy,
            entropy=entropy,
            coherence=coherence
        )
        self.history.append(state)
    
    def _compute_energy(self) -> float:
        """Compute total energy of field."""
        # Kinetic energy (gradient squared)
        grad_x = np.diff(self.field, axis=1)
        grad_y = np.diff(self.field, axis=0)
        kinetic = 0.5 * (np.sum(grad_x**2) + np.sum(grad_y**2))
        
        # Potential energy
        potential = 0.5 * self.potential_strength * np.sum(self.field**2)
        
        return float(kinetic + potential)
    
    def _compute_entropy(self) -> float:
        """Compute field entropy."""
        # Normalize field to probability
        prob = np.abs(self.field)**2
        prob = prob / np.sum(prob) + 1e-10
        
        # Shannon entropy
        entropy = -np.sum(prob * np.log(prob))
        return float(entropy)
    
    def _compute_coherence(self) -> float:
        """Compute quantum coherence metric."""
        # FFT-based coherence
        fft = np.fft.fft2(self.field)
        power = np.abs(fft)**2
        
        # Coherence is inverse of spectral spread
        coherence = 1.0 / (1.0 + np.std(power) / (np.mean(power) + 1e-10))
        return float(coherence)
    
    def apply_measurement(self, position: Tuple[int, int], 
                          strength: float = 0.5) -> np.ndarray:
        """
        Apply measurement-like interaction at position.
        
        Args:
            position: (y, x) position
            strength: Measurement strength
            
        Returns:
            Updated field
        """
        y, x = position
        height, width = self.field_size
        
        # Create measurement kernel (Gaussian collapse)
        Y, X = np.meshgrid(
            np.arange(height) - y,
            np.arange(width) - x,
            indexing='ij'
        )
        kernel = np.exp(-(X**2 + Y**2) / (2 * 5**2))
        
        # Collapse toward local value
        local_value = self.field[y, x]
        self.field = (1 - strength) * self.field + strength * kernel * local_value
        
        return self.field
    
    def visualize_state(self, state: QuantumState = None, 
                        save_path: str = None) -> np.ndarray:
        """
        Generate visualization of quantum state.
        
        Args:
            state: State to visualize (default: current)
            save_path: Optional path to save image
            
        Returns:
            RGB image array
        """
        if state is None:
            field = self.field
        else:
            field = state.field
        
        # Normalize to [0, 1]
        normalized = (field - field.min()) / (field.max() - field.min() + 1e-10)
        
        # Apply colormap (crystal-like purple/gold)
        r = np.clip(normalized * PHI, 0, 1)
        g = np.clip(normalized * PHI_INV, 0, 1)
        b = np.clip(1 - normalized, 0, 1)
        
        image = np.stack([r, g, b], axis=-1)
        image = (image * 255).astype(np.uint8)
        
        if save_path:
            try:
                from PIL import Image
                img = Image.fromarray(image)
                img.save(save_path)
                print(f"[QUANTUM] Saved visualization: {save_path}")
            except ImportError:
                np.save(save_path.replace('.png', '.npy'), image)
        
        return image
    
    def get_metrics(self) -> Dict:
        """Get current field metrics."""
        return {
            'time': self.time,
            'energy': self._compute_energy(),
            'entropy': self._compute_entropy(),
            'coherence': self._compute_coherence(),
            'field_mean': float(np.mean(self.field)),
            'field_std': float(np.std(self.field)),
            'field_min': float(np.min(self.field)),
            'field_max': float(np.max(self.field))
        }


def main():
    """Demo quantum field simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="7D mH-Q Quantum Field Simulator")
    parser.add_argument('--size', type=int, default=64, help='Field size')
    parser.add_argument('--steps', type=int, default=200, help='Evolution steps')
    parser.add_argument('--init', default='gaussian', 
                       choices=['gaussian', 'plane_wave', 'superposition', 'crystal', 'random'])
    parser.add_argument('--output', '-o', help='Output path for visualization')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("7D mH-Q QUANTUM FIELD SIMULATOR")
    print("="*60)
    
    # Create simulator
    simulator = QuantumFieldSimulator(
        field_size=(args.size, args.size),
        dt=0.01
    )
    
    # Initialize
    simulator.initialize_field(init_type=args.init)
    
    # Evolve
    history = simulator.evolve(steps=args.steps, record_interval=20)
    
    # Report
    print("\n" + "-"*40)
    print("EVOLUTION SUMMARY")
    print("-"*40)
    
    initial = history[0]
    final = history[-1]
    
    print(f"  Initial Energy: {initial.energy:.4f}")
    print(f"  Final Energy: {final.energy:.4f}")
    print(f"  Energy Change: {(final.energy - initial.energy) / initial.energy * 100:.2f}%")
    print(f"  Initial Entropy: {initial.entropy:.4f}")
    print(f"  Final Entropy: {final.entropy:.4f}")
    print(f"  Initial Coherence: {initial.coherence:.4f}")
    print(f"  Final Coherence: {final.coherence:.4f}")
    
    # Save visualization
    if args.output:
        simulator.visualize_state(save_path=args.output)
    
    print("\n✅ Simulation complete!")


if __name__ == "__main__":
    main()

