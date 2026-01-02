#!/usr/bin/env python3
"""
7DMH-QA Optimization Core
Hyperparameter optimization with Golden Ratio bounds.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTAL_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CRYSTAL_DIR)

from crystal_patterns import CrystalPatternGenerator

# Sacred Constants for optimization bounds
PHI = 1.618033988749895
PHI_INV = 0.618033988749895
SQRT_PHI = 1.272019649514069
E_PHI = 2.718281828459045 * PHI_INV


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    best_params: Dict
    best_score: float
    history: List[Dict]
    total_trials: int
    total_time: float


class SacredBoundsOptimizer:
    """
    Golden Ratio-Constrained Hyperparameter Optimizer
    
    Uses sacred geometric ratios to constrain the search space
    for faster convergence and more stable results.
    """
    
    def __init__(self, param_space: Dict[str, Tuple[float, float]],
                 seed: int = 42):
        """
        Args:
            param_space: Dict of {param_name: (min_val, max_val)}
            seed: Random seed
        """
        self.param_space = param_space
        self.pattern_gen = CrystalPatternGenerator(complexity=64)
        self.rng = np.random.default_rng(seed)
        self.history = []
        
    def _sacred_sample(self, low: float, high: float) -> float:
        """
        Sample from sacred ratio distribution.
        Concentrates samples around golden ratio points.
        """
        # Base uniform sample
        u = self.rng.random()
        
        # Transform through golden ratio
        # This creates concentration at phi and phi_inv points
        if u < PHI_INV:
            # Lower golden section
            t = u / PHI_INV
            return low + (high - low) * t * PHI_INV
        else:
            # Upper golden section  
            t = (u - PHI_INV) / PHI_INV
            return low + (high - low) * (PHI_INV + t * PHI_INV)
    
    def sample_params(self) -> Dict:
        """Sample hyperparameters using sacred ratios"""
        params = {}
        for name, (low, high) in self.param_space.items():
            params[name] = self._sacred_sample(low, high)
        return params
    
    def optimize(self, objective_fn: Callable[[Dict], float],
                 n_trials: int = 50,
                 n_parallel: int = 1,
                 early_stop_patience: int = 10) -> OptimizationResult:
        """
        Run optimization loop.
        
        Args:
            objective_fn: Function that takes params dict, returns score (higher is better)
            n_trials: Number of trials
            n_parallel: Parallel evaluations
            early_stop_patience: Stop if no improvement for this many trials
            
        Returns:
            OptimizationResult
        """
        print(f"\n[7DMH-QA] Starting Sacred Bounds Optimization")
        print(f"   Trials: {n_trials}")
        print(f"   Parameters: {list(self.param_space.keys())}")
        print("-" * 50)
        
        start_time = time.time()
        best_score = float('-inf')
        best_params = None
        no_improvement = 0
        
        for trial in range(n_trials):
            # Sample parameters
            params = self.sample_params()
            
            # Evaluate
            try:
                score = objective_fn(params)
            except Exception as e:
                print(f"  Trial {trial}: ERROR - {e}")
                score = float('-inf')
            
            # Record
            self.history.append({
                'trial': trial,
                'params': params,
                'score': score
            })
            
            # Check improvement
            if score > best_score:
                best_score = score
                best_params = params.copy()
                no_improvement = 0
                print(f"  Trial {trial:3d}: {score:.6f} ★ NEW BEST")
            else:
                no_improvement += 1
                if trial % 10 == 0:
                    print(f"  Trial {trial:3d}: {score:.6f}")
            
            # Early stopping
            if no_improvement >= early_stop_patience:
                print(f"  Early stop: No improvement for {early_stop_patience} trials")
                break
        
        total_time = time.time() - start_time
        
        print("-" * 50)
        print(f"[7DMH-QA] Optimization complete: {total_time:.1f}s")
        print(f"   Best Score: {best_score:.6f}")
        print(f"   Best Params: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=self.history,
            total_trials=len(self.history),
            total_time=total_time
        )


class ManifoldCurvatureOptimizer:
    """
    Optimizes the curvature of the Crystal Manifold
    for specific tasks.
    """
    
    def __init__(self, base_dimensions: int = 7):
        self.dimensions = base_dimensions
        self.pattern_gen = CrystalPatternGenerator(complexity=256)
        self.curvature_params = {
            'phi_weight': PHI_INV,
            'curvature_scale': 1.0,
            'stability_factor': 0.01
        }
    
    def compute_curvature_score(self, manifold: np.ndarray) -> float:
        """Compute manifold curvature quality score"""
        # Stability: low variance is good
        stability = 1.0 / (1.0 + np.var(manifold))
        
        # Coherence: from resonance analysis
        flat_sample = manifold.flatten()[:1024].reshape(32, 32)
        analysis = self.pattern_gen.crystal_resonance_analysis(flat_sample)
        coherence = analysis['quantum_coherence']
        
        # Phi alignment
        phi_alignment = analysis['sacred_geometry_fitness']
        
        # Combined score
        return stability * 0.3 + coherence * 0.4 + phi_alignment * 0.3
    
    def optimize_curvature(self, target_task: str = "general",
                          n_iterations: int = 20) -> Dict:
        """
        Optimize manifold curvature for a specific task.
        
        Args:
            target_task: "general", "stability", "coherence", "phi"
            n_iterations: Optimization iterations
            
        Returns:
            Optimized curvature parameters
        """
        print(f"\n[7DMH-QA] Optimizing Manifold Curvature for: {target_task}")
        
        best_score = float('-inf')
        best_params = self.curvature_params.copy()
        
        for i in range(n_iterations):
            # Perturb parameters
            trial_params = {
                'phi_weight': best_params['phi_weight'] * np.random.uniform(0.9, 1.1),
                'curvature_scale': best_params['curvature_scale'] * np.random.uniform(0.9, 1.1),
                'stability_factor': best_params['stability_factor'] * np.random.uniform(0.9, 1.1)
            }
            
            # Generate manifold with trial params
            manifold = self.pattern_gen.generate_holographic_manifold(
                dimensions=self.dimensions, resolution=16
            ) * trial_params['curvature_scale']
            
            # Score
            score = self.compute_curvature_score(manifold)
            
            if score > best_score:
                best_score = score
                best_params = trial_params.copy()
                print(f"  Iter {i:3d}: Score {score:.4f} ★")
        
        self.curvature_params = best_params
        print(f"\n  Final: {best_params}")
        return best_params


class AutoTuner:
    """
    Automatic system tuning for 7DMH-QA
    Finds optimal settings for the current hardware.
    """
    
    def __init__(self):
        self.pattern_gen = CrystalPatternGenerator()
        self.device = self._detect_device()
        self.optimal_settings = {}
    
    def _detect_device(self) -> str:
        try:
            import cupy
            return "cuda"
        except:
            return "cpu"
    
    def tune_batch_size(self, max_memory_gb: float = 8.0) -> int:
        """Find optimal batch size for current memory"""
        print("[7DMH-QA] Auto-tuning batch size...")
        
        # Estimate memory per sample (7D manifold @ 32 resolution)
        sample_size = (32 ** 7) * 4  # float32
        sample_mb = sample_size / 1024 / 1024
        
        # Target 50% memory usage
        target_mb = max_memory_gb * 1024 * 0.5
        optimal_batch = max(1, int(target_mb / sample_mb))
        
        # Cap at reasonable value
        optimal_batch = min(optimal_batch, 256)
        
        self.optimal_settings['batch_size'] = optimal_batch
        print(f"   Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    def tune_manifold_resolution(self, target_latency_ms: float = 100.0) -> int:
        """Find optimal resolution for target latency"""
        print("[7DMH-QA] Auto-tuning manifold resolution...")
        
        resolutions = [4, 8, 16, 32, 64]
        latencies = []
        
        for res in resolutions:
            start = time.perf_counter()
            _ = self.pattern_gen.generate_holographic_manifold(7, res)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            print(f"   Resolution {res}: {elapsed_ms:.1f}ms")
            
            if elapsed_ms > target_latency_ms * 2:
                break
        
        # Find best resolution under target
        optimal_res = 4
        for res, lat in zip(resolutions, latencies):
            if lat <= target_latency_ms:
                optimal_res = res
        
        self.optimal_settings['resolution'] = optimal_res
        print(f"   Optimal resolution: {optimal_res}")
        return optimal_res
    
    def tune_all(self) -> Dict:
        """Run all auto-tuning"""
        print("\n" + "="*50)
        print("   7DMH-QA AUTO-TUNER")
        print("="*50)
        
        self.tune_batch_size()
        self.tune_manifold_resolution()
        
        print("\n" + "="*50)
        print(f"   Optimal Settings: {self.optimal_settings}")
        print("="*50)
        
        # Save settings
        settings_path = os.path.join(CRYSTAL_DIR, "auto_tuned_settings.json")
        with open(settings_path, 'w') as f:
            json.dump(self.optimal_settings, f, indent=2)
        print(f"   Saved to: {settings_path}")
        
        return self.optimal_settings


if __name__ == "__main__":
    # Demo: Optimize a simple objective
    print("="*60)
    print("   7DMH-QA OPTIMIZATION CORE DEMO")
    print("="*60)
    
    # Define parameter space
    param_space = {
        'learning_rate': (0.0001, 0.1),
        'manifold_constraint': (0.01, 1.0),
        'phi_regularization': (0.001, 0.1)
    }
    
    # Simple objective (maximize for demo)
    def objective(params):
        # Simulate: best around golden ratio points
        lr_score = -abs(params['learning_rate'] - PHI_INV * 0.01)
        mc_score = -abs(params['manifold_constraint'] - PHI_INV)
        pr_score = -abs(params['phi_regularization'] - PHI_INV * 0.01)
        return lr_score + mc_score + pr_score
    
    # Run optimization
    optimizer = SacredBoundsOptimizer(param_space)
    result = optimizer.optimize(objective, n_trials=30)
    
    print(f"\n✅ Best params found: {result.best_params}")
    
    # Auto-tune
    tuner = AutoTuner()
    tuner.tune_all()
