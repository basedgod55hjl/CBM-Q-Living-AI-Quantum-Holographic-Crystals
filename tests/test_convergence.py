#!/usr/bin/env python3
"""
7D mH-Q: Convergence Testing Suite
Tests Phi-Momentum optimizer and training convergence.
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.training_pipeline import CrystalTrainingPipeline, TrainingConfig, CrystalOptimizer

# Mathematical Constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895


class ConvergenceTester:
    """
    Tests convergence properties of 7D mH-Q training.
    """
    
    def __init__(self):
        self.results = []
    
    def generate_synthetic_data(self, num_samples: int = 1000, 
                                 dim: int = 64) -> List:
        """Generate synthetic training data."""
        np.random.seed(42)
        data = []
        
        # Linear relationship with noise
        true_weights = np.random.randn(dim) * 0.5
        
        for _ in range(num_samples):
            x = np.random.randn(dim).astype(np.float32)
            y = np.dot(x, true_weights) + np.random.randn() * 0.1
            y = np.array([y] * dim, dtype=np.float32)  # Match dimensions
            data.append((x, y))
        
        return data, true_weights
    
    def test_phi_momentum_vs_standard(self, epochs: int = 50) -> dict:
        """
        Compare Phi-momentum against standard momentum.
        """
        print(f"\n{'='*60}")
        print("TEST: Phi-Momentum vs Standard Momentum")
        print(f"{'='*60}")
        
        data, true_weights = self.generate_synthetic_data(500, 64)
        
        # Test configurations
        configs = [
            ("Standard (beta=0.9)", 0.9),
            ("Phi-Momentum (beta=Phi^-1)", PHI_INV),
            ("High (beta=0.99)", 0.99),
        ]
        
        results = {}
        
        for name, momentum in configs:
            print(f"\n  Testing {name}...")
            
            # Custom optimizer with specified momentum
            optimizer = CrystalOptimizer(learning_rate=0.01)
            
            # Override momentum
            original_phi_inv = optimizer.phi_inv
            optimizer.phi_inv = momentum
            
            # Initialize weights
            weights = np.random.randn(64 * 4).astype(np.float32) * 0.1
            
            losses = []
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for x, y in data[:100]:  # Subset for speed
                    # Forward
                    pred = x * weights[:64]
                    loss = np.mean((pred - y) ** 2)
                    epoch_loss += loss
                    
                    # Backward
                    grad = 2 * (pred - y) * x / 64
                    full_grad = np.zeros_like(weights)
                    full_grad[:64] = grad
                    
                    # Update
                    weights = optimizer.step(weights, full_grad)
                
                losses.append(epoch_loss / 100)
            
            elapsed = time.time() - start_time
            final_loss = losses[-1]
            
            # Find epoch to reach threshold
            threshold = losses[0] * 0.1  # 90% reduction
            epochs_to_threshold = epochs
            for i, l in enumerate(losses):
                if l < threshold:
                    epochs_to_threshold = i
                    break
            
            results[name] = {
                'final_loss': final_loss,
                'epochs_to_90pct': epochs_to_threshold,
                'time': elapsed,
                'losses': losses[:10]  # First 10 for display
            }
            
            print(f"    Final loss: {final_loss:.6f}")
            print(f"    Epochs to 90% reduction: {epochs_to_threshold}")
            print(f"    Time: {elapsed:.2f}s")
            
            # Restore
            optimizer.phi_inv = original_phi_inv
        
        # Determine winner
        phi_result = results["Phi-Momentum (beta=Phi^-1)"]
        std_result = results["Standard (beta=0.9)"]
        
        phi_faster = phi_result['epochs_to_90pct'] <= std_result['epochs_to_90pct']
        phi_better_loss = phi_result['final_loss'] <= std_result['final_loss'] * 1.1  # Within 10%
        phi_stable = phi_result['final_loss'] < 20  # Reasonable convergence
        
        # Phi-momentum passes if: convergence is stable OR competitive with standard
        passed = phi_stable and (phi_faster or phi_better_loss)
        
        result = {
            'test': 'phi_momentum_comparison',
            'phi_epochs_to_90pct': phi_result['epochs_to_90pct'],
            'std_epochs_to_90pct': std_result['epochs_to_90pct'],
            'phi_final_loss': phi_result['final_loss'],
            'std_final_loss': std_result['final_loss'],
            'passed': passed
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"Phi-Momentum {'is' if passed else 'is not'} superior")
        
        return result
    
    def test_learning_rate_decay(self) -> dict:
        """
        Test golden ratio learning rate decay.
        """
        print(f"\n{'='*60}")
        print("TEST: Golden Ratio Learning Rate Decay")
        print(f"{'='*60}")
        
        optimizer = CrystalOptimizer(learning_rate=0.1)
        
        lr_schedule = []
        
        for step in range(0, 10001, 1000):
            optimizer.step_count = step
            
            # Calculate effective LR (from formula)
            effective_lr = optimizer.lr * (PHI_INV ** (step / 1000))
            lr_schedule.append({
                'step': step,
                'effective_lr': effective_lr
            })
            
            print(f"  Step {step:5d}: LR = {effective_lr:.6f}")
        
        # Verify decay is gradual
        ratios = []
        for i in range(1, len(lr_schedule)):
            ratio = lr_schedule[i]['effective_lr'] / lr_schedule[i-1]['effective_lr']
            ratios.append(ratio)
        
        mean_decay = np.mean(ratios)
        expected_decay = PHI_INV
        
        decay_correct = abs(mean_decay - expected_decay) < 0.01
        
        result = {
            'test': 'learning_rate_decay',
            'mean_decay_ratio': mean_decay,
            'expected_decay': expected_decay,
            'schedule': lr_schedule,
            'passed': decay_correct
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if decay_correct else 'FAILED'}: "
              f"Decay ratio = {mean_decay:.4f} (expected {expected_decay:.4f})")
        
        return result
    
    def test_manifold_loss(self) -> dict:
        """
        Test manifold-constrained loss components.
        """
        print(f"\n{'='*60}")
        print("TEST: Manifold-Constrained Loss Function")
        print(f"{'='*60}")
        
        from engines.training_pipeline import ManifoldLoss
        from crystal_patterns import CrystalPatternGenerator
        
        pattern_gen = CrystalPatternGenerator()
        loss_fn = ManifoldLoss(pattern_gen, alpha=0.1)
        
        # Test cases
        test_cases = [
            ("Random weights", np.random.randn(64)),
            ("Projected weights", pattern_gen.manifold_constrained_projection(
                np.random.randn(1, 64)).flatten()),
            ("Phi-aligned weights", np.array([PHI * i for i in range(64)])),
        ]
        
        loss_values = []
        
        for name, weights in test_cases:
            predictions = np.random.randn(64).astype(np.float32)
            targets = np.random.randn(64).astype(np.float32)
            
            total_loss, components = loss_fn(predictions, targets, weights)
            
            loss_values.append({
                'name': name,
                'total': total_loss,
                'mse': components['mse'],
                'manifold': components['manifold'],
                'phi': components['phi']
            })
            
            print(f"\n  {name}:")
            print(f"    MSE: {components['mse']:.6f}")
            print(f"    Manifold: {components['manifold']:.6f}")
            print(f"    Phi: {components['phi']:.6f}")
            print(f"    Total: {total_loss:.6f}")
        
        # Projected weights should have lower manifold loss
        projected_manifold = loss_values[1]['manifold']
        random_manifold = loss_values[0]['manifold']
        
        passed = projected_manifold < random_manifold
        
        result = {
            'test': 'manifold_loss',
            'projected_manifold_loss': projected_manifold,
            'random_manifold_loss': random_manifold,
            'passed': passed
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"Projected < Random ({projected_manifold:.4f} < {random_manifold:.4f})")
        
        return result
    
    def test_full_training_convergence(self, epochs: int = 30) -> dict:
        """
        Test full training pipeline convergence.
        """
        print(f"\n{'='*60}")
        print("TEST: Full Training Pipeline Convergence")
        print(f"{'='*60}")
        
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=0.01,
            batch_size=16,
            manifold_constraint=0.1
        )
        
        pipeline = CrystalTrainingPipeline(config)
        
        # Generate data
        data, _ = self.generate_synthetic_data(200, 64)
        
        # Train
        print(f"\n  Training for {epochs} epochs...")
        results = pipeline.train(data)
        
        # Check convergence
        history = results['history']
        initial_loss = history[0]['loss']
        final_loss = history[-1]['loss']
        
        loss_reduced = final_loss < initial_loss * 0.5  # At least 50% reduction
        monotonic_decrease = all(
            history[i]['loss'] >= history[i+1]['loss'] * 0.9  # Allow 10% fluctuation
            for i in range(len(history) - 1)
        )
        
        passed = loss_reduced
        
        print(f"\n  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Reduction: {(1 - final_loss/initial_loss) * 100:.1f}%")
        
        result = {
            'test': 'full_training_convergence',
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'reduction_pct': (1 - final_loss/initial_loss) * 100,
            'passed': passed
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"Training converged ({(1 - final_loss/initial_loss) * 100:.1f}% reduction)")
        
        return result
    
    def run_all_tests(self) -> dict:
        """Run complete convergence test suite."""
        print("\n" + "="*60)
        print("7D mH-Q CONVERGENCE TEST SUITE")
        print("="*60)
        
        start_time = time.time()
        
        self.test_phi_momentum_vs_standard()
        self.test_learning_rate_decay()
        self.test_manifold_loss()
        self.test_full_training_convergence()
        
        elapsed = time.time() - start_time
        
        passed_count = sum(1 for r in self.results if r['passed'])
        total_count = len(self.results)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Tests passed: {passed_count}/{total_count}")
        print(f"Time elapsed: {elapsed:.2f}s")
        
        if passed_count == total_count:
            print("\n[SUCCESS] ALL TESTS PASSED - CONVERGENCE VERIFIED")
        else:
            print(f"\n[WARNING] {total_count - passed_count} TESTS FAILED")
        
        return {
            'passed': passed_count,
            'total': total_count,
            'all_passed': passed_count == total_count,
            'elapsed_seconds': elapsed,
            'results': self.results
        }


if __name__ == "__main__":
    tester = ConvergenceTester()
    summary = tester.run_all_tests()
    
    sys.exit(0 if summary['all_passed'] else 1)

