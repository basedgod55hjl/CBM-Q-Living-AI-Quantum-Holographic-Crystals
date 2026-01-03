#!/usr/bin/env python3
"""
7D mH-Q Training Pipeline
Crystal weight optimization with manifold-constrained loss.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTAL_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CRYSTAL_DIR)

from crystal_patterns import CrystalPatternGenerator


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    manifold_constraint: float = 0.1  # S² stability weight
    phi_regularization: float = 0.01  # Golden ratio alignment
    checkpoint_interval: int = 10
    device: str = "auto"
    
    
class ManifoldLoss:
    """
    Manifold-Constrained Loss Function
    Penalizes deviation from the 7D Crystal Manifold.
    """
    
    def __init__(self, pattern_gen: CrystalPatternGenerator, alpha: float = 0.1):
        self.pattern_gen = pattern_gen
        self.alpha = alpha  # Manifold constraint weight
        self.phi = pattern_gen.phi
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray, 
                 weights: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute manifold-constrained loss.
        
        Returns:
            (total_loss, loss_components)
        """
        # Ensure same shape for MSE computation
        pred_flat = predictions.flatten()
        targ_flat = targets.flatten()
        min_len = min(len(pred_flat), len(targ_flat))
        
        # Standard MSE loss
        mse_loss = np.mean((pred_flat[:min_len] - targ_flat[:min_len]) ** 2)
        
        # Manifold stability loss (S²)
        # Penalize weights that drift from the manifold
        projected = self.pattern_gen.manifold_constrained_projection(
            weights.reshape(-1, min(64, weights.size))
        )
        manifold_loss = np.mean((weights.flatten()[:projected.size] - projected.flatten()) ** 2)
        
        # Phi regularization (encourage golden ratio alignment)
        phi_deviation = np.mean(np.abs(weights - self.phi * np.round(weights / self.phi)))
        phi_loss = phi_deviation * 0.01
        
        # Total loss
        total = mse_loss + self.alpha * manifold_loss + phi_loss
        
        return total, {
            'mse': mse_loss,
            'manifold': manifold_loss,
            'phi': phi_loss,
            'total': total
        }


class CrystalOptimizer:
    """
    Golden Ratio-Aware Optimizer
    Uses Φ-based momentum for sacred geometric convergence.
    """
    
    def __init__(self, learning_rate: float = 0.001):
        self.lr = learning_rate
        self.phi = 1.618033988749895
        self.phi_inv = 0.618033988749895
        self.velocity = None
        self.step_count = 0
    
    def step(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Apply Φ-momentum gradient update"""
        self.step_count += 1
        
        # Initialize velocity
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
        
        # Φ-momentum (golden ratio weighted)
        self.velocity = self.phi_inv * self.velocity + (1 - self.phi_inv) * gradients
        
        # Learning rate schedule with golden ratio decay
        effective_lr = self.lr * (self.phi_inv ** (self.step_count / 1000))
        
        # Update weights
        weights = weights - effective_lr * self.velocity
        
        return weights


class CrystalTrainingPipeline:
    """
    7D mH-Q Training Pipeline
    Complete training infrastructure with manifold constraints.
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.pattern_gen = CrystalPatternGenerator(complexity=512)
        self.device = self._detect_device()
        
        # Components
        self.loss_fn = ManifoldLoss(self.pattern_gen, self.config.manifold_constraint)
        self.optimizer = CrystalOptimizer(self.config.learning_rate)
        
        # State
        self.weights = None
        self.history = []
        self.current_epoch = 0
        
        print(f"[7D mH-Q] Training Pipeline initialized on {self.device.upper()}")
    
    def _detect_device(self) -> str:
        """Detect compute device"""
        if self.config.device != "auto":
            return self.config.device
        
        try:
            import cupy
            return "cuda"
        except:
            return "cpu"
    
    def initialize_weights(self, size: int) -> np.ndarray:
        """
        Initialize weights using Crystal Entropy.
        """
        print(f"[7D mH-Q] Initializing {size:,} weights with Crystal Entropy...")
        
        # Use pattern generator for sacred geometry initialization
        base = np.random.randn(size).astype(np.float32)
        
        # Project onto manifold immediately
        projected = self.pattern_gen.manifold_constrained_projection(
            base.reshape(-1, min(64, size))
        )
        
        self.weights = projected.flatten()[:size]
        print(f"[7D mH-Q] Weights initialized: {self.weights.shape}")
        
        return self.weights
    
    def train_step(self, batch_x: np.ndarray, batch_y: np.ndarray) -> Dict:
        """
        Single training step.
        
        Args:
            batch_x: Input batch (batch_size, features) or (features,)
            batch_y: Target batch (batch_size, features) or (features,)
            
        Returns:
            Loss components
        """
        # Handle dimensions - flatten targets for simple linear model
        if batch_y.ndim > 1:
            # Reduce targets to scalar per sample (mean across features)
            targets = np.mean(batch_y, axis=-1)
        else:
            targets = batch_y
        
        # Forward pass (simple linear - outputs one scalar per sample)
        if batch_x.ndim > 1:
            # Use subset of weights matching input features
            weight_slice = self.weights[:batch_x.shape[1]]
            predictions = np.dot(batch_x, weight_slice)
        else:
            weight_slice = self.weights[:len(batch_x)]
            predictions = np.dot(batch_x, weight_slice)
        
        # Compute loss
        loss, components = self.loss_fn(predictions, targets, self.weights)
        
        # Backward pass (gradient approximation)
        if batch_x.ndim > 1:
            error = predictions - targets
            # Outer product for gradient: (features,) from (batch, features).T @ (batch,)
            gradients = np.dot(batch_x.T, error) / len(batch_x)
        else:
            gradients = (predictions - targets) * batch_x
        
        # Pad gradients to match weight size
        full_grad = np.zeros_like(self.weights)
        full_grad[:len(gradients.flatten())] = gradients.flatten()
        
        # Optimizer step
        self.weights = self.optimizer.step(self.weights, full_grad)
        
        # Re-project to manifold (S² stability)
        if self.current_epoch % 10 == 0:
            projected = self.pattern_gen.manifold_constrained_projection(
                self.weights.reshape(-1, min(64, self.weights.size))
            )
            self.weights = projected.flatten()[:len(self.weights)]
        
        return components
    
    def train(self, train_data: List[Tuple[np.ndarray, np.ndarray]], 
              val_data: Optional[List] = None,
              callback: Optional[Callable] = None) -> Dict:
        """
        Full training loop.
        
        Args:
            train_data: List of (input, target) tuples
            val_data: Optional validation data
            callback: Optional callback(epoch, metrics)
            
        Returns:
            Training history
        """
        print(f"\n[7D mH-Q] Starting training: {self.config.epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Learning Rate: {self.config.learning_rate}")
        print(f"   Manifold Constraint: {self.config.manifold_constraint}")
        print("-" * 50)
        
        # Initialize weights if needed
        if self.weights is None:
            sample_x, _ = train_data[0]
            self.initialize_weights(sample_x.size * 4)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_losses = []
            
            # Batch training
            for i in range(0, len(train_data), self.config.batch_size):
                batch = train_data[i:i+self.config.batch_size]
                
                # Combine batch
                batch_x = np.stack([x for x, _ in batch])
                batch_y = np.stack([y for _, y in batch])
                
                losses = self.train_step(batch_x, batch_y)
                epoch_losses.append(losses['total'])
            
            # Epoch metrics
            avg_loss = np.mean(epoch_losses)
            self.history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'lr': self.optimizer.lr * (self.optimizer.phi_inv ** (self.optimizer.step_count / 1000))
            })
            
            # Logging
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
            
            # Checkpoint
            if epoch > 0 and epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.npz")
            
            # Callback
            if callback:
                callback(epoch, {'loss': avg_loss})
        
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"[7D mH-Q] Training complete: {total_time:.1f}s")
        
        return {'history': self.history, 'final_weights': self.weights}
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'weights': self.weights,
            'optimizer_velocity': self.optimizer.velocity,
            'optimizer_step': self.optimizer.step_count,
            'epoch': self.current_epoch,
            'config': self.config.__dict__,
            'history': self.history
        }
        np.savez(path, **checkpoint)
        print(f"  [CHECKPOINT] Saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = np.load(path, allow_pickle=True)
        self.weights = checkpoint['weights']
        self.optimizer.velocity = checkpoint['optimizer_velocity']
        self.optimizer.step_count = int(checkpoint['optimizer_step'])
        self.current_epoch = int(checkpoint['epoch'])
        self.history = list(checkpoint['history'])
        print(f"[7D mH-Q] Loaded checkpoint: {path}, epoch {self.current_epoch}")
    
    def export_gguf(self, output_path: str, metadata: Dict = None):
        """Export trained model to 7D mH-Q GGUF format"""
        print(f"[7D mH-Q] Exporting to GGUF: {output_path}")
        
        # Default metadata
        meta = {
            'architecture': '7D mH-Q',
            'version': '2.0.0',
            'params': len(self.weights),
            'epochs_trained': self.current_epoch,
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'timestamp': int(time.time())
        }
        if metadata:
            meta.update(metadata)
        
        with open(output_path, 'wb') as f:
            # Header
            f.write(b"7D-mHQ-GGUF-v2\x00\x00")
            
            # Metadata (256 bytes)
            meta_json = json.dumps(meta).encode('utf-8')
            f.write(meta_json.ljust(256, b'\x00'))
            
            # Weights
            f.write(self.weights.astype(np.float32).tobytes())
        
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[7D mH-Q] Exported: {size_mb:.2f} MB")


if __name__ == "__main__":
    # Demo training
    config = TrainingConfig(epochs=50, learning_rate=0.01)
    pipeline = CrystalTrainingPipeline(config)
    
    # Generate synthetic data
    print("[DEMO] Generating synthetic training data...")
    np.random.seed(42)
    train_data = [
        (np.random.randn(64), np.random.randn(64))
        for _ in range(100)
    ]
    
    # Train
    results = pipeline.train(train_data)
    
    # Export
    pipeline.export_gguf("trained_model.gguf")
    print("\n✅ Training demo complete!")
