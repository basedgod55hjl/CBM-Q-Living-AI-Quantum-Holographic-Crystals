#!/usr/bin/env python3
"""
7D mH-Q Crystal Inference Engine
Model inference with holographic pattern matching.
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Optional, Union

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTAL_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CRYSTAL_DIR)

from crystal_patterns import CrystalPatternGenerator


class CrystalInferenceEngine:
    """
    7D mH-Q: Crystal-Optimized Inference Engine
    
    Provides high-performance inference using holographic pattern matching
    and manifold-constrained projections for stability.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the Crystal Inference Engine.
        
        Args:
            model_path: Path to 7D mH-Q GGUF model file
            device: "cuda", "hip", "cpu", or "auto"
        """
        self.model_path = model_path
        self.device = self._detect_device(device)
        self.pattern_gen = CrystalPatternGenerator(complexity=512)
        self.model_loaded = False
        self.weights = None
        self.metadata = {}
        
        print(f"[7D mH-Q] Inference Engine initialized on {self.device.upper()}")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _detect_device(self, device: str) -> str:
        """Auto-detect best available compute device"""
        if device != "auto":
            return device
        
        # Try CUDA
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return "cuda"
        except:
            pass
        
        # Try HIP/ROCm
        try:
            import subprocess
            result = subprocess.run(['rocm-smi'], capture_output=True)
            if result.returncode == 0:
                return "hip"
        except:
            pass
        
        return "cpu"
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a 7D mH-Q GGUF model.
        
        Args:
            model_path: Path to .gguf model file
            
        Returns:
            True if loaded successfully
        """
        print(f"[7D mH-Q] Loading model: {model_path}")
        
        try:
            with open(model_path, "rb") as f:
                # Read header (16 bytes)
                header = f.read(16)
                if not header.startswith(b"mH-QA"):
                    print("   [!] Warning: Non-standard GGUF format")
                
                # Read metadata (256 bytes)
                meta_block = f.read(256)
                meta_json = meta_block.rstrip(b'\x00').decode('utf-8')
                self.metadata = json.loads(meta_json)
                print(f"   [+] Architecture: {self.metadata.get('architecture', 'unknown')}")
                print(f"   [+] Parameters: {self.metadata.get('params', 0):,}")
                
                # Read seed (1024 bytes)
                seed_block = f.read(1024)
                self.seed = np.frombuffer(seed_block[:1024], dtype=np.float16)
                print(f"   [+] Seed Complexity: {len(self.seed)}")
                
                # Read hash (64 bytes)
                seed_hash = f.read(64)
                
                # Read weights (remainder)
                weight_data = f.read()
                self.weights = np.frombuffer(weight_data, dtype=np.float32)
                print(f"   [+] Weight Tensor: {self.weights.shape}")
                
            self.model_loaded = True
            print(f"[7D mH-Q] Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[7D mH-Q] Load failed: {e}")
            return False
    
    def infer(self, input_data: Union[np.ndarray, List], 
              temperature: float = 0.7,
              top_k: int = 40) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor or list
            temperature: Sampling temperature (0.0 = deterministic)
            top_k: Top-K sampling parameter
            
        Returns:
            Output tensor
        """
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Convert to numpy
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)
        
        # Project input onto manifold for stability
        input_projected = self.pattern_gen.manifold_constrained_projection(
            input_data.reshape(-1, min(64, len(input_data)))
        )
        
        # Holographic pattern matching
        if self.device == "cuda":
            output = self._cuda_inference(input_projected, temperature)
        elif self.device == "hip":
            output = self._hip_inference(input_projected, temperature)
        else:
            output = self._cpu_inference(input_projected, temperature)
        
        # Apply top-k sampling
        if top_k > 0 and temperature > 0:
            output = self._top_k_sample(output, top_k, temperature)
        
        return output
    
    def _cpu_inference(self, input_data: np.ndarray, temperature: float) -> np.ndarray:
        """CPU-based inference using NumPy"""
        # Holographic cross-correlation
        correlation = np.correlate(input_data.flatten()[:len(self.weights)], 
                                   self.weights[:1000], mode='same')
        
        # Apply crystal resonance
        resonance = np.sin(correlation * self.pattern_gen.phi) * self.pattern_gen.phi_inv
        
        # Temperature scaling
        if temperature > 0:
            resonance = resonance / temperature
        
        # Softmax-like normalization
        exp_res = np.exp(resonance - np.max(resonance))
        output = exp_res / np.sum(exp_res)
        
        return output
    
    def _cuda_inference(self, input_data: np.ndarray, temperature: float) -> np.ndarray:
        """CUDA-accelerated inference"""
        try:
            import cupy as cp
            
            d_input = cp.array(input_data)
            d_weights = cp.array(self.weights[:1000])
            
            # GPU correlation
            correlation = cp.correlate(d_input.flatten()[:len(d_weights)], d_weights, mode='same')
            
            # Crystal resonance on GPU
            phi = cp.float32(self.pattern_gen.phi)
            resonance = cp.sin(correlation * phi) / phi
            
            # Temperature
            if temperature > 0:
                resonance = resonance / temperature
            
            # Softmax
            exp_res = cp.exp(resonance - cp.max(resonance))
            output = exp_res / cp.sum(exp_res)
            
            return output.get()
        except Exception as e:
            print(f"[7D mH-Q] CUDA fallback to CPU: {e}")
            return self._cpu_inference(input_data, temperature)
    
    def _hip_inference(self, input_data: np.ndarray, temperature: float) -> np.ndarray:
        """AMD HIP inference (uses same CuPy interface via ROCm)"""
        return self._cuda_inference(input_data, temperature)
    
    def _top_k_sample(self, probs: np.ndarray, k: int, temperature: float) -> np.ndarray:
        """Apply top-k sampling"""
        # Get top-k indices
        top_indices = np.argsort(probs)[-k:]
        
        # Zero out non-top-k
        mask = np.zeros_like(probs)
        mask[top_indices] = probs[top_indices]
        
        # Re-normalize
        if np.sum(mask) > 0:
            mask = mask / np.sum(mask)
        
        return mask
    
    def batch_infer(self, batch: List[np.ndarray], 
                    temperature: float = 0.7) -> List[np.ndarray]:
        """
        Batch inference for multiple inputs.
        
        Args:
            batch: List of input tensors
            temperature: Sampling temperature
            
        Returns:
            List of output tensors
        """
        results = []
        for item in batch:
            results.append(self.infer(item, temperature))
        return results
    
    def get_model_info(self) -> Dict:
        """Get loaded model information"""
        return {
            "loaded": self.model_loaded,
            "device": self.device,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "weight_shape": self.weights.shape if self.weights is not None else None
        }


# Streaming inference support
class CrystalStreamingInference:
    """Streaming inference for real-time applications"""
    
    def __init__(self, engine: CrystalInferenceEngine):
        self.engine = engine
        self.buffer = []
        self.context_window = 2048
    
    def stream(self, token: np.ndarray) -> np.ndarray:
        """Process single token in streaming mode"""
        self.buffer.append(token)
        
        # Maintain context window
        if len(self.buffer) > self.context_window:
            self.buffer = self.buffer[-self.context_window:]
        
        # Combine buffer for inference
        context = np.concatenate(self.buffer)
        return self.engine.infer(context)
    
    def reset(self):
        """Clear the streaming buffer"""
        self.buffer = []


if __name__ == "__main__":
    # Demo
    engine = CrystalInferenceEngine()
    
    # Try to load default model
    default_model = os.path.join(CRYSTAL_DIR, "genesis_v1.gguf")
    if os.path.exists(default_model):
        engine.load_model(default_model)
        
        # Test inference
        test_input = np.random.randn(64)
        output = engine.infer(test_input)
        print(f"\n[TEST] Input: {test_input.shape} -> Output: {output.shape}")
    else:
        print(f"\n[INFO] No model found at {default_model}")
        print("[INFO] Generate one with: python sovereign_genesis.py")
        print("[AUTH] 7D mH-Q Architecture by Sir Charles Spikes | Ohio, USA 🇺🇸")
