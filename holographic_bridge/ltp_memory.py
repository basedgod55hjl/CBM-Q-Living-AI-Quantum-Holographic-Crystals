import cupy as cp
import numpy as np
import os
import time

class LTPMemory:
    """
    Long-Term Potentiation (LTP) Memory.
    Persists the 7D Holographic Manifold state to disk.
    """
    def __init__(self, storage_dir="bridge/ltp"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def snapshot(self, weight_matrix, name="manifold"):
        """Saves the current weight matrix to .hsm format (compressed npz)."""
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.hsm"
        filepath = os.path.join(self.storage_dir, filename)
        
        # Convert to CPU for saving if it's not too massive, 
        # or use cupy.savez_compressed if we want to stay on GPU.
        # .hsm is just our custom extension for Holographic Synaptic Map.
        cp.savez_compressed(filepath, weight_matrix=weight_matrix)
        
        # Keep latest symlink
        latest_path = os.path.join(self.storage_dir, f"{name}_latest.hsm")
        if os.path.exists(latest_path):
            os.remove(latest_path)
            
        # On Windows we can't easily symlink without admin, so we copy
        cp.savez_compressed(latest_path, weight_matrix=weight_matrix)
        
        return filepath

    def load_latest(self, name="manifold"):
        """Loads the latest .hsm snapshot."""
        latest_path = os.path.join(self.storage_dir, f"{name}_latest.hsm")
        if os.path.exists(latest_path):
            data = cp.load(latest_path)
            return data['weight_matrix']
        return None

    def verify_stability(self, weight_matrix):
        """Checks if the manifold is stable (no NaNs or Infs)."""
        if cp.any(cp.isnan(weight_matrix)):
            return False
        if cp.any(cp.isinf(weight_matrix)):
            return False
        return True
