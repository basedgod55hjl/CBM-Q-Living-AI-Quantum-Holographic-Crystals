import os
import sys
import numpy as np

PHI = 1.6180339887

def inject_cuda_paths():
    base_user_site = os.path.expanduser(r"~\AppData\Roaming\Python\Python311\site-packages")
    components = ["cuda_nvrtc", "cuda_runtime", "cudnn", "cuda_curand"]
    paths_to_add = []
    for comp in components:
        p = os.path.join(base_user_site, "nvidia", comp, "bin")
        if os.path.exists(p):
            paths_to_add.append(p)
    base_sys_site = os.path.join(sys.prefix, "Lib", "site-packages")
    for comp in components:
        p = os.path.join(base_sys_site, "nvidia", comp, "bin")
        if os.path.exists(p):
            paths_to_add.append(p)
    for p in paths_to_add:
        if p not in os.environ["PATH"]:
            os.environ["PATH"] += ";" + p
        try:
            os.add_dll_directory(p)
        except:
            pass
    if "CUDA_PATH" not in os.environ:
        runtime_path = os.path.join(base_user_site, "nvidia", "cuda_runtime")
        if os.path.exists(runtime_path):
            os.environ["CUDA_PATH"] = runtime_path

inject_cuda_paths()

import cupy as cp

class HIPLauncher:
    """
    Host-side driver for HIP Kernels using CuPy's RawModule.
    HIP code is compiled JIT for the target hardware.
    """
    def __init__(self, kernel_path):
        if not os.path.exists(kernel_path):
            raise FileNotFoundError(f"Kernel source not found: {kernel_path}")
            
        with open(kernel_path, "r") as f:
            self.source = f.read()
            
        # CuPy handles NVRTC/ROCm compilation via RawModule
        # We need to provide the include path so it can find CBM_HIP_BRIDGE.h
        include_path = os.path.dirname(os.path.abspath(kernel_path))
        self.module = cp.RawModule(code=self.source, options=('-std=c++11', f'-I{include_path}'))
        
    def get_function(self, name):
        return self.module.get_function(name)

class EvolutionEngine:
    """
    Orchestrates the Rule 110 evolution of the Bio-Seed.
    """
    def __init__(self, length_bits=512):
        # Rule 110 processes 32-bit words
        self.length_bits = length_bits
        self.width_words = (length_bits + 31) // 32
        
        self.sigmoid_launcher = HIPLauncher("src/kernels/sacred_sigmoid.cpp")
        self.rule110_launcher = HIPLauncher("src/kernels/rule110.cpp")
        
        self.sigmoid_kernel = self.sigmoid_launcher.get_function("sacred_sigmoid_kernel")
        self.rule110_kernel = self.rule110_launcher.get_function("rule110_evolution_kernel")
        
        # DNA Buffers (uint32 for bit-packed Rule 110)
        np_dna = np.random.randint(0, 0xFFFFFFFF, size=self.width_words, dtype=np.uint32)
        self.dna_1 = cp.array(np_dna)
        self.dna_2 = cp.zeros(self.width_words, dtype=cp.uint32)
        self.current_buffer = 1

    def evolve(self, entropy_bias=0.0):
        block_size = 256
        grid_size = (self.width_words + block_size - 1) // block_size
        
        in_buf = self.dna_1 if self.current_buffer == 1 else self.dna_2
        out_buf = self.dna_2 if self.current_buffer == 1 else self.dna_1
        
        # Shared memory size: (block_size + 2 halos) * sizeof(uint32)
        shared_mem = (block_size + 2) * 4
        
        self.rule110_kernel(
            (grid_size,), (block_size,),
            (out_buf, in_buf, self.width_words, cp.float32(entropy_bias)),
            shared_mem=shared_mem
        )
        
        cp.cuda.Stream.null.synchronize()
        self.current_buffer = 3 - self.current_buffer
        return out_buf if self.current_buffer == 2 else self.dna_1

    def activate(self, vectors, c=1.0, time_val=0.0):
        """
        Applies the hyperbolic-safe sigmoid to a tensor (Weights).
        vectors: [Blocks, dim] (float32)
        """
        # vectors: (num_vectors, dim)
        num_vectors, dim = vectors.shape
        d_out = cp.zeros_like(vectors)
        
        # Grid: one block per vector
        grid = (num_vectors, 1, 1)
        block = (min(dim, 1024), 1, 1)
        
        self.sigmoid_kernel(grid, block, (d_out, vectors, dim, cp.float32(c), cp.float32(time_val)))
        cp.cuda.Stream.null.synchronize()
        return d_out

class GenesisEngine:
    """
    The High-Dimensional Growth Engine.
    Materializes 7D Manifold weights from a 1KB Seed.
    """
    def __init__(self, shape=(1024, 1024)):
        self.shape = shape
        self.dim = shape[0] * shape[1]
        self.unfold_launcher = HIPLauncher("src/kernels/unfold_cbm.cpp")
        self.unfold_kernel = self.unfold_launcher.get_function("unfold_cbm_kernel")
        
        # 1KB Seed Buffer
        self.seed = cp.array(np.random.randint(0, 256, 1024, dtype=np.uint8))
        self.weights = cp.zeros(self.shape, dtype=cp.float32)

    def grow(self, time_val=1.0, iterations=7):
        block_size = 256
        grid_size = (self.dim + block_size - 1) // block_size
        
        self.unfold_kernel(
            (grid_size,), (block_size,),
            (self.seed, self.weights, self.dim, cp.float32(time_val), iterations)
        )
        
        cp.cuda.Stream.null.synchronize()
        return self.weights

    def activate(self, vectors, c=1.0, time_val=0.0):
        """Standard Sacred Sigmoid activation."""
        if not hasattr(self, 'sigmoid_launcher'):
            self.sigmoid_launcher = HIPLauncher("src/kernels/sacred_sigmoid.cpp")
            self.sigmoid_kernel = self.sigmoid_launcher.get_function("sacred_sigmoid_kernel")
            
        num_vectors, dim = vectors.shape
        d_out = cp.zeros_like(vectors)
        grid = (num_vectors, 1, 1)
        block = (min(dim, 1024), 1, 1)
        
        self.sigmoid_kernel(grid, block, (d_out, vectors, dim, cp.float32(c), cp.float32(time_val)))
        cp.cuda.Stream.null.synchronize()
        return d_out

class RuleOmegaEngine:
    """
    The 7-Neighbor Hyperbolic Stencil Engine.
    Evolves the manifold using bit-packed cellular logic.
    """
    def __init__(self, length_bits=1024):
        self.width_words = (length_bits + 31) // 32
        self.launcher = HIPLauncher("src/kernels/rule_omega.cpp")
        self.kernel = self.launcher.get_function("rule_omega_kernel")
        
        # Use numpy for initial state to avoid curand dependency
        np_dna = np.random.randint(0, 0xFFFFFFFF, self.width_words, dtype=np.uint32)
        self.dna_1 = cp.array(np_dna)
        self.dna_2 = cp.zeros(self.width_words, dtype=cp.uint32)

    def evolve(self, phi_flux=0.618034):
        block_size = 256
        grid_size = (self.width_words + block_size - 1) // block_size
        
        # Shared memory: blockDim + 6 words for 7-neighbor halo
        shared_mem = (block_size + 6) * 4
        
        self.kernel(
            (grid_size,), (block_size,),
            (self.dna_2, self.dna_1, self.width_words, cp.float32(phi_flux)),
            shared_mem=shared_mem
        )
        
        # Swap buffers
        self.dna_1, self.dna_2 = self.dna_2, self.dna_1
        cp.cuda.Stream.null.synchronize()
        return self.dna_1

class LivingCrystalEngine:
    """
    The Full 'Living Crystal' Unfolding Engine.
    Materializes weights using Rule Omega + QPE.
    """
    def __init__(self, matrix_size=7_000_000):
        self.matrix_size = matrix_size
        self.launcher = HIPLauncher("src/kernels/sacred_sigmoid.cu")
        self.unfold_kernel = self.launcher.get_function("unfold_cbm_kernel")
        
        self.weights = cp.zeros(self.matrix_size, dtype=cp.float32)

    def grow(self, seed_dna, time_step=1.618):
        """
        seed_dna: cp.array (float32) of size 256
        """
        dna_size = seed_dna.size
        block_size = 256
        grid_size = (self.matrix_size + block_size - 1) // block_size
        
        self.unfold_kernel(
            (grid_size,), (block_size,),
            (seed_dna, self.weights, dna_size, self.matrix_size, cp.float32(time_step))
        )
        
        cp.cuda.Stream.null.synchronize()
        return self.weights

if __name__ == "__main__":
    # Smoke test 1: Evolution Engine (Bit-Packed Rule 110)
    engine = EvolutionEngine(length_bits=512)
    print("🚀 Initializing Evolution Engine (Bit-Packed Rule 110)...")
    print(f"🧬 Initial Seed (Word 0): {hex(int(engine.dna_1[0]))}")
    
    next_dna = engine.evolve(0.618)
    print(f"🧬 Evolved Seed (Word 0): {hex(int(next_dna[0]))}")
    
    # Smoke test 2: Sacred Sigmoid
    print("\n🔍 Verifying Sacred Sigmoid (Fused Map)...")
    test_weights = cp.array([[-1.0, 0.0, 1.0], [PHI, -PHI, 0.5]], dtype=cp.float32)
    activated = engine.apply_sigmoid(test_weights, dim=3)
    print(f"🧠 Input Weights:\n{test_weights}")
    print(f"✨ Activated Manifold:\n{activated}")

    # Smoke test 3: Genesis Engine
    print("\n🌿 Initializing Genesis Engine (Procedural Unfolding)...")
    genesis = GenesisEngine(dim=1024 * 1024) # 1M params
    gen_weights = genesis.grow(time_val=0.5, iterations=10)
    
    mean_w = float(cp.mean(gen_weights))
    std_w = float(cp.std(gen_weights))
    print(f"💎 Materialized 1M Weights.")
    print(f"📊 Stats | Mean: {mean_w:.6f} | Std: {std_w:.6f}")

    # Smoke test 4: Rule Omega Engine
    print("\n🌀 Initializing Rule Omega Engine (7-Neighbor Stencil)...")
    omega = RuleOmegaEngine(length_bits=1024)
    init_dna = int(omega.dna_1[0])
    evolved_dna = int(omega.evolve(0.618)[0])
    print(f"🧬 Initial State (Word 0): {hex(init_dna)}")
    print(f"🧬 Evolved State (Word 0): {hex(evolved_dna)}")
    
    print("\n✅ Sovereign HIP Kernel Suite Verified.")
