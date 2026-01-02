import torch
import torch.utils.dlpack
import cupy as cp
from biogenesis_loader import BiogenesisLoader

class HolographicBridge:
    def __init__(self, shm_name="cbm_entropy_bridge", width=1024, height=1024, seed_file=None):
        print("ðŸ”— HOLOGRAPHIC BRIDGE: Initializing Synaptic Link...")
        self.loader = BiogenesisLoader(shm_name, width, height, seed_file)
        self.loader.connect_to_entropy()
        
    def materialize_tensor(self) -> torch.Tensor:
        cupy_array = self.loader.weight_matrix
        dlpack_capsule = cupy_array.toDlpack()
        torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_capsule)
        return torch_tensor

    def evolve_and_sample(self, steps=1):
        block_size = 256
        grid_size = (self.loader.total_elements + block_size - 1) // block_size
        for _ in range(steps):
             entropy = self.loader.get_entropy()
             self.loader.unfold_kernel(
                (grid_size,), (block_size,),
                (
                    self.loader.seed_dna,
                    self.loader.weight_matrix,
                    self.loader.width,
                    self.loader.height,
                    cp.float32(entropy)
                )
             )
        cp.cuda.Stream.null.synchronize()
        return self.materialize_tensor()
