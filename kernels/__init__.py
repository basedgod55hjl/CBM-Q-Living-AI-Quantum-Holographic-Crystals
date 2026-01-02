"""
7DMH-QA Crystal Architecture - Kernels Package
CUDA/CPU compute kernels for manifold operations.
"""

from .kernel_bridge import KernelBridge, ComputeDevice, get_kernel_bridge

__all__ = [
    'KernelBridge',
    'ComputeDevice',
    'get_kernel_bridge',
]

__version__ = '2.0.0'
