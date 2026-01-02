#ifndef CBM_HIP_BRIDGE_H
#define CBM_HIP_BRIDGE_H

/**
 * CBM HIP Bridge: Unifies CUDA and HIP for the Sovereign architecture.
 */

#ifdef __HIPCC__
    #include <hip/hip_runtime.h>
#else
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#endif

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// PHI Constant (Golden Ratio)
#define PHI 1.6180339887f

// Warp Size adaptation
#ifndef WARP_SIZE
    #ifdef __HIPCC__
        #define WARP_SIZE 64
    #else
        #define WARP_SIZE 32
    #endif
#endif

// NVIDIA Pattern: Error checking macros (critical for production stability)
// Reduces debugging time from hours to minutes

#ifdef __HIPCC__
    // HIP error checking
    #define GPU_CHECK(call) do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
    
    #define GPU_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) do { \
        hipLaunchKernelGGL(kernel, grid, block, shared, stream, __VA_ARGS__); \
        GPU_CHECK(hipGetLastError()); \
        GPU_CHECK(hipStreamSynchronize(stream)); \
    } while(0)
#else
    // CUDA error checking
    #define GPU_CHECK(call) do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
    
    #define GPU_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) do { \
        kernel<<<grid, block, shared, stream>>>(__VA_ARGS__); \
        GPU_CHECK(cudaGetLastError()); \
        GPU_CHECK(cudaStreamSynchronize(stream)); \
    } while(0)
    
    // HIP -> CUDA mapping (or vice versa)
    #define hipLaunchKernelGGL(kernel, grid, block, shared, stream, ...) \
        GPU_KERNEL_LAUNCH(kernel, grid, block, shared, stream, __VA_ARGS__)
    
    // Warp Shuffles (NVIDIA) - assume newer CUDA with _sync
    #ifndef __shfl_down
    #define __shfl_down(var, offset) __shfl_down_sync(0xFFFFFFFF, var, offset)
    #endif

    #ifndef __shfl
    #define __shfl(var, lane) __shfl_sync(0xFFFFFFFF, var, lane)
    #endif
#endif

// Warp-level parallel reduction
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

#endif // CBM_HIP_BRIDGE_H
