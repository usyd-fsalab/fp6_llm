/*
 * AMD ROCm HIP port of kernel_reduction.cuh
 * Changes: cuda.h/cuda_fp16.h/cuda_runtime.h -> hip/hip_runtime.h, hip/hip_fp16.h
 * Logic unchanged — no CUDA-specific intrinsics used here.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define REDUCTION_ELEMENT_PER_THREADBLOCK   256
#define HALF_PER_128BIT                     8

__global__ void SplitK_Reduction(half* C, float* Reduction_Workspace, size_t M_Global, size_t N_Global, int Split_K)
{
    half*  WARP_GPTR_C   = C                   + REDUCTION_ELEMENT_PER_THREADBLOCK * blockIdx.x;
    float* WARP_GPTR_R   = Reduction_Workspace + REDUCTION_ELEMENT_PER_THREADBLOCK * blockIdx.x;
    half*  THREAD_GPTR_C = WARP_GPTR_C         + threadIdx.x * HALF_PER_128BIT;
    float* THREAD_GPTR_R = WARP_GPTR_R         + threadIdx.x * HALF_PER_128BIT;

    float Results[HALF_PER_128BIT];
    #pragma unroll
    for (int i = 0; i < HALF_PER_128BIT; i++) Results[i] = 0.0f;

    for (int i = 0; i < Split_K; i++) {
        #pragma unroll
        for (int j = 0; j < HALF_PER_128BIT; j++) Results[j] += THREAD_GPTR_R[j];
        THREAD_GPTR_R += M_Global * N_Global;
    }

    #pragma unroll
    for (int i = 0; i < HALF_PER_128BIT; i++) THREAD_GPTR_C[i] = __float2half_rn(Results[i]);
}
