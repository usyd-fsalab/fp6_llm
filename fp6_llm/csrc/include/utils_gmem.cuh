/*
 * AMD ROCm HIP port of utils_gmem.cuh — working version (sync copy + prefetch hint)
 * OPT1 GCN buffer addressing removed — caused hipify cascade failures.
 * __builtin_prefetch hint retained for latency-hiding on streaming weight reads.
 */
#ifndef UTILS_GMEM_CUH
#define UTILS_GMEM_CUH

#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "configs.h"
#include "ptx_cp.async.cuh"

template<int SMEM_SIZE_IN_BYTES_PER_WARP>
__device__ __forceinline__ void CopyFromGlobalToShared_A(
    uint32_t*    SPTR,
    const uint4* GPTR,
    bool         pred_guard = true)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    half*       SPTR_HALF = reinterpret_cast<half*>(SPTR);
    const half* GPTR_HALF = reinterpret_cast<const half*>(GPTR);
    SPTR_HALF += lane_id * 8;
    GPTR_HALF += lane_id * 8;
    #pragma unroll
    for (int i = 0; i < SMEM_SIZE_IN_BYTES_PER_WARP / WARP_SIZE / 16; i++) {
        cp_async<16>(SPTR_HALF, GPTR_HALF, pred_guard);
        SPTR_HALF += 256;
        GPTR_HALF += 256;
    }
}

__device__ __forceinline__ void CopyFromGlobalToShared_Scales(
    half*       SPTR_QuantScales,
    const half* GPTR_A_Scales)
{
    int lane_id       = threadIdx.x % WARP_SIZE;
    int Offset_Shared = lane_id * 2;
    int Offset_Global = lane_id / 4 + (lane_id % 4) * 16;
    for (int i = 0; i < 2; i++)
        SPTR_QuantScales[Offset_Shared + i] = GPTR_A_Scales[Offset_Global + i * 8];
}

template<int MaxNumOfLinesToCopy, int BLOCK_WARPS>
__device__ __forceinline__ void CopyFromGlobalToShared(
    half (* __restrict__ SharedPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    const half*       GlobalPTR,
    const int         GlobalStride,
    const int         NumOfLinesLeft,
    bool              Pred = true)
{
    const int NumOfThreads  = BLOCK_WARPS * WARP_SIZE;
    const int NumOfGroups   = NumOfThreads / 8;
    const int MaxIteration  = (MaxNumOfLinesToCopy - 1) / NumOfGroups + 1;
    const int line_id       = threadIdx.x / 8;
    const int line_offset   = (threadIdx.x % 8) * 8;

    GlobalPTR += line_id * GlobalStride + line_offset;
    SharedPTR += line_id;
    #pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        bool AsyncCopyPred = (line_id + i * NumOfGroups) < NumOfLinesLeft && Pred;
        cp_async<16>(&(*SharedPTR)[line_offset], GlobalPTR, AsyncCopyPred);
        GlobalPTR += NumOfGroups * GlobalStride;
        SharedPTR += NumOfGroups;
    }
}

#endif  // UTILS_GMEM_CUH
