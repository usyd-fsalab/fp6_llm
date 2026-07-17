/*
 * AMD ROCm HIP port of ptx_mma.cuh — v4
 * Replaces NVIDIA PTX mma.sync.aligned.m16n8k16 with AMD MFMA.
 * All packing/unpacking uses __builtin_memcpy — avoids address-of-vector-element errors.
 * __restrict__ moved after * for pointer-to-array params (clang requirement).
 */
#ifndef PTX_MMA_CUH
#define PTX_MMA_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <assert.h>
#include "configs.h"

template <typename TilingConfig>
__device__ __forceinline__ void B_FromSharedToReg(
    uint32_t                  Reg[][4],
    half                      (*read_SPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    int                       slice_id)
{
    const int warpId  = threadIdx.x / WARP_SIZE;
    int       lane_id = threadIdx.x % WARP_SIZE;
    int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_col = TilingConfig::WARP_COL_MMA_TENSORS * MMA_8 * WARP_j;
    int col = (lane_id % 8) + (lane_id / 16) * 8;
    int row = (lane_id % 16) / 8 * 8;
    const half* base = &read_SPTR[warp_start_col + col][slice_id * MMA_16 + row];
    if (TilingConfig::WARP_COL_MMA_TENSORS == 1) {
        __builtin_memcpy(&Reg[0][0], base,     4);
        __builtin_memcpy(&Reg[0][1], base + 2, 4);
    } else {
        #pragma unroll
        for (int i = 0; i < TilingConfig::WARP_COL_MMA_TENSORS / 2; i++) {
            const half* p = base + i * 16 * (WARP_K + PADDING_SHARED_MEM_FOR_B_8);
            __builtin_memcpy(&Reg[i][0], p,      4);
            __builtin_memcpy(&Reg[i][1], p + 2,  4);
            __builtin_memcpy(&Reg[i][2], p + 8,  4);
            __builtin_memcpy(&Reg[i][3], p + 10, 4);
        }
    }
}

__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t c[], uint32_t * __restrict__ a, uint32_t * __restrict__ b)
{
    using half4  = __attribute__((ext_vector_type(4))) _Float16;
    using float4 = __attribute__((ext_vector_type(4))) float;
    float f0, f1, f2, f3;
    __builtin_memcpy(&f0, &c[0], 4);
    __builtin_memcpy(&f1, &c[1], 4);
    __builtin_memcpy(&f2, &c[2], 4);
    __builtin_memcpy(&f3, &c[3], 4);
    float4 acc = {f0, f1, f2, f3};
    uint64_t a0_64 = (uint64_t)a[0] | ((uint64_t)a[1] << 32);
    uint64_t a1_64 = (uint64_t)a[2] | ((uint64_t)a[3] << 32);
    uint64_t b0_64 = (uint64_t)b[0];
    uint64_t b1_64 = (uint64_t)b[1];
    half4 a_frag0, a_frag1, b_frag0, b_frag1;
    __builtin_memcpy(&a_frag0, &a0_64, 8);
    __builtin_memcpy(&a_frag1, &a1_64, 8);
    __builtin_memcpy(&b_frag0, &b0_64, 8);
    __builtin_memcpy(&b_frag1, &b1_64, 8);
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag0, b_frag0, acc, 0, 0, 0);
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag1, b_frag1, acc, 0, 0, 0);
    float r0 = acc[0], r1 = acc[1], r2 = acc[2], r3 = acc[3];
    __builtin_memcpy(&c[0], &r0, 4);
    __builtin_memcpy(&c[1], &r1, 4);
    __builtin_memcpy(&c[2], &r2, 4);
    __builtin_memcpy(&c[3], &r3, 4);
}

#endif  // PTX_MMA_CUH
