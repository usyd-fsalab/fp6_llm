/*
 * AMD ROCm HIP port of utils_core.cuh
 * No CUDA-specific intrinsics in this file beyond what ptx_mma.cuh provides.
 * Changes: cuda.h/cuda_fp16.h -> included via hip headers in dependencies
 */
#ifndef UTILS_CORE_CUH
#define UTILS_CORE_CUH

#include <assert.h>
#include "configs.h"
#include "ptx_mma.cuh"
#include "utils_parallel_dequant.cuh"

template<int NUM_INT_PER_THREAD>
__device__ __forceinline__ void CopyFromSharedToRegister_AFrag(uint32_t Reg[], uint32_t* SPTR, int slice_id)
{
    SPTR += slice_id * (NUM_INT_PER_THREAD * WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < NUM_INT_PER_THREAD; i++) {
        Reg[i] = SPTR[lane_id + i * WARP_SIZE];
    }
}

template <typename TilingConfig, int EXPONENT, int MANTISSA>
__device__ __forceinline__ void initialize_mma_slice(
    uint32_t                  (*a)[4],
    uint32_t                  (*b)[4],
    uint32_t* __restrict__    A_1BIT_SPTR_read,
    uint32_t* __restrict__    A_2BIT_SPTR_read,
    uint32_t* __restrict__    A_4BIT_SPTR_read,
    half      (* __restrict__  B_SPTR_read)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    uint32_t*                 RPTR_Scales)
{
    constexpr int BIT_WIDTH    = 1 + EXPONENT + MANTISSA;
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;

    uint32_t a_1bit[1];
    uint32_t a_2bit[2];
    uint32_t a_4bit[4];
    if (USE_SEG_1BIT) CopyFromSharedToRegister_AFrag<1>(a_1bit, A_1BIT_SPTR_read, 0);
    if (USE_SEG_2BIT) CopyFromSharedToRegister_AFrag<2>(a_2bit, A_2BIT_SPTR_read, 0);
    if (USE_SEG_4BIT) CopyFromSharedToRegister_AFrag<4>(a_4bit, A_4BIT_SPTR_read, 0);
    Dequant_32FP6_4Way<EXPONENT, MANTISSA>(a, a_1bit, a_2bit, a_4bit, RPTR_Scales);
    B_FromSharedToReg<TilingConfig>(b, B_SPTR_read, 0);
}

template <typename TilingConfig, int EXPONENT, int MANTISSA>
__device__ __forceinline__ void core_mma_slice(
    float                     c[][REG_PER_THREAD_C_TENSOR_16_16],
    uint32_t                  (*a)[4],
    uint32_t                  (*b)[4],
    uint32_t* __restrict__    A_1bit_SPTR_read,
    uint32_t* __restrict__    A_2bit_SPTR_read,
    uint32_t* __restrict__    A_4bit_SPTR_read,
    half      (* __restrict__  B_SPTR_read)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    uint32_t*                 RPTR_Scales,
    int                       slice_id)
{
    constexpr int BIT_WIDTH    = 1 + EXPONENT + MANTISSA;
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;

    const int NumRegSets_a = WARP_ROW_MMA_TENSORS;
    const int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS == 1)
                             ? 1 : TilingConfig::WARP_COL_MMA_TENSORS / 2;
    uint32_t (*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] =
        reinterpret_cast<uint32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(c);

    uint32_t (*a_read )[4] = a;
    uint32_t (*a_write)[4] = a;
    uint32_t (*b_read )[4] = b;
    uint32_t (*b_write)[4] = b;
    if (slice_id % 2 == 1) { b_write += NumRegSets_b; a_write += NumRegSets_a; }
    else                   { b_read  += NumRegSets_b; a_read  += NumRegSets_a; }

    #pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
        if (TilingConfig::WARP_COL_MMA_TENSORS == 1) {
            MMA_FP16_M16N8K16(c_uint_ptr[i], a_read[i], b_read[0]);
        } else {
            #pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS / 2; j++) {
                MMA_FP16_M16N8K16(c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS],     a_read[i], b_read[j]);
                MMA_FP16_M16N8K16(c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS] + 4, a_read[i], b_read[j] + 2);
            }
        }
    }

    uint32_t a_1bit[1];
    uint32_t a_2bit[2];
    uint32_t a_4bit[4];
    if (USE_SEG_1BIT) CopyFromSharedToRegister_AFrag<1>(a_1bit, A_1bit_SPTR_read, slice_id);
    if (USE_SEG_2BIT) CopyFromSharedToRegister_AFrag<2>(a_2bit, A_2bit_SPTR_read, slice_id);
    if (USE_SEG_4BIT) CopyFromSharedToRegister_AFrag<4>(a_4bit, A_4bit_SPTR_read, slice_id);
    Dequant_32FP6_4Way<EXPONENT, MANTISSA>(a_write, a_1bit, a_2bit, a_4bit, RPTR_Scales);
    B_FromSharedToReg<TilingConfig>(b_write, B_SPTR_read, slice_id);
}

template <typename TilingConfig>
__device__ __forceinline__ void StoreToSharedMemoryFromRegister(
    float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4],
    float  c[][REG_PER_THREAD_C_TENSOR_16_16])
{
    const int lane_id         = threadIdx.x % WARP_SIZE;
    const int warpId          = threadIdx.x / WARP_SIZE;
    int warp_row_offset       = warpId * (MMA_16 * WARP_ROW_MMA_TENSORS);
    #pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
        #pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS; j++) {
            int RegSetID          = i + (j / 2) * WARP_ROW_MMA_TENSORS;
            int RegOffset         = (j % 2) * (REG_PER_THREAD_C_TENSOR_16_16 / 2);
            int Tensor_row_offset = warp_row_offset + i * MMA_16;
            int Tensor_col_offset = j * MMA_8;
            #pragma unroll
            for (int r = 0; r < REG_PER_THREAD_C_TENSOR_16_16 / 2; r++) {
                int row_offset = lane_id / 4;
                if (r >= 2) row_offset += 8;
                int col_offset = (lane_id % 4) * 2;
                if (r % 2 == 1) col_offset += 1;
                smem_CFrag[Tensor_col_offset + col_offset][Tensor_row_offset + row_offset] =
                    c[RegSetID][r + RegOffset];
            }
        }
    }
}

#endif  // UTILS_CORE_CUH
