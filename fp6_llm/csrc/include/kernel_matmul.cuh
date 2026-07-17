/*
 * AMD ROCm HIP port of kernel_matmul.cuh
 * Changes:
 *   - cuda.h/cuda_fp16.h/cuda_runtime.h -> hip/hip_runtime.h, hip/hip_fp16.h
 *   - extern __shared__ half smem[] alignment: hip supports __align__
 *   - __shared__ aligned syntax identical in HIP
 *   - std::is_same<> usable in HIP device code (C++17)
 *   - WARP_SIZE now 64 on AMD via configs.h
 */
#include "configs.h"
#include "utils_gmem.cuh"
#include "utils_core.cuh"

/************************** Bitwidth of Weight Segments ************************/
#define BIT_WIDTH_1   1
#define BIT_WIDTH_2   2
#define BIT_WIDTH_4   4
/*************************** 64*64 Weights of Weight Matrix *********************/
#define WEIGHT_PER_WARP           (WARP_M * WARP_K)
#define SMEM_SIZE_PER_WARP_1BIT   (WEIGHT_PER_WARP * BIT_WIDTH_1 / 8)
#define SMEM_SIZE_PER_WARP_2BIT   (WEIGHT_PER_WARP * BIT_WIDTH_2 / 8)
#define SMEM_SIZE_PER_WARP_4BIT   (WEIGHT_PER_WARP * BIT_WIDTH_4 / 8)
#define SMEM_SIZE_PER_TB_1BIT     (SMEM_SIZE_PER_WARP_1BIT * TilingConfig::BLOCK_WARPS * PIPELINE_LEVEL_GMEM)
#define SMEM_SIZE_PER_TB_2BIT     (SMEM_SIZE_PER_WARP_2BIT * TilingConfig::BLOCK_WARPS * PIPELINE_LEVEL_GMEM)
#define SMEM_SIZE_PER_TB_4BIT     (SMEM_SIZE_PER_WARP_4BIT * TilingConfig::BLOCK_WARPS * PIPELINE_LEVEL_GMEM)
#define SMEM_SIZE_PER_TB_A_TILE   (SMEM_SIZE_PER_TB_1BIT + SMEM_SIZE_PER_TB_2BIT + SMEM_SIZE_PER_TB_4BIT)
/******************** Global Memory Layout For QUANTIZED DATA *******************/
#define NUM_INT4_PER_WARP_1BIT    (WEIGHT_PER_WARP * BIT_WIDTH_1 / 128)
#define NUM_INT4_PER_WARP_2BIT    (WEIGHT_PER_WARP * BIT_WIDTH_2 / 128)
#define NUM_INT4_PER_WARP_4BIT    (WEIGHT_PER_WARP * BIT_WIDTH_4 / 128)

template<typename TilingConfig, typename OutputDataType, int EXPONENT, int MANTISSA>
__global__ void QUANT_GEMM_Kernel(
    const uint4*    Weight,
    const half*     Scales,
    const half*     B,
    OutputDataType* C,
    const size_t    M_Global,
    const size_t    N_Global,
    const size_t    K_Global,
    int             Split_K)
{
#ifdef DEBUG_MODE
    assert(K_Global % TilingConfig::TILE_K == 0);
    assert(M_Global % TilingConfig::TILE_M == 0);
    assert(gridDim.y == Split_K * (M_Global / TilingConfig::TILE_M));
#endif
    constexpr int BIT_WIDTH    = 1 + EXPONENT + MANTISSA;
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;

    const uint4* Weight_1bit = Weight;
    const uint4* Weight_2bit = Weight_1bit + (USE_SEG_1BIT ? M_Global * K_Global * BIT_WIDTH_1 / 128 : 0);
    const uint4* Weight_4bit = Weight_2bit + (USE_SEG_2BIT ? M_Global * K_Global * BIT_WIDTH_2 / 128 : 0);

    extern __shared__ __align__(128) half smem[];
    half (*smem_array)[WARP_K + PADDING_SHARED_MEM_FOR_B_8] =
        reinterpret_cast<half (*)[WARP_K + PADDING_SHARED_MEM_FOR_B_8]>(
            smem + SMEM_SIZE_PER_TB_A_TILE / 2);
    __shared__ half QuantScales[64 * TilingConfig::BLOCK_WARPS];

    const size_t BatchID       = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const size_t x             = blockIdx.x;
    const size_t y             = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    const size_t Tile_Start_M  = y * TilingConfig::TILE_M;
    const size_t Tile_Start_N  = x * TilingConfig::TILE_N;
    const size_t NumColumnToCopy = (N_Global - Tile_Start_N) < TilingConfig::TILE_N
                                   ? (N_Global - Tile_Start_N) : TilingConfig::TILE_N;
    const size_t NumBlock_K        = K_Global / TilingConfig::TILE_K;
    const size_t AverageNumBlock_K = NumBlock_K / Split_K;
    const size_t ExtraNumBlock_K   = NumBlock_K - AverageNumBlock_K * Split_K;
    size_t NumIter       = AverageNumBlock_K;
    size_t StartBlockID_K = AverageNumBlock_K * BatchID;
    if (BatchID < ExtraNumBlock_K) { NumIter++; StartBlockID_K += BatchID; }
    else                            StartBlockID_K += ExtraNumBlock_K;

    const int warpId = threadIdx.x / WARP_SIZE;
    int WARP_i = warpId / TilingConfig::BLOCK_COL_WARPS;

    const uint4* TB_StartGPTR_A_1BIT = Weight_1bit + (y * TilingConfig::BLOCK_ROW_WARPS) * NumBlock_K * NUM_INT4_PER_WARP_1BIT;
    const uint4* TB_StartGPTR_A_2BIT = Weight_2bit + (y * TilingConfig::BLOCK_ROW_WARPS) * NumBlock_K * NUM_INT4_PER_WARP_2BIT;
    const uint4* TB_StartGPTR_A_4BIT = Weight_4bit + (y * TilingConfig::BLOCK_ROW_WARPS) * NumBlock_K * NUM_INT4_PER_WARP_4BIT;

    const uint4* WARP_StartGPTR_A_1BIT = TB_StartGPTR_A_1BIT + WARP_i * NumBlock_K * NUM_INT4_PER_WARP_1BIT;
    const uint4* WARP_StartGPTR_A_2BIT = TB_StartGPTR_A_2BIT + WARP_i * NumBlock_K * NUM_INT4_PER_WARP_2BIT;
    const uint4* WARP_StartGPTR_A_4BIT = TB_StartGPTR_A_4BIT + WARP_i * NumBlock_K * NUM_INT4_PER_WARP_4BIT;

    const size_t WARP_Start_UnitID_K = StartBlockID_K;
    WARP_StartGPTR_A_1BIT += WARP_Start_UnitID_K * NUM_INT4_PER_WARP_1BIT;
    WARP_StartGPTR_A_2BIT += WARP_Start_UnitID_K * NUM_INT4_PER_WARP_2BIT;
    WARP_StartGPTR_A_4BIT += WARP_Start_UnitID_K * NUM_INT4_PER_WARP_4BIT;

    uint32_t* AFrag_1BIT_SPTR = reinterpret_cast<uint32_t*>(smem);
    uint32_t* AFrag_2BIT_SPTR = AFrag_1BIT_SPTR + SMEM_SIZE_PER_TB_1BIT / 4;
    uint32_t* AFrag_4BIT_SPTR = AFrag_2BIT_SPTR + SMEM_SIZE_PER_TB_2BIT / 4;
    AFrag_1BIT_SPTR += warpId * SMEM_SIZE_PER_WARP_1BIT / 4;
    AFrag_2BIT_SPTR += warpId * SMEM_SIZE_PER_WARP_2BIT / 4;
    AFrag_4BIT_SPTR += warpId * SMEM_SIZE_PER_WARP_4BIT / 4;

    for (int i = 0; i < PIPELINE_LEVEL_GMEM - 1; i++) {
        if (USE_SEG_1BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_1BIT>(AFrag_1BIT_SPTR + i * SMEM_SIZE_PER_WARP_1BIT / 4 * 4, WARP_StartGPTR_A_1BIT);
        if (USE_SEG_2BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_2BIT>(AFrag_2BIT_SPTR + i * SMEM_SIZE_PER_WARP_2BIT / 4 * 4, WARP_StartGPTR_A_2BIT);
        if (USE_SEG_4BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_4BIT>(AFrag_4BIT_SPTR + i * SMEM_SIZE_PER_WARP_4BIT / 4 * 4, WARP_StartGPTR_A_4BIT);
        WARP_StartGPTR_A_1BIT += SMEM_SIZE_PER_WARP_1BIT / 16;
        WARP_StartGPTR_A_2BIT += SMEM_SIZE_PER_WARP_2BIT / 16;
        WARP_StartGPTR_A_4BIT += SMEM_SIZE_PER_WARP_4BIT / 16;
    }

    const half* TB_StartGPTR_A_Scale    = Scales + (y * TilingConfig::BLOCK_ROW_WARPS) * 64;
    const half* WARP_StartGPTR_A_Scales = TB_StartGPTR_A_Scale + WARP_i * 64;
    CopyFromGlobalToShared_Scales(QuantScales + WARP_i * 64, WARP_StartGPTR_A_Scales);

    const half* BTile_GPTR = B + Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K;
    for (int i = 0; i < PIPELINE_LEVEL_GMEM - 1; i++) {
        CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS>(
            smem_array + i * TilingConfig::TILE_N, BTile_GPTR, K_Global, NumColumnToCopy);
        BTile_GPTR += TilingConfig::TILE_K;
    }

    constexpr int NumRegSets_a = WARP_ROW_MMA_TENSORS;
    constexpr int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS == 1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS / 2;
    uint32_t a[NumRegSets_a * PIPELINE_LEVEL_SMEM][4];
    uint32_t b[NumRegSets_b * PIPELINE_LEVEL_SMEM][4];
    float    c[NumRegSets_a * NumRegSets_b][REG_PER_THREAD_C_TENSOR_16_16];
    for (int i = 0; i < NumRegSets_a * NumRegSets_b; i++)
        for (int j = 0; j < REG_PER_THREAD_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;

    cp_async_wait_all();
    __syncthreads();

    uint32_t Scales_RPTR[4];
    ExtractFromSharedToReg_Scales(Scales_RPTR, QuantScales + WARP_i * 64);
    initialize_mma_slice<TilingConfig, EXPONENT, MANTISSA>(a, b, AFrag_1BIT_SPTR, AFrag_2BIT_SPTR, AFrag_4BIT_SPTR, smem_array, Scales_RPTR);

    #pragma unroll(1)
    for (size_t tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        uint32_t* __restrict__ read_SPTR_Frag_1bit  = AFrag_1BIT_SPTR + ((tile_id_k + 0) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_1BIT / 4 * 4;
        uint32_t* __restrict__ read_SPTR_Frag_2bit  = AFrag_2BIT_SPTR + ((tile_id_k + 0) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_2BIT / 4 * 4;
        uint32_t* __restrict__ read_SPTR_Frag_4bit  = AFrag_4BIT_SPTR + ((tile_id_k + 0) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_4BIT / 4 * 4;
        uint32_t* __restrict__ read2_SPTR_Frag_1bit = AFrag_1BIT_SPTR + ((tile_id_k + 1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_1BIT / 4 * 4;
        uint32_t* __restrict__ read2_SPTR_Frag_2bit = AFrag_2BIT_SPTR + ((tile_id_k + 1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_2BIT / 4 * 4;
        uint32_t* __restrict__ read2_SPTR_Frag_4bit = AFrag_4BIT_SPTR + ((tile_id_k + 1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_4BIT / 4 * 4;
        uint32_t* __restrict__ write_SPTR_Frag_1bit = AFrag_1BIT_SPTR + ((tile_id_k + (PIPELINE_LEVEL_GMEM - 1)) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_1BIT / 4 * 4;
        uint32_t* __restrict__ write_SPTR_Frag_2bit = AFrag_2BIT_SPTR + ((tile_id_k + (PIPELINE_LEVEL_GMEM - 1)) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_2BIT / 4 * 4;
        uint32_t* __restrict__ write_SPTR_Frag_4bit = AFrag_4BIT_SPTR + ((tile_id_k + (PIPELINE_LEVEL_GMEM - 1)) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_4BIT / 4 * 4;

        half (* __restrict__ read_SPTR )[WARP_K + PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k + 0) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
        half (* __restrict__ read2_SPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k + 1) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
        half (* __restrict__ write_SPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k + (PIPELINE_LEVEL_GMEM - 1)) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;

        bool GlobalCopy = (tile_id_k + PIPELINE_LEVEL_GMEM - 1) < NumIter;

        if (USE_SEG_1BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_1BIT>(write_SPTR_Frag_1bit, WARP_StartGPTR_A_1BIT, GlobalCopy);
        if (USE_SEG_2BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_2BIT>(write_SPTR_Frag_2bit, WARP_StartGPTR_A_2BIT, GlobalCopy);
        if (USE_SEG_4BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_4BIT>(write_SPTR_Frag_4bit, WARP_StartGPTR_A_4BIT, GlobalCopy);
        CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS>(write_SPTR, BTile_GPTR, K_Global, NumColumnToCopy, GlobalCopy);
        cp_async_group_commit();

        core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read_SPTR_Frag_1bit,  read_SPTR_Frag_2bit,  read_SPTR_Frag_4bit,  read_SPTR,  Scales_RPTR, 1);
        core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read_SPTR_Frag_1bit,  read_SPTR_Frag_2bit,  read_SPTR_Frag_4bit,  read_SPTR,  Scales_RPTR, 2);
        core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read_SPTR_Frag_1bit,  read_SPTR_Frag_2bit,  read_SPTR_Frag_4bit,  read_SPTR,  Scales_RPTR, 3);

        cp_async_wait_group<PIPELINE_LEVEL_GMEM - 2>();
        __syncthreads();

        core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read2_SPTR_Frag_1bit, read2_SPTR_Frag_2bit, read2_SPTR_Frag_4bit, read2_SPTR, Scales_RPTR, 0);

        WARP_StartGPTR_A_1BIT += SMEM_SIZE_PER_WARP_1BIT / 16;
        WARP_StartGPTR_A_2BIT += SMEM_SIZE_PER_WARP_2BIT / 16;
        WARP_StartGPTR_A_4BIT += SMEM_SIZE_PER_WARP_4BIT / 16;
        BTile_GPTR += TilingConfig::TILE_K;
    }

    float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4] =
        reinterpret_cast<float (*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();

    OutputDataType* BlockGlobalPTR =
        C + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
    for (size_t i = warpId; i < NumColumnToCopy; i += TilingConfig::BLOCK_WARPS)
        #pragma unroll
        for (size_t j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE) {
            if constexpr (std::is_same<OutputDataType, half>::value)
                BlockGlobalPTR[j + i * M_Global] = __float2half_rn(smem_CFrag[i][j]);
            else
                BlockGlobalPTR[j + i * M_Global] = smem_CFrag[i][j];
        }
}
