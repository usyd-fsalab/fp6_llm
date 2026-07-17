#ifndef CONFIGS_H
#define CONFIGS_H

//#define DEBUG_MODE
#define PIPELINE_LEVEL_GMEM 2
#define PIPELINE_LEVEL_SMEM 2       // only support 2

/************************ Hardware Parameters ************************/
// AMD CDNA (gfx942 MI300X/MI325X): wavefront = 64 threads
// The MMA tile shape (16x8x16) still uses 32-lane groups internally,
// but we must account for the 64-thread wavefront in warp-level indexing.
#ifdef __HIP_PLATFORM_AMD__
  #define WARP_SIZE                         64   // AMD CDNA wavefront width
#else
  #define WARP_SIZE                         32   // NVIDIA warp width
#endif
#define REG_BIT_WIDTH                       32
// mma: M=16 K=16 N=8  (AMD MFMA v_mfma_f32_16x16x16f16 geometry)
#define MMA_8                               8
#define MMA_16                              16
// for memory access
#define THREAD_OPT_ACCESS_BIT_WIDTH_128     128 // LDS.128, global load 128-bit, ...
#define BIT_WIDTH_PER_HALF                  16  // Half precision: FP16

/******************** Register Allocation For GEMM ********************/
#define REG_PER_THREAD_C_TENSOR_16_16       8   // 8 for FP32 Accumulation
/********************** Memory Padding Parameters **********************/
// Eliminating bank-conflict
#define PADDING_BYTES_16                    16 // Padding 16 bytes each column
#define PADDING_SHARED_MEM_FOR_B_8          8  // Padding 8 half each column
#define PADDING_SHARED_MEM_FOR_C_4          4  // Padding 4 float each column
/************************* WARP Tiling part-1 *************************/
// On AMD, each 64-thread wavefront covers the same M=64 tile as two 32-thread CUDA warps
// We keep WARP_ROW_MMA_TENSORS=4 (4 x MMA_16 = 64 rows) — matches MFMA 16x16x16 tiling
#define WARP_ROW_MMA_TENSORS                4
#define WARP_M                              (WARP_ROW_MMA_TENSORS * MMA_16)       // 64
#define WARP_K_MMA_TENSORS                  4
#define WARP_K                              (WARP_K_MMA_TENSORS   * MMA_16)       // 64

template<int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_MMA_TENSORS_>
struct TilingConfig {
    static constexpr int BLOCK_ROW_WARPS        = BLOCK_ROW_WARPS_;
    static constexpr int BLOCK_COL_WARPS        = BLOCK_COL_WARPS_;
    static constexpr int WARP_COL_MMA_TENSORS   = WARP_COL_MMA_TENSORS_;
    /************************* WARP Tiling part-2 *************************/
    static constexpr int WARP_N                 = WARP_COL_MMA_TENSORS * MMA_8;
    /*************************Thread Block Tiling *************************/
    static constexpr int TILE_M                 = WARP_M * BLOCK_ROW_WARPS;
    static constexpr int TILE_N                 = MMA_8  * WARP_COL_MMA_TENSORS * BLOCK_COL_WARPS;
    static constexpr int TILE_K                 = WARP_K;
    /********************** #Thread per Thread Block **********************/
    static constexpr int BLOCK_WARPS   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
    /******************************* Others *******************************/
    static constexpr int SMEM_SIZE_B_TILE   = TILE_N * (TILE_K + PADDING_BYTES_16) * 2 * PIPELINE_LEVEL_GMEM;
    static constexpr int SMEM_SIZE_C_TILE   = TILE_N * (TILE_M + PADDING_BYTES_16) * 4;
};

#endif  // CONFIGS_H
