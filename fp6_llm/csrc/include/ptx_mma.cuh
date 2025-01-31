/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#ifndef PTX_MMA_CUH
#define PTX_MMA_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <assert.h>
#include "configs.h"

template <typename TilingConfig>
__device__ __forceinline__ void B_FromSharedToReg(uint32_t  __restrict__    Reg[][4],
                                                  half      __restrict__    (*read_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                  int                       slice_id) {
    #ifdef DEBUG_MODE
        static_assert( (TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0) );
    #endif
    
    const int   warpId  = threadIdx.x / WARP_SIZE;
    int         lane_id = threadIdx.x % WARP_SIZE;
    int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_col = TilingConfig::WARP_COL_MMA_TENSORS * MMA_8 * WARP_j;   // each warp may start from reading warp_start_col'th column of the B tile in shared memory
    #ifdef DEBUG_MODE
        assert( warp_start_col==0 );
    #endif    

    int col = (lane_id%8) + (lane_id/16)*8;
    int row = (lane_id%16) / 8 * 8;
    uint32_t smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&read_SPTR[warp_start_col+col][slice_id*MMA_16 + row]));
    if(TilingConfig::WARP_COL_MMA_TENSORS==1) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(Reg[0][0]), "=r"(Reg[0][1])
                     : "r"(smem_local_ptr));
    }
    else {
        #pragma unroll
        for (int i = 0; i < TilingConfig::WARP_COL_MMA_TENSORS/2; i++)
        {
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(Reg[i][0]), "=r"(Reg[i][1]), "=r"(Reg[i][2]), "=r"(Reg[i][3])
                         : "r"(smem_local_ptr));
            smem_local_ptr += 16 * (WARP_K+PADDING_SHARED_MEM_FOR_B_8) * sizeof(half);
        }
    }
}

__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]),
                   "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

#endif