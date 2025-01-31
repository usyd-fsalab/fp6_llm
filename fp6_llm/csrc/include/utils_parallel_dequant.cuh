#ifndef UTILS_PARALLELDEQUANT_CUH
#define UTILS_PARALLELDEQUANT_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/*
 * Input:   R1
 * Outputs: R1, R2
 * Note:    Simplified Exponent calculation is applied.
 */
template<int EXPONENT, int MANTISSA>
__device__ __forceinline__ void FPx_FP16_Cast_4Way(u_int32_t *In, u_int32_t *Out1, u_int32_t *Out2) {
    //
    constexpr int RIGHT_SHIFT = 5 - EXPONENT;
    constexpr int MASK1 = 0x80000000;
    constexpr int MASK2 = MASK1 >> EXPONENT + MANTISSA;
    constexpr int MASK3 = MASK2 & 0x7fffffff;
    constexpr int MASK  = MASK3 | MASK3 >> 16;
    //
    *Out1  = *In & 0x80008000;
    *Out1 |= ( (*In) & MASK ) >> RIGHT_SHIFT;
    //
    *In    = (*In) << 8;
    *Out2  = *In & 0x80008000;
    *Out2 |= ( (*In) & MASK ) >> RIGHT_SHIFT;
}

template<int EXPONENT, int MANTISSA>
__device__ __forceinline__ u_int32_t MultScale(u_int32_t PackedFP16Pair, half Scale) {
    constexpr int BIAS_OFFSET = (int(1) << (5-1)) - (int(1) << (EXPONENT-1));
    constexpr int BIAS        = int(1) << BIAS_OFFSET;
    //
    half* FP16_1 = reinterpret_cast<half*>(&PackedFP16Pair);
    half* FP16_2 = FP16_1 + 1;
    uint32_t output;
    half* output_half_ptr = reinterpret_cast<half*>(&output);
    output_half_ptr[0] = __hmul( __hmul(*FP16_1,__float2half(1.0f*BIAS)), Scale);
    output_half_ptr[1] = __hmul( __hmul(*FP16_2,__float2half(1.0f*BIAS)), Scale);   
    return output;
}

template<int EXPONENT, int MANTISSA>
__device__ __forceinline__ void Dequant_32FP6_4Way(u_int32_t __restrict__   Reg[][4], 
                                                   u_int32_t __restrict__   *read_RPTR_1bit,
                                                   u_int32_t __restrict__   *read_RPTR_2bit, 
                                                   u_int32_t __restrict__   *read_RPTR_4bit,
                                                   u_int32_t                *Scales) {
    // 1+2+4 weight split
    constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;
    //
    u_int32_t *OutputRegs    = reinterpret_cast<u_int32_t*> (Reg);
    u_int32_t *Frag_PTR_1bit = read_RPTR_1bit;
    u_int32_t *Frag_PTR_2bit = read_RPTR_2bit;
    u_int32_t *Frag_PTR_4bit = read_RPTR_4bit;
    half      *Scale_RPTR    = reinterpret_cast<half*>(Scales);
    // Dequantizing 32 FP6, each Loop dequantizing 4 FP6
    #pragma unroll(8)
    for(int i=0; i<8; i++) { 
        u_int32_t Packed_FP6 = 0;
        u_int32_t tmp        = 0;
        // 1bit Frag
        if(USE_SEG_1BIT) {
            tmp = (*Frag_PTR_1bit) & 0x80808080;
            Packed_FP6 |= tmp >> (BIT_WIDTH & 0);
            if(i%8==7)  Frag_PTR_1bit++;
            else        (*Frag_PTR_1bit) = (*Frag_PTR_1bit) << 1;
        }
        // 2bit Frag
        if(USE_SEG_2BIT) {
            tmp = (*Frag_PTR_2bit) & 0xc0c0c0c0;
            Packed_FP6 |= tmp >> (BIT_WIDTH & 1);
            if(i%4==3)  Frag_PTR_2bit++;
            else        (*Frag_PTR_2bit) = (*Frag_PTR_2bit) << 2;
        }
        // 4bit Frag2
        if(USE_SEG_4BIT) {
            tmp = (*Frag_PTR_4bit) & 0xf0f0f0f0;
            Packed_FP6 |= tmp >> (BIT_WIDTH & 3);
            if(i%2==1)  Frag_PTR_4bit++;
            else        (*Frag_PTR_4bit) = (*Frag_PTR_4bit) << 4;
        }
        //
        u_int32_t out1, out2;
        FPx_FP16_Cast_4Way<EXPONENT, MANTISSA>(&Packed_FP6, &out1, &out2);
        //
        *OutputRegs = MultScale<EXPONENT, MANTISSA>(out1, Scale_RPTR[0]  );       // Muliply FP16 scales
        OutputRegs += 1;
        *OutputRegs = MultScale<EXPONENT, MANTISSA>(out2, Scale_RPTR[1]);         // Muliply FP16 scales
        OutputRegs += 1;
        // Updating offset for FP16 scales for every two iterations
        if(i%2==1)  Scale_RPTR += 2;
    }
    
}

/*
 * 
 */
__device__ __forceinline__ void ExtractFromSharedToReg_Scales(uint32_t* Scales, half* WARP_SPTR_Scales) {
    int lane_id = threadIdx.x % WARP_SIZE;
    uint32_t* SPTR_uint = reinterpret_cast<uint32_t*>(WARP_SPTR_Scales);
    uint32_t tmpReg = SPTR_uint[lane_id];
    #pragma unroll
    for(int i=0; i<4; i++) {
        // T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize); 
        Scales[i] = __shfl_sync(0xffffffff, tmpReg, i, 4); 
    }
}

#endif