#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.h"

template<int EXPONENT, int MANTISSA>
void DeQuantMatrix_FPx_To_FP16(half* A_16bit_h, unsigned char* A_x_bit_h, size_t M, size_t K, half* scale) {
    //
    assert(M%64==0);                 // Currently, M must be a multiple of 64.
    assert(K%64==0);                 // Currently, K must be a multiple of 64.
    constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    assert(BIT_WIDTH<=8);
    size_t TotalSizeInByte = M * K * BIT_WIDTH / 8;
    //
    half* OutPTR = A_16bit_h;
    for(size_t i=0; i<TotalSizeInByte/BIT_WIDTH; i++) {    // Processing BIT_WIDTH Bytes for each Loop, generating 8 FP16.
        unsigned char Bytes[BIT_WIDTH];
        for(int x=0; x<BIT_WIDTH; x++)  Bytes[x] = A_x_bit_h[i*BIT_WIDTH+x];
        unsigned char OUT[8];
        for(int x=0; x<8; x++) {                        // Prepare Initial memory layout for Dequant
            int ByteOffset  = BIT_WIDTH * x / 8;
            int BitOffset   = BIT_WIDTH * x % 8;
            OUT[x] = Extract_X_Bits_To_A_Byte<EXPONENT, MANTISSA>(Bytes, ByteOffset, BitOffset);
        }
        // Dequant
        constexpr int MASK1 = 0x80000000;
        constexpr int MASK2 = MASK1 >> EXPONENT + MANTISSA;
        constexpr int MASK  = MASK2 & 0x7fffffff;
        constexpr int RIGHT_SHIFT = 5 - EXPONENT;
        constexpr int BIAS_OFFSET = (int(1) << (5-1)) - (int(1) << (EXPONENT-1));
        constexpr int BIAS        = int(1) << BIAS_OFFSET;
        for(int x=0; x<8; x++) {
            unsigned int OUT_fp16;        // Storing fp16 in the high 16 bits.
            OUT_fp16 = int(OUT[x]) << 24;
            OUT_fp16 = (OUT_fp16 & 0x80000000) | ( (OUT_fp16 & MASK) >> RIGHT_SHIFT );
            OUT_fp16 = OUT_fp16 >> 16;
            //
            half* OUT_FP16_PTR = reinterpret_cast<half*>(&OUT_fp16);
            OutPTR[x] = __float2half_rn ( __half2float(*OUT_FP16_PTR) * (1.0f*BIAS) * __half2float(scale[(8*i)/K]) );
        }   
        //
        OutPTR +=8;
    }
}


void DeQuantMatrix_FP6_To_FP16(half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale) {
    DeQuantMatrix_FPx_To_FP16<3, 2>(A_16bit_h, A_6bit_h, M, K, scale);
}
void dequant_matrix_fp_eXmY_to_fp16(const int EXPONENT, const int MANTISSA, half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale){
    if(EXPONENT==2 && MANTISSA==2)
        return DeQuantMatrix_FPx_To_FP16<2, 2>(A_16bit_h, A_6bit_h, M, K, scale);
    if(EXPONENT==3 && MANTISSA==2)
        return DeQuantMatrix_FPx_To_FP16<3, 2>(A_16bit_h, A_6bit_h, M, K, scale);
    printf("DeQuantMatrix Error: Unsupported EXPONENT=%d, MANTISSA=%d!\n", EXPONENT, MANTISSA);
    exit(-1);
}