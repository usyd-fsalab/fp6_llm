#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include "common.h"

/*
 * Inputs:
 * (1) unsigned char Weight_6bit [M*K*6/8]
 * Outputs:
 * (1) unsigned char Weight_2bit [M*K*2/8]
 * (2) unsigned char Weight_4bit [M*K*4/8]
 * 
 * Assumption: Weight_6bit, Weight_2bit, Weight_4bit all stored continuously in row-major.
 * 8 FP6 = 6 Bytes
 * 8 FP4 = 4 Bytes
 * 8 FP2 = 2 Bytes
 */

using namespace std;


void Extract_segments_from_8_padded_fpx(unsigned char Seg_xbit[], unsigned char Padded_8_FPx[], int bit_width, int bit_offset){
    for(int i=0; i< bit_width; i++)
        Seg_xbit[i] = 0;
    for(int i=0; i<8; i++){
        unsigned int seg = (Padded_8_FPx[i] << bit_offset) & 0x000000ff;
        int mask = 0xffffff00;
        seg &= mask >> bit_width;
        //
        int Seg_idx = (i * bit_width) / 8;
        int Seg_off = (i * bit_width) % 8;
        Seg_xbit[Seg_idx] |= seg >> Seg_off;
    }
}


// dealing with 4 1*8 blocks of FPx
template<int EXPONENT, int MANTISSA>
void Assign_32_FPx_To_4_Thread(vector<unsigned char> Vec_Seg_1bit[], vector<unsigned char> Vec_Seg_2bit[], vector<unsigned char> Vec_Seg_4bit[], unsigned char* PTR[]) 
{
    constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    assert(BIT_WIDTH<8);
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;
    //
    constexpr int nTHREADS = 4;
    constexpr int FPx_PER_THREAD = 8;
    unsigned char Padded_8_FPx[nTHREADS][FPx_PER_THREAD];
    for(int i=0; i<nTHREADS; i++){                             // 4 threads
        for(int j=0; j<FPx_PER_THREAD; j++){                   // 8 FPx per thread
            int offset = (i*2 + j%2) * BIT_WIDTH;
            int ByteOffset = offset / 8;
            int BitOffset  = offset % 8;
            Padded_8_FPx[i][j] = Extract_X_Bits_To_A_Byte<EXPONENT, MANTISSA>(PTR[j/2], ByteOffset, BitOffset);
        }
    }
    //
    unsigned char Seg_1bit[nTHREADS][1];
    unsigned char Seg_2bit[nTHREADS][2];
    unsigned char Seg_4bit[nTHREADS][4];
    for(int t=0; t<nTHREADS; t++){
        Extract_segments_from_8_padded_fpx(Seg_1bit[t], Padded_8_FPx[t], 1, int(BIT_WIDTH & 0));
        Extract_segments_from_8_padded_fpx(Seg_2bit[t], Padded_8_FPx[t], 2, int(BIT_WIDTH & 1));
        Extract_segments_from_8_padded_fpx(Seg_4bit[t], Padded_8_FPx[t], 4, int(BIT_WIDTH & 3));
    }
    //
    for(int t=0; t<4; t++)
    {
        if (USE_SEG_1BIT) {
            Vec_Seg_1bit[t].push_back(Seg_1bit[t][0]);
        }
        if (USE_SEG_2BIT) {
            Vec_Seg_2bit[t].push_back(Seg_2bit[t][0]);
            Vec_Seg_2bit[t].push_back(Seg_2bit[t][1]);
        }
        if (USE_SEG_4BIT) {
            Vec_Seg_4bit[t].push_back(Seg_4bit[t][0]);
            Vec_Seg_4bit[t].push_back(Seg_4bit[t][1]);
            Vec_Seg_4bit[t].push_back(Seg_4bit[t][2]);
            Vec_Seg_4bit[t].push_back(Seg_4bit[t][3]);
        }
    }
}

template<int BIT_WIDTH>
void BitInterleaving_x_bit(unsigned char* PTR_4Bytes)
{
    unsigned int *PTR_UINT = reinterpret_cast<unsigned int*>(PTR_4Bytes);
    unsigned int input  = *PTR_UINT;
    //
    int* order = NULL;
    int order_1bit[32] = {2,6,10,14,18,22,26,30,
                          4,8,12,16,20,24,28,32,
                          1,5,9, 13,17,21,25,29,
                          3,7,11,15,19,23,27,31};  // pre-defined order for bit-interleaving in FP6-LLM
    int order_2bit[16] = {2,6,10,14,4,8,12,16,1,5,9,13,3,7,11,15};  // pre-defined order for bit-interleaving in FP6-LLM
    int order_4bit[8] = {2,6,4,8,1,5,3,7};  // pre-defined order for bit-interleaving in FP6-LLM
    if(BIT_WIDTH==1) order = order_1bit;
    if(BIT_WIDTH==2) order = order_2bit;
    if(BIT_WIDTH==4) order = order_4bit;
    assert(order);
    //
    int mask = 0x80000000;
    assert(BIT_WIDTH>=1);
    mask = mask >> (BIT_WIDTH-1);
    //
    unsigned int output = 0x00000000;
    for(int i=0; i<32/BIT_WIDTH; i++){
        unsigned int Frag_xbit = ( input << BIT_WIDTH*(order[i]-1) ) & mask;    // The highest x bits are used to store the extracted fragments.
        output |= Frag_xbit >> (i*BIT_WIDTH);
    }
    //
    *PTR_UINT = output;
}

template<int EXPONENT, int MANTISSA>
void weight_matrix_prepacking_x_bit(int* packed_weights, int *FPxWeights, size_t M, size_t K)
{
    assert(M % 64 == 0);
    assert(K % 64 == 0);
    //
    constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    assert(BIT_WIDTH<8);
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;
    //
    unsigned char* Weight_xbit = reinterpret_cast<unsigned char*>(FPxWeights);
    unsigned char* Weight_1bit = reinterpret_cast<unsigned char*>(packed_weights);
    unsigned char* Weight_2bit = Weight_1bit + (USE_SEG_1BIT ? M*K*1/8 : 0);
    unsigned char* Weight_4bit = Weight_2bit + (USE_SEG_2BIT ? M*K*2/8 : 0);
    //
    vector<unsigned char> A_Segment_1bit[32];
    vector<unsigned char> A_Segment_2bit[32];
    vector<unsigned char> A_Segment_4bit[32];
    //
    size_t BytesPerRow = K*BIT_WIDTH/8;
    // Pass-1: (1) 1+2+4 split; (2) assign weights to 32 threads.
    for (size_t i = 0; i < M / 64; i++){
        for (size_t j = 0; j < K / 16; j++){
            for(size_t k=0; k<64/16; k++){
                size_t row = i*64 + k*16;
                size_t col = j*16;
                unsigned char* StartPTR_1 = Weight_xbit + row*BytesPerRow + col*(BIT_WIDTH)/8;
                unsigned char* StartPTR_2 = StartPTR_1 + 8*BytesPerRow;
                unsigned char* StartPTR_3 = StartPTR_1 + 8*(BIT_WIDTH)/8;
                unsigned char* StartPTR_4 = StartPTR_2 + 8*(BIT_WIDTH)/8;
                // Dealing with each 16*16 blocks then...
                for(int l=0; l<8; l++) {
                    unsigned char* PTR[4]={StartPTR_1+l*BytesPerRow, StartPTR_2+l*BytesPerRow, StartPTR_3+l*BytesPerRow, StartPTR_4+l*BytesPerRow};
                    Assign_32_FPx_To_4_Thread<EXPONENT,MANTISSA>(&A_Segment_1bit[l*4], &A_Segment_2bit[l*4], &A_Segment_4bit[l*4], PTR);
                }
            }
        }
    }
    // Verifying the length of 1/2/4_bit segments.
    size_t BytesPerThread_1bit = M*K*1/8/32;
    size_t BytesPerThread_2bit = M*K*2/8/32;
    size_t BytesPerThread_4bit = M*K*4/8/32;
    for(int i=0; i<32; i++){
        if(USE_SEG_1BIT)    assert(A_Segment_1bit[i].size()==BytesPerThread_1bit);
        else                assert(A_Segment_1bit[i].size()==0);
        if(USE_SEG_2BIT)    assert(A_Segment_2bit[i].size()==BytesPerThread_2bit);
        else                assert(A_Segment_2bit[i].size()==0);
        if(USE_SEG_4BIT)    assert(A_Segment_4bit[i].size()==BytesPerThread_4bit);
        else                assert(A_Segment_4bit[i].size()==0);
    }
    // Pass-2: Optimizing coleasced global memory access
    if(USE_SEG_1BIT)
    for(size_t i=0; i<BytesPerThread_1bit/4; i++) 
        for(int t=0; t<32; t++)
            for(int b=0; b<4; b++)              // why (3-b): special byte order within a register
                Weight_1bit[i*128+t*4+(3-b)] = A_Segment_1bit[t][i*4+b];    
    if(USE_SEG_2BIT)
    for(size_t i=0; i<BytesPerThread_2bit/4; i++) 
        for(int t=0; t<32; t++)
            for(int b=0; b<4; b++)              // why (3-b): special byte order within a register
                Weight_2bit[i*128+t*4+(3-b)] = A_Segment_2bit[t][i*4+b];    
    if(USE_SEG_4BIT)
    for(size_t i=0; i<BytesPerThread_4bit/4; i++) 
        for(int t=0; t<32; t++)
            for(int b=0; b<4; b++)              // why (3-b):special byte order within a register
                Weight_4bit[i*128+t*4+(3-b)] = A_Segment_4bit[t][i*4+b];
    // Pass-3: Bit-level interleaving
    if(USE_SEG_1BIT)
    for(size_t i=0; i<BytesPerThread_1bit*32/4; i++)
        BitInterleaving_x_bit<1>(Weight_1bit+4*i);
    if(USE_SEG_2BIT)
    for(size_t i=0; i<BytesPerThread_2bit*32/4; i++)
        BitInterleaving_x_bit<2>(Weight_2bit+4*i);
    if(USE_SEG_4BIT)
    for(size_t i=0; i<BytesPerThread_4bit*32/4; i++)
        BitInterleaving_x_bit<4>(Weight_4bit+4*i);
}


void weight_matrix_prepacking(int* packed_weights, int *FP6Weights, size_t M, size_t K){
    weight_matrix_prepacking_x_bit<3, 2>(packed_weights, FP6Weights, M, K);
}

//
void weight_matrix_prepacking_fp_eXmY(const int EXPONENT, const int MANTISSA, int* packed_weights, int *FPxWeights, size_t M, size_t K){
    if(EXPONENT==2 && MANTISSA==2)
        return weight_matrix_prepacking_x_bit<2, 2>(packed_weights, FPxWeights, M, K);
    if(EXPONENT==3 && MANTISSA==2)
        return weight_matrix_prepacking_x_bit<3, 2>(packed_weights, FPxWeights, M, K);
    printf("Weight_prepacking Error: Unsupported EXPONENT=%d, MANTISSA=%d!\n", EXPONENT, MANTISSA);
    exit(-1);
}