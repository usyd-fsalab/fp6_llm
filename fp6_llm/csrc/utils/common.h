#ifndef UTILS_COMMON_H
#define UTILS_COMMON_H

template<int EXPONENT, int MANTISSA>
unsigned char Extract_X_Bits_To_A_Byte(unsigned char* Bytes, int ByteOffset, int BitOffset){
    assert (sizeof(unsigned int)==4);
    unsigned int tmp_int32_word=0;
    unsigned char* uchar_ptr = reinterpret_cast<unsigned char*>(&tmp_int32_word);
    uchar_ptr[3] = Bytes[ByteOffset+0];
    uchar_ptr[2] = Bytes[ByteOffset+1];
    tmp_int32_word = tmp_int32_word << BitOffset;
    //
    signed int mask = 0x80000000;
    mask = mask >> (EXPONENT+MANTISSA);
    tmp_int32_word &= mask;
    //
    unsigned char out = uchar_ptr[3];
    return out;
}

#endif