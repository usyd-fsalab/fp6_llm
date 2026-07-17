// pybind.cpp — pure C++ (compiled by g++, not hipcc)
// All PyTorch wrapper functions and CPU utility headers are here,
// safely away from hipcc's device compilation context.
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include "fp6_linear.cuh"
// CPU utility headers — only needed for Python-side weight prepacking/dequant.
// These use std::vector which is incompatible with hipcc device compilation.
#include "utils/weight_prepacking.h"
#include "utils/weight_dequant.h"
#include "utils/weight_quant.h"

// CPU utility headers -- only needed for PyTorch wrapper functions

// prepare_tensors removed (unused placeholder)

torch::Tensor fp6_linear_forward_cuda(
    torch::Tensor _in_feats, torch::Tensor _weights, torch::Tensor _scales, int splitK=1)
{
    int N=_weights.size(0), K=_in_feats.size(1), M_=_in_feats.size(0);
    assert(K%64==0); assert((K/16*3)==_weights.size(1));
    auto weight   = reinterpret_cast<const uint4*>(_weights.data_ptr<int>());
    auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales   = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
    auto options  = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out   = torch::empty({M_, N}, options);
    at::Tensor _ws    = torch::empty({splitK, M_, N}, torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device()));
    auto out_feats = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
    auto ws        = reinterpret_cast<float*>(_ws.data_ptr<float>());
    auto stream    = at::hip::getCurrentHIPStream(_in_feats.device().index()).stream();
    fp6_linear_kernel(stream, weight, scales, in_feats, out_feats, N, M_, K, ws, splitK);
    return _out;
}

torch::Tensor weight_matrix_prepacking_cpu(torch::Tensor fp6_tensor)
{
    size_t OC=fp6_tensor.size(0), IC=fp6_tensor.size(1);
    assert(IC%3==0); IC=IC*16/3;
    assert((OC%256==0)&&(IC%64==0));
    auto packed=torch::empty_like(fp6_tensor);
    weight_matrix_prepacking(reinterpret_cast<int*>(packed.data_ptr<int>()),
                             reinterpret_cast<int*>(fp6_tensor.data_ptr<int>()), OC, IC);
    return packed;
}

torch::Tensor weight_matrix_dequant_cpu(torch::Tensor fp6_tensor, torch::Tensor fp16_scale)
{
    int OC=fp6_tensor.size(0); assert(fp6_tensor.size(1)%3==0);
    int IC=fp6_tensor.size(1)/3*16; assert(fp16_scale.size(0)==OC);
    auto options=torch::TensorOptions().dtype(fp16_scale.dtype()).device(fp16_scale.device());
    at::Tensor fp16=torch::empty({OC,IC},options);
    DeQuantMatrix_FP6_To_FP16(reinterpret_cast<half*>(fp16.data_ptr<at::Half>()),
        (unsigned char*)fp6_tensor.data_ptr<int>(), OC, IC,
        reinterpret_cast<half*>(fp16_scale.data_ptr<at::Half>()));
    return fp16;
}

torch::Tensor fp_eXmY_linear_forward_cuda(
    int EXPONENT, int MANTISSA,
    torch::Tensor _in_feats, torch::Tensor _weights, torch::Tensor _scales, int splitK=1)
{
    int N=_weights.size(0), K=_in_feats.size(1), M_=_in_feats.size(0);
    assert(K%64==0); assert((K/32*(1+EXPONENT+MANTISSA))==_weights.size(1));
    auto weight   = reinterpret_cast<const uint4*>(_weights.data_ptr<int>());
    auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales   = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
    auto options  = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out = torch::empty({M_, N}, options);
    at::Tensor _ws  = torch::empty({splitK, M_, N}, torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device()));
    auto stream     = at::hip::getCurrentHIPStream(_in_feats.device().index()).stream();
    fp_eXmY_linear_kernel(EXPONENT, MANTISSA, stream,
        weight, scales, in_feats, reinterpret_cast<half*>(_out.data_ptr<at::Half>()),
        N, M_, K, reinterpret_cast<float*>(_ws.data_ptr<float>()), splitK);
    return _out;
}

// OPT3: FP8 forward (new PyTorch op)
// fp8_linear_forward_cuda: deferred


torch::Tensor weight_matrix_prepacking_fp_eXmY_cpu(int EXPONENT, int MANTISSA, torch::Tensor fpx_tensor)
{
    int BIT_WIDTH=1+EXPONENT+MANTISSA;
    size_t OC=fpx_tensor.size(0), IC=fpx_tensor.size(1);
    assert(IC%BIT_WIDTH==0); IC=IC*32/BIT_WIDTH;
    assert((OC%256==0)&&(IC%64==0));
    auto packed=torch::empty_like(fpx_tensor);
    weight_matrix_prepacking_fp_eXmY(EXPONENT, MANTISSA,
        reinterpret_cast<int*>(packed.data_ptr<int>()),
        reinterpret_cast<int*>(fpx_tensor.data_ptr<int>()), OC, IC);
    return packed;
}

torch::Tensor weight_matrix_dequant_fp_eXmY_cpu(int EXPONENT, int MANTISSA,
    torch::Tensor fpx_tensor, torch::Tensor fp16_scale)
{
    int BIT_WIDTH=1+EXPONENT+MANTISSA, OC=fpx_tensor.size(0);
    assert(fpx_tensor.size(1)%BIT_WIDTH==0);
    int IC=fpx_tensor.size(1)/BIT_WIDTH*32; assert(fp16_scale.size(0)==OC);
    auto options=torch::TensorOptions().dtype(fp16_scale.dtype()).device(fp16_scale.device());
    at::Tensor fp16=torch::empty({OC,IC},options);
    dequant_matrix_fp_eXmY_to_fp16(EXPONENT, MANTISSA,
        reinterpret_cast<half*>(fp16.data_ptr<at::Half>()),
        (unsigned char*)fpx_tensor.data_ptr<int>(), OC, IC,
        reinterpret_cast<half*>(fp16_scale.data_ptr<at::Half>()));
    return fp16;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("linear_forward_cuda",        &fp6_linear_forward_cuda,        "FP6-FP16 GEMM (adaptive splitK).");
    m.def("weight_prepacking_cpu",      &weight_matrix_prepacking_cpu,   "FP6 weight prepacking.");
    m.def("weight_dequant_cpu",         &weight_matrix_dequant_cpu,      "Dequantize FP6->FP16.");
    m.def("linear_forward_eXmY_cuda",   &fp_eXmY_linear_forward_cuda,   "FPx-FP16 GEMM.");
    m.def("weight_prepacking_eXmY_cpu", &weight_matrix_prepacking_fp_eXmY_cpu, "FPx weight prepacking.");
    m.def("weight_dequant_eXmY_cpu",    &weight_matrix_dequant_fp_eXmY_cpu,    "Dequantize FPx->FP16.");
}
