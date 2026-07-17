#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

hipError_t fp6_linear_kernel(hipStream_t, const uint4*, const half*, const half*, half*,
    const size_t, const size_t, const size_t, float*, int);
hipError_t fp_eXmY_linear_kernel(const int, const int, hipStream_t,
    const uint4*, const half*, const half*, half*,
    const size_t, const size_t, const size_t, float*, int);
hipError_t fp8_linear_kernel(hipStream_t, const uint4*, const half*, const half*, half*,
    const size_t, const size_t, const size_t, float*, int);

#ifndef NO_PYTORCH
#include <torch/extension.h>
torch::Tensor fp6_linear_forward_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int);
torch::Tensor weight_matrix_prepacking_cpu(torch::Tensor);
torch::Tensor weight_matrix_dequant_cpu(torch::Tensor, torch::Tensor);
torch::Tensor fp_eXmY_linear_forward_cuda(int, int, torch::Tensor, torch::Tensor, torch::Tensor, int);
torch::Tensor weight_matrix_prepacking_fp_eXmY_cpu(int, int, torch::Tensor);
torch::Tensor weight_matrix_dequant_fp_eXmY_cpu(int, int, torch::Tensor, torch::Tensor);
torch::Tensor fp8_linear_forward_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int);
#endif
