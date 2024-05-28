#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "fp6_linear.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // Old Interfaces.
    m.def("linear_forward_cuda", &fp6_linear_forward_cuda, "Computes FP6-FP16 GEMM.");
    m.def("weight_prepacking_cpu", &weight_matrix_prepacking_cpu, "Weight prepacking.");
    m.def("weight_dequant_cpu", &weight_matrix_dequant_cpu, "Dequantize weight from fp6 to fp16.");
    // New Interfaces.
    m.def("linear_forward_eXmY_cuda", &fp_eXmY_linear_forward_cuda, "Computes FPx-FP16 GEMM.");
    m.def("weight_prepacking_eXmY_cpu", &weight_matrix_prepacking_fp_eXmY_cpu, "FPx Weight prepacking.");
    m.def("weight_dequant_eXmY_cpu", &weight_matrix_dequant_fp_eXmY_cpu, "Dequantize weight from fpx to fp16.");    
}