from fp6_llm_cuda import (
    # Original FP6 interface
    linear_forward_cuda,
    weight_prepacking_cpu,
    weight_dequant_cpu,
    # General FPx interface
    linear_forward_eXmY_cuda,
    weight_prepacking_eXmY_cpu,
    weight_dequant_eXmY_cpu,
    # OPT3: FP8 E4M3 interface
    linear_forward_fp8_cuda,
)

import math

def Num_Wave(M, N, SplitK, Num_GPU_SMs):
    return math.ceil(M/512) * math.ceil(N/64) * SplitK / Num_GPU_SMs

def HeuristicFuntion_SplitK(M, N, Num_GPU_SMs):
    SplitK, Efficiency = 1, 0.0
    for i in range(1, 100):
        numWave = Num_Wave(M, N, i, Num_GPU_SMs)
        eff = numWave / math.ceil(numWave)
        if eff >= 0.8:
            SplitK, Efficiency = i, eff
            break
    return SplitK
