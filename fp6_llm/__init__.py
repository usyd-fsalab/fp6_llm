from fp6_llm_cuda import linear_forward_cuda, weight_prepacking_cpu, weight_dequant_cpu, linear_forward_eXmY_cuda, weight_prepacking_eXmY_cpu, weight_dequant_eXmY_cpu



import math
def Num_Wave(M, N, SplitK, Num_GPU_SMs):
    Num_Wave = math.ceil(M/512) * math.ceil(N/64) * SplitK / Num_GPU_SMs
    return Num_Wave

# The shape of the MatMul: (M, K)*(K, N)->(M, N).
# Typically, M is the number of rows of weight matrix, and N is the inference batch size.
# Num_GPU_SMs is the number of SMs (Streaming Multiprocessors) within the GPUs, e,g. each A100 GPU has 108 SMs.
def HeuristicFuntion_SplitK(M, N, Num_GPU_SMs):
    SplitK=1
    Efficiency=0.0
    for i in range(1, 100):
        numWave = Num_Wave(M, N, i, Num_GPU_SMs)
        eff = numWave / math.ceil(numWave)
        if eff >= 0.8:
            SplitK = i
            Efficiency = eff
            break
    return SplitK
    