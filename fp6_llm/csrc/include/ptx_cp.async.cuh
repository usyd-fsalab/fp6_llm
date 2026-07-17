/*
 * AMD ROCm HIP port of ptx_cp.async.cuh — v3 optimized
 *
 * Uses __builtin_prefetch (GCC/clang standard, works on all AMD GPU arches
 * including gfx942) to hint the next cache line into L1/L2 ahead of the
 * synchronous load. This partially restores the latency-hiding effect of
 * CUDA cp.async by overlapping prefetch with MFMA compute.
 *
 * Note: __builtin_amdgcn_s_prefetch_data requires gfx12 (RDNA4) — not
 * available on gfx942 (CDNA3/MI300X). Use __builtin_prefetch instead.
 *
 * KB refs: ptx_asm_to_hip_builtins (production_ready)
 */
#ifndef PTX_CP_ASYNC_CUH
#define PTX_CP_ASYNC_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template<int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert(SizeInBytes == 16, "Size is not supported");
    if (pred_guard) {
        // Prefetch the next 64 bytes (next cacheline) into L1D cache
        // while this iteration's load is being processed.
        // __builtin_prefetch(addr, rw=0(read), locality=3(L1)) is supported on gfx942.
        __builtin_prefetch(global_ptr + 16, 0, 3);

        // 128-bit synchronous global->LDS copy
        const uint4* src = reinterpret_cast<const uint4*>(global_ptr);
        uint4*       dst = reinterpret_cast<uint4*>(smem_ptr);
        *dst = *src;
    }
}

__device__ __forceinline__ void cp_async_group_commit() { /* no-op */ }

template<int N>
__device__ __forceinline__ void cp_async_wait_group() { __syncthreads(); }

__device__ __forceinline__ void cp_async_wait_all() { __syncthreads(); }

#endif  // PTX_CP_ASYNC_CUH
