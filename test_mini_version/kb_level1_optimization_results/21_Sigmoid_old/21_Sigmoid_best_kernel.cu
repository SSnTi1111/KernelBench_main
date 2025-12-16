import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define THREADS 256          // threads / block (fixed)
#define VEC_F32 4            // elements / thread (float4)
#define VEC_F16 8            // elements / thread (Half8)
#define TILE_F32 (THREADS * VEC_F32)
#define TILE_F16 (THREADS * VEC_F16)

// ---------------------------------------------------------------------
// Fast sigmoid helpers
// ---------------------------------------------------------------------
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// half-precision:  x / (1 + |x|) * 0.5 + 0.5  (no exp, all fp16 math)
__device__ __forceinline__ __half sigmoid_half(__half h) {
    const __half one  = __float2half_rn(1.0f);
    const __half half = __float2half_rn(0.5f);
    return __hadd(__hmul(__hdiv(h, __hadd(one, __habs(h))), half), half);
}

__device__ __forceinline__ __half2 sigmoid_half2(__half2 h2) {
    const __half2 one2  = __half2half2(__float2half(1.0f));
    const __half2 half2 = __half2half2(__float2half(0.5f));
    return __hadd2(__hmul2(__h2div(h2, __hadd2(one2, __habs2(h2))), half2), half2);
}

// ---------------------------------------------------------------------
// Vector struct for 8 half elements (16 B, same as float4)
// ---------------------------------------------------------------------
struct __align__(16) Half8 {
    __half2 v0, v1, v2, v3;
};

// ---------------------------------------------------------------------
// Existing vector kernels (kept for reference / fallback)
// ---------------------------------------------------------------------
__global__ void sigmoid_kernel_vec4_f32(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        const int64_t numel) {
    const int64_t vec_idx  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base_idx = vec_idx * 4;

    if (base_idx + 3 < numel) {
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        float4* __restrict__ y4       = reinterpret_cast<float4*>(y);

        float4 v = x4[vec_idx];
        v.x = sigmoidf(v.x);
        v.y = sigmoidf(v.y);
        v.z = sigmoidf(v.z);
        v.w = sigmoidf(v.w);
        y4[vec_idx] = v;
    } else {
        for (int k = 0; k < 4; ++k) {
            int64_t idx = base_idx + k;
            if (idx < numel) {
                y[idx] = sigmoidf(x[idx]);
            }
        }
    }
}

__global__ void sigmoid_kernel_vec8_f16(const __half* __restrict__ x,
                                        __half* __restrict__ y,
                                        const int64_t numel) {
    const int64_t vec_idx  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base_idx = vec_idx * 8;

    if (base_idx + 7 < numel) {
        const Half8* __restrict__ x8 = reinterpret_cast<const Half8*>(x);
        Half8* __restrict__ y8       = reinterpret_cast<Half8*>(y);

        Half8 v = x8[vec_idx];

        __half2 lanes[4] = {v.v0, v.v1, v.v2, v.v3};

        #pragma unroll
        for (int i = 0; i < 4; ++i)
            lanes[i] = sigmoid_half2(lanes[i]);

        v.v0 = lanes[0]; v.v1 = lanes[1];
        v.v2 = lanes[2]; v.v3 = lanes[3];

        y8[vec_idx] = v;
    } else {
        for (int k = 0; k < 8; ++k) {
            int64_t idx = base_idx + k;
            if (idx < numel) {
                y[idx] = sigmoid_half(x[idx]);
            }
        }
    }
}

// ---------------------------------------------------------------------
// NEW tiled shared-memory kernels
// ---------------------------------------------------------------------
__global__ void sigmoid_kernel_tile_f32(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        const int64_t numel) {
    __shared__ float4 s_vec[THREADS];          // 4 kB smem (256 × 16 B)

    const int tid = threadIdx.x;
    const int64_t thread_base_elem = tid * VEC_F32;

    for (int64_t tile_base = static_cast<int64_t>(blockIdx.x) * TILE_F32;
         tile_base < numel;
         tile_base += static_cast<int64_t>(gridDim.x) * TILE_F32) {

        // global vector index for this thread
        const int64_t gvec = (tile_base / VEC_F32) + tid;
        const bool full = tile_base + thread_base_elem + (VEC_F32 - 1) < numel;

        // ------------------ load ------------------
        if (full) {
            s_vec[tid] = reinterpret_cast<const float4*>(x)[gvec];
        } else {
            float tmp[VEC_F32];
            #pragma unroll
            for (int k = 0; k < VEC_F32; ++k) {
                int64_t idx = tile_base + thread_base_elem + k;
                tmp[k] = (idx < numel) ? x[idx] : 0.0f;
            }
            s_vec[tid] = *reinterpret_cast<float4*>(tmp);
        }
        __syncthreads();

        // ---------------- compute -----------------
        float4 v = s_vec[tid];
        v.x = sigmoidf(v.x);
        v.y = sigmoidf(v.y);
        v.z = sigmoidf(v.z);
        v.w = sigmoidf(v.w);
        s_vec[tid] = v;
        __syncthreads();

        // ---------------- store -------------------
        if (full) {
            reinterpret_cast<float4*>(y)[gvec] = v;
        } else {
            float *t = reinterpret_cast<float*>(&v);
            #pragma unroll
            for (int k = 0; k < VEC_F32; ++k) {
                int64_t idx = tile_base + thread_base_elem + k;
                if (idx < numel) {
                    y[idx] = t[k];
                }
            }
        }
    }
}

__global__ void sigmoid_kernel_tile_f16(const __half* __restrict__ x,
                                        __half* __restrict__ y,
                                        const int64_t numel) {
    __shared__ Half8 s_vec[THREADS];           // 4 kB smem

    const int tid = threadIdx.x;
    const int64_t thread_base_elem = tid * VEC_F16;

    for (int64_t tile_base = static_cast<int64_t>(blockIdx.x) * TILE_F16;
         tile_base < numel;
         tile_base += static_cast<int64_t>(gridDim.x) * TILE_F16) {

        const int64_t gvec = (tile_base / VEC_F16) + tid;
        const bool full = tile_base + thread_base_elem + (VEC_F16 - 1) < numel;

        // ------------------ load ------------------
        if (full) {
            s_vec[tid] = reinterpret_cast<const Half8*>(x)[gvec];
        } else {
            Half8 hv;
            __half* hp = reinterpret_cast<__half*>(&hv);
            #pragma unroll
            for (int k = 0; k < VEC_F16; ++k) {
                int64_t idx = tile_base + thread_base_elem + k;
                hp[k] = (idx < numel) ? x[idx] : __float2half(0.0f);
            }
            s_vec[tid] = hv;
        }
        __syncthreads();

        // ---------------- compute -----------------
        Half8 v = s_vec[tid];
        __half2 lanes[4] = {v.v0, v.v1, v.v2, v.v3};

        #pragma unroll
        for (int i = 0; i < 4; ++i)
            lanes[i] = sigmoid_half2(lanes[i]);

        v.v0 = lanes[0]; v.v1 = lanes[1];
        v.v2 = lanes[2]; v.v3 = lanes[3];
        s_vec[tid] = v;
        __syncthreads();

        // ---------------- store -------------------
        if (full) {
            reinterpret_cast<Half8*>(y)[gvec] = v;
        } else {
            __half* hp = reinterpret_cast<__half*>(&v);
            #pragma unroll
            for (int k = 0; k < VEC_F16; ++k) {
                int64_t idx = tile_base + thread_base_elem + k;
                if (idx < numel) {
                    y[idx] = hp[k];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------
torch::Tensor sigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
        "Only float32 and float16 tensors are supported");

    auto y = torch::empty_like(x);
    const int64_t numel = x.numel();
    cudaError_t err;

    if (x.scalar_type() == at::kFloat) {
        const int grids = static_cast<int>((numel + TILE_F32 - 1) / TILE_F32);
        sigmoid_kernel_tile_f32<<<grids, THREADS>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            numel);
        err = cudaGetLastError();
    } else {  // at::kHalf
        const int grids = static_cast<int>((numel + TILE_F16 - 1) / TILE_F16);
        sigmoid_kernel_tile_f16<<<grids, THREADS>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            numel);
        err = cudaGetLastError();
    }

    TORCH_CHECK(err == cudaSuccess, "sigmoid kernel launch failed: ",
                cudaGetErrorString(err));
    return y;
}
'''

cpp_src = r'''
torch::Tensor sigmoid_forward(torch::Tensor x);
'''

sigmoid_cuda = load_inline(
    name='sigmoid_cuda_opt_tiled',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['sigmoid_forward'],
    with_cuda=True,
    verbose=True,
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-Xptxas', '-dlcm=ca',
        '--ptxas-options=-v',
        '-arch=sm_70'
    ],
)

class ModelNew(nn.Module):
    """
    Model wrapper calling the optimised CUDA sigmoid.
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = sigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.sigmoid_forward(x)