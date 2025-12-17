import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -------------------------------------------------------------
// Helper – fast sigmoid for FP32 (uses fast-math intrinsic expf)
// -------------------------------------------------------------
__device__ __forceinline__ float sigmoidf(float v)
{
    return 1.f / (1.f + __expf(-v));
}

// -------------------------------------------------------------
// Generic scalar kernel (kept for FP64 or fallback)
// -------------------------------------------------------------
template<typename scalar_t>
__global__ void sigmoid_kernel_scalar(const scalar_t* __restrict__ x,
                                      scalar_t*       __restrict__ y,
                                      const int64_t   numel)
{
    const int idx    = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride)
    {
        scalar_t v = x[i];
        y[i] = scalar_t(1.0) / (scalar_t(1.0) + exp(-v));
    }
}

// -------------------------------------------------------------
// Vectorised kernel – processes 4 FP32 elements at a time
// -------------------------------------------------------------
__global__ void sigmoid_kernel_vec4(const float* __restrict__ x,
                                    float*       __restrict__ y,
                                    const int64_t numel)
{
    const int idx    = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // number of full float4 elements
    const int64_t vec_elems = numel >> 2;       // numel / 4

    // reinterpret pointers
    const float4* __restrict__ x4 =
        reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 =
        reinterpret_cast<float4*>(y);

    // ---- main vectorised loop ----
    for (int64_t i = idx; i < vec_elems; i += stride)
    {
        float4 v = x4[i];        // 128-bit load
        v.x = sigmoidf(v.x);
        v.y = sigmoidf(v.y);
        v.z = sigmoidf(v.z);
        v.w = sigmoidf(v.w);
        y4[i] = v;               // 128-bit store
    }

    // ---- scalar tail (handles numel % 4) ----
    for (int64_t i = (vec_elems << 2) + idx; i < numel; i += stride)
    {
        y[i] = sigmoidf(x[i]);
    }
}

// -------------------------------------------------------------
// Host wrapper
// -------------------------------------------------------------
torch::Tensor sigmoid_cuda(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(),  "Input must reside on CUDA device");
    TORCH_CHECK(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat64,
                "Supported dtypes are float32 and float64");

    auto x_contig = x.contiguous();
    auto y        = torch::empty_like(x_contig);
    const int64_t numel = x_contig.numel();

    // ---------------- dispatch on dtype ----------------
    if (x_contig.scalar_type() == torch::kFloat32)
    {
        const int threads = 256;
        // each thread handles 4 elements -> divide by 4 when computing blocks
        const int blocks  = ( ( (numel + 3) >> 2 ) + threads - 1) / threads;

        sigmoid_kernel_vec4<<<blocks, threads>>>(
            x_contig.data_ptr<float>(),
            y.data_ptr<float>(),
            numel);
    }
    else    // FP64
    {
        const int threads = 256;
        const int blocks  = (numel + threads - 1) / threads;

        sigmoid_kernel_scalar<double>
            <<<blocks, threads>>>(
                x_contig.data_ptr<double>(),
                y.data_ptr<double>(),
                numel);
    }

    // check for kernel launch / execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("sigmoid_cuda failed: ")
                                 + cudaGetErrorString(err));

    return y;
}
'''

cpp_src = r'''
torch::Tensor sigmoid_cuda(torch::Tensor x);
'''

sigmoid_mod = load_inline(
    name         = 'sigmoid_mod',
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ['sigmoid_cuda'],
    with_cuda    = True,
    verbose      = True,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--ptxas-options=-v'],
)

class ModelNew(nn.Module):
    """
    Drop-in replacement of original Model using an optimized CUDA sigmoid.
    """
    def __init__(self):
        super().__init__()
        self.sigmoid_cuda = sigmoid_mod.sigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_cuda(x)