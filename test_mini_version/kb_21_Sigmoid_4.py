import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------
// Fast sigmoid for FP32 (uses CUDA fast-math intrinsic __expf)
// ---------------------------------------------------------------------
__device__ __forceinline__ float sigmoidf(float v)
{
    return 1.f / (1.f + __expf(-v));
}

// ---------------------------------------------------------------------
// Scalar in-place kernel (generic, used for FP64 fallback)
// ---------------------------------------------------------------------
template<typename scalar_t>
__global__ __launch_bounds__(256, 2)
void sigmoid_kernel_scalar_ip(scalar_t* __restrict__ data,
                              const int64_t          numel)
{
    const int idx    = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride)
    {
        scalar_t v = data[i];
        data[i] = scalar_t(1.0) / (scalar_t(1.0) + exp(-v));
    }
}

// ---------------------------------------------------------------------
// Vectorised FP32 kernel – processes 4 elements at a time, in-place
// ---------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void sigmoid_kernel_vec4_ip(float*       __restrict__ data,
                            const int64_t            numel)
{
    const int idx    = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Number of float4 elements
    const int64_t vec_elems = numel >> 2;  // numel / 4

    // Reinterpret data pointer as float4*
    float4* __restrict__ d4 = reinterpret_cast<float4*>(data);

    // ---- main vectorised loop ----
    for (int64_t i = idx; i < vec_elems; i += stride)
    {
        float4 v = d4[i];   // 128-bit load
        v.x = sigmoidf(v.x);
        v.y = sigmoidf(v.y);
        v.z = sigmoidf(v.z);
        v.w = sigmoidf(v.w);
        d4[i] = v;          // 128-bit store (same location)
    }

    // ---- scalar tail (handles numel % 4) ----
    for (int64_t i = (vec_elems << 2) + idx; i < numel; i += stride)
    {
        data[i] = sigmoidf(data[i]);
    }
}

// ---------------------------------------------------------------------
// Host wrapper – launches in-place kernels
// ---------------------------------------------------------------------
torch::Tensor sigmoid_cuda_inplace(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(),  "Input must reside on CUDA device");
    TORCH_CHECK(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat64,
                "Supported dtypes are float32 and float64");

    // Ensure contiguous layout; otherwise make a contiguous copy
    auto x_contig = x.contiguous();
    const int64_t numel = x_contig.numel();

    if (x_contig.scalar_type() == torch::kFloat32)
    {
        const int threads = 256;
        // Each thread handles 4 elements via float4
        const int blocks  = (((numel + 3) >> 2) + threads - 1) / threads;

        sigmoid_kernel_vec4_ip<<<blocks, threads>>>(
            x_contig.data_ptr<float>(),
            numel);
    }
    else    // FP64 path
    {
        const int threads = 256;
        const int blocks  = (numel + threads - 1) / threads;

        sigmoid_kernel_scalar_ip<double><<<blocks, threads>>>(
            x_contig.data_ptr<double>(),
            numel);
    }

    // Check for launch / runtime errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("sigmoid_cuda_inplace failed: ")
                                 + cudaGetErrorString(err));

    return x_contig;
}
'''

cpp_src = r'''
torch::Tensor sigmoid_cuda_inplace(torch::Tensor x);
'''

sigmoid_mod = load_inline(
    name         = 'sigmoid_mod_ip_1765982826680',
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ['sigmoid_cuda_inplace'],
    with_cuda    = True,
    verbose      = True,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--ptxas-options=-v'],
)

class ModelNew(nn.Module):
    """
    Drop-in replacement using an in-place, vectorised CUDA sigmoid.
    """
    def __init__(self):
        super().__init__()
        self.sigmoid_cuda = sigmoid_mod.sigmoid_cuda_inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_cuda(x)