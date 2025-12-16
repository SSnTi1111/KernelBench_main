import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_functions.h>   // for __expf

////////////////////////////////////////////////////////////////////////////////
// CUDA helpers
////////////////////////////////////////////////////////////////////////////////
__device__ inline float  fast_exp(float  x) { return __expf(x); }  // intrinsics
__device__ inline double fast_exp(double x) { return exp(x);   }  // fallback

////////////////////////////////////////////////////////////////////////////////
// CUDA kernel : element-wise Sigmoid (grid-stride)
////////////////////////////////////////////////////////////////////////////////
template<typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ inp,
                               scalar_t* __restrict__ out,
                               const int64_t numel) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += gridDim.x * blockDim.x) {

        scalar_t x      = inp[idx];
        scalar_t denom  = static_cast<scalar_t>(1.0) + fast_exp(-x);
        out[idx]        = static_cast<scalar_t>(1.0) / denom;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host wrapper (FP32 only)
////////////////////////////////////////////////////////////////////////////////
torch::Tensor sigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),      "Input must reside on CUDA device.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "Only float32 tensors are supported.");

    auto y       = torch::empty_like(x);
    const int64_t numel = x.numel();

    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    // launch kernel (compile only float path)
    using scalar_t = float;
    sigmoid_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                  y.data_ptr<scalar_t>(),
                                                  numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed with error: ") +
                                 cudaGetErrorString(err));
    }
    return y;
}
'''

cpp_src = r'''
torch::Tensor sigmoid_forward(torch::Tensor x);
'''

sigmoid_cuda = load_inline(
    name='sigmoid_cuda_fast',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['sigmoid_forward'],
    with_cuda=True,
    extra_cuda_cflags=['-O2', '--ptxas-options=-v', '-use_fast_math'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = sigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.sigmoid_forward(x)