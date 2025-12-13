import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid_func(scalar_t x) {
    return scalar_t(1) / (scalar_t(1) + exp(-x));
}

// Kernel: element-wise Sigmoid
template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t numel) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = input[idx];
        output[idx] = sigmoid_func(val);
    }
}

torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty_like(input);

    const int64_t numel = input.numel();
    const int threads = 256;
    const int64_t blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_forward_cuda", ([&] {
        sigmoid_kernel<scalar_t><<<blocks, threads, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed with error code ", err);
    return output;
}
'''

cpp_src = r'''
torch::Tensor sigmoid_forward(torch::Tensor input);
'''

sigmoid_module = load_inline(
    name='sigmoid_cuda_1765629704110',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['sigmoid_forward'],
    with_cuda=True,
 verbose=True,
    extra_cuda_cflags=['-O2','--ptxas-options=-v']
)


class ModelNew(nn.Module):
    """
    CUDA-accelerated model that applies element-wise Sigmoid.
    Mirrors the original Model interface.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.sigmoid_forward(x)