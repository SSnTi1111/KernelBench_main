import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////
// Element-wise Softsign: y = x / (1 + |x|)
////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void softsign_kernel(const scalar_t* __restrict__ x,
                                scalar_t* __restrict__ y,
                                const int64_t N)
{
    // flat thread id
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (; idx < N; idx += stride) {
        scalar_t val = x[idx];
        scalar_t denom = scalar_t(1.0) + abs(val);
        y[idx] = val / denom;
    }
}

torch::Tensor softsign_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 tensors are supported");
    
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    const int64_t N = x_contig.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_contig.scalar_type(), "softsign_kernel_launch", ([&] {
        softsign_kernel<scalar_t><<<blocks, threads>>>(
            x_contig.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            N);
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return y;
}
'''

cpp_src = r'''
torch::Tensor softsign_forward(torch::Tensor x);
'''

softsign_module = load_inline(
    name='softsign_module',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['softsign_forward'],
    with_cuda=True,
    verbose=True,
    extra_cuda_cflags=['-O3', '--ptxas-options=-v']
)

class ModelNew(nn.Module):
    """
    CUDA-accelerated Softsign model.
    Mirrors the original Model interface.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # no parameters to initialize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softsign_module.softsign_forward(x)