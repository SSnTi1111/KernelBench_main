import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------------------
# CUDA source
# ------------------------------------------------------------------------------
source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////
// Sigmoid kernel
/////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void sigmoid_forward_kernel(const scalar_t* __restrict__ x,
                                       scalar_t* __restrict__ y,
                                       const int64_t numel)
{
    // Grid-stride loop so the kernel works for any tensor size
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += blockDim.x * gridDim.x)
    {
        if (idx < numel) {           // Boundary guard (redundant but safe)
            const scalar_t val = x[idx];
            y[idx] = static_cast<scalar_t>(1) /
                     (static_cast<scalar_t>(1) + exp(-val));
        }
    }
}

/////////////////////////////////////////////////////////////////
// Host wrapper
/////////////////////////////////////////////////////////////////
torch::Tensor sigmoid_cuda(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32,
                "Only float32 tensors are supported");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);

    const int64_t numel = input.numel();
    if (numel == 0) return output;

    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    // Launch kernel
    sigmoid_forward_kernel<float><<<blocks, threads, 0>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sigmoid_cuda kernel launch failed : ",
                cudaGetErrorString(err));

    return output;
}
'''

# ------------------------------------------------------------------------------
# C++ prototypes exposed to Python
# ------------------------------------------------------------------------------
cpp_src = r'''
torch::Tensor sigmoid_cuda(torch::Tensor input);
'''

# ------------------------------------------------------------------------------
# Compile & load
# ------------------------------------------------------------------------------
sigmoid_extension = load_inline(
    name         = 'sigmoid_extension',
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ['sigmoid_cuda'],
    with_cuda    = True,
    verbose      = True,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--ptxas-options=-v'],
)

# ------------------------------------------------------------------------------
# nn.Module that calls the custom CUDA kernel
# ------------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Mirrors the original Model but utilizes a custom CUDA sigmoid kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous for the kernel
        if not x.is_contiguous():
            x = x.contiguous()
        return sigmoid_extension.sigmoid_cuda(x)