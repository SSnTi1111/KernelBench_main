import torch
import torch.nn as nn
import math                         # Added to use math.sqrt
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>      // required if math functions are needed

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/***********************************************************************
* Kernel: one thread computes exactly one output element
***********************************************************************/
__global__ void conv_transpose2d_kernel(
        const float *__restrict__ input,          // (N, Cin, Hin, Win)
        const float *__restrict__ weight,         // (Cin, Cout, kH, kW)
        const float *__restrict__ bias,           // (Cout) or nullptr
        float *__restrict__ output,               // (N, Cout, Hout, Wout)
        const int N,
        const int Cin,
        const int Hin,
        const int Win,
        const int Cout,
        const int kH,
        const int kW,
        const int stride,
        const int padding,
        const int output_padding,
        const int Hout,
        const int Wout,
        const bool has_bias)
{
    const int total_elems = N * Cout * Hout * Wout;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total_elems) return;

    /* unravel linear index -> (n, cout, h_out, w_out) */
    int tmp = global_idx;
    const int w_out = tmp % Wout;        tmp /= Wout;
    const int h_out = tmp % Hout;        tmp /= Hout;
    const int c_out = tmp % Cout;        tmp /= Cout;
    const int n      = tmp;

    float val = 0.f;

    for (int c_in = 0; c_in < Cin; ++c_in) {
        for (int kh = 0; kh < kH; ++kh) {
            int in_h_nom = h_out + padding - kh;
            if (in_h_nom % stride != 0) continue;
            int in_h = in_h_nom / stride;
            if (in_h < 0 || in_h >= Hin) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int in_w_nom = w_out + padding - kw;
                if (in_w_nom % stride != 0) continue;
                int in_w = in_w_nom / stride;
                if (in_w < 0 || in_w >= Win) continue;

                const int inp_idx = ((n * Cin + c_in) * Hin + in_h) * Win + in_w;
                const int w_idx   = ((c_in * Cout + c_out) * kH + kh) * kW + kw;
                val += input[inp_idx] * weight[w_idx];
            }
        }
    }

    if (has_bias) val += bias[c_out];

    const int out_idx = ((n * Cout + c_out) * Hout + h_out) * Wout + w_out;
    output[out_idx] = val;
}

/***********************************************************************
* Host wrapper
***********************************************************************/
torch::Tensor conv_transpose2d_cuda(
        torch::Tensor input,          // (N, Cin, Hin, Win)
        torch::Tensor weight,         // (Cin, Cout, kH, kW)
        torch::Tensor bias,           // () or (Cout)
        int64_t stride,
        int64_t padding,
        int64_t output_padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);

    TORCH_CHECK(input.dim()  == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be (Cin, Cout, kH, kW)");
    TORCH_CHECK(weight.size(2) == weight.size(3),
                "kernel must be square in this implementation");

    const int N     = input.size(0);
    const int Cin   = input.size(1);
    const int Hin   = input.size(2);
    const int Win   = input.size(3);

    const int kH = weight.size(2);
    const int kW = weight.size(3);

    TORCH_CHECK(kH == kW, "Only square kernels are supported");

    const int Cout = weight.size(1);
    const bool has_bias = bias.defined() && bias.numel() > 0;

    /* output sizes (PyTorch formula) */
    const int Hout = (Hin - 1) * stride - 2 * padding + kH + output_padding;
    const int Wout = (Win - 1) * stride - 2 * padding + kW + output_padding;

    auto output = torch::empty({N, Cout, Hout, Wout}, input.options());

    const int threads = 256;
    const int total_elems = N * Cout * Hout * Wout;
    const int blocks = ceil_div(total_elems, threads);

    conv_transpose2d_kernel<<<blocks, threads, 0>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, Cin, Hin, Win, Cout,
        kH, kW,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        Hout, Wout,
        has_bias
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv_transpose2d_kernel failed with error code ", cudaGetErrorString(err));
    return output;
}
'''
cpp_src = r'''
torch::Tensor conv_transpose2d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t stride,
        int64_t padding,
        int64_t output_padding);
'''

conv_transpose_2d_op = load_inline(
    name='conv_transpose2d_op',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['conv_transpose2d_cuda'],
    with_cuda=True,
    verbose=True,                                # make compilation verbose
    extra_cuda_cflags=['-O3', '--ptxas-options=-v'],
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False
    ) -> None:
        super().__init__()
        # Removed assert statements as per requirement
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        weight_shape = (in_channels, out_channels // groups, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # use same initialization strategy as PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose_2d_op.conv_transpose2d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.contiguous() if self.bias is not None else torch.Tensor().to(x.device),
            self.stride,
            self.padding,
            self.output_padding
        )