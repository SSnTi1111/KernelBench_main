import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------- CUDA / C++ Sources --------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x)        TORCH_CHECK((x).is_cuda(),        #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK((x).is_contiguous(),  #x " must be contiguous")
#define CHECK_FLOAT(x)       TORCH_CHECK((x).dtype() == torch::kFloat, #x " must be float32")
#define CHECK_INPUT(x)       CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

/* ---------------- Kernel configuration ---------------- */
constexpr int BLK_H = 16;
constexpr int BLK_W = 16;

/* ---------------- CUDA kernel ---------------- */
__global__ void conv_transpose2d_kernel(
        const float* __restrict__ in,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ out,
        int B, int in_c, int out_c,
        int H_in, int W_in,
        int H_out, int W_out,
        int kH, int kW,
        int stride, int padding,
        int output_padding,
        int groups)
{
    int n  = blockIdx.x;
    int oh = blockIdx.y * BLK_H + threadIdx.y;
    int ow = blockIdx.z * BLK_W + threadIdx.x;

    if (oh >= H_out || ow >= W_out) return;

    for (int oc = 0; oc < out_c; ++oc)
    {
        float acc = (bias != nullptr) ? bias[oc] : 0.0f;

        int out_per_grp = out_c / groups;
        int in_per_grp  = in_c  / groups;
        int g           = oc / out_per_grp;
        int oc_within   = oc - g * out_per_grp;

        int ic_start = g * in_per_grp;
        int ic_end   = ic_start + in_per_grp;

        for (int ic = ic_start; ic < ic_end; ++ic)
        {
            for (int kh = 0; kh < kH; ++kh)
            {
                int h_in_nom = oh + padding - kh;
                if (h_in_nom < 0)            continue;
                if (h_in_nom % stride != 0)  continue;
                int ih = h_in_nom / stride;
                if (ih >= H_in)              continue;

                for (int kw = 0; kw < kW; ++kw)
                {
                    int w_in_nom = ow + padding - kw;
                    if (w_in_nom < 0)           continue;
                    if (w_in_nom % stride != 0) continue;
                    int iw = w_in_nom / stride;
                    if (iw >= W_in)             continue;

                    size_t in_idx =
                        (((size_t)n * in_c + ic) * H_in + ih) * W_in + iw;

                    size_t w_idx =
                        ((((size_t)ic - ic_start) * out_per_grp + oc_within) * kH + kh) * kW + kw;

                    acc += in[in_idx] * weight[w_idx];
                }
            }
        }

        size_t out_idx =
            (((size_t)n * out_c + oc) * H_out + oh) * W_out + ow;
        out[out_idx] = acc;
    }
}

/* ---------------- C++ front-end ---------------- */
torch::Tensor conv_transpose2d_forward(
        torch::Tensor input,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias_opt,
        int64_t stride,
        int64_t padding,
        int64_t output_padding,
        int64_t groups)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    /* Safely handle optional bias (avoid bad optional access) */
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias_tensor = bias_opt.value();
        CHECK_INPUT(bias_tensor);
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    TORCH_CHECK(input.scalar_type()  == torch::kFloat, "Only float32 is supported");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat, "Only float32 is supported");

    int64_t B    = input.size(0);
    int64_t in_c = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int kH = weight.size(2);
    int kW = weight.size(3);
    int out_c_per_grp = weight.size(1);
    int out_c         = out_c_per_grp * groups;

    TORCH_CHECK(in_c % groups == 0, "in_channels must be divisible by groups");

    int H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;

    auto output = torch::empty({B, out_c, H_out, W_out}, input.options());

    dim3 block(BLK_W, BLK_H);
    dim3 grid(B,
              (H_out + BLK_H - 1) / BLK_H,
              (W_out + BLK_W - 1) / BLK_W);

    conv_transpose2d_kernel<<<grid, block, 0>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        B, in_c, out_c,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        static_cast<int>(groups));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "conv_transpose2d_kernel launch failed with error: ",
                cudaGetErrorString(err));

    return output;
}
"""

cpp_src = r"""
torch::Tensor conv_transpose2d_forward(
        torch::Tensor input,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias_opt,
        int64_t stride,
        int64_t padding,
        int64_t output_padding,
        int64_t groups);
"""

# --------------------------- Build extension ---------------------------
conv_transpose2d_mod = load_inline(
    name           = "conv_transpose2d_cuda",
    cpp_sources    = cpp_src,
    cuda_sources   = source,
    functions      = ["conv_transpose2d_forward"],
    with_cuda      = True,
    verbose        = True,
    extra_cuda_cflags=["-O3", "--ptxas-options=-v"]
)

# --------------------------- Python wrapper ----------------------------
class ModelNew(nn.Module):
    """
    Drop-in replacement for nn.ConvTranspose2d backed by the custom CUDA kernel.
    """
    def __init__(self,
                 in_channels:     int,
                 out_channels:    int,
                 kernel_size:     int,
                 stride:          int = 1,
                 padding:         int = 0,
                 output_padding:  int = 0,
                 groups:          int = 1,
                 bias:            bool = False):
        super().__init__()

        # ensure square kernel without using assert
        if isinstance(kernel_size, (tuple, list)):
            if kernel_size[0] != kernel_size[1]:
                raise RuntimeError("Kernel must be square")
            kernel_size = kernel_size[0]

        weight_shape = (in_channels,
                        out_channels // groups,
                        kernel_size,
                        kernel_size)
        self.weight = nn.Parameter(
            torch.randn(*weight_shape) *
            (1.0 / (in_channels * kernel_size * kernel_size) ** 0.5)
        )

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.stride          = stride
        self.padding         = padding
        self.output_padding  = output_padding
        self.groups          = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_mod.conv_transpose2d_forward(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )