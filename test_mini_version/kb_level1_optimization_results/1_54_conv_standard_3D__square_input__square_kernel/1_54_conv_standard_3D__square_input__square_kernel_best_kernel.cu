import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA / C++ source
# ----------------------------------------------------------------------
source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

///////////////////////////////////////////////////////////////////
// Kernel : one thread computes one output element (n, oc, od, oh, ow)
///////////////////////////////////////////////////////////////////
__global__ void conv3d_forward_kernel(
        const float *__restrict__ input,      // [N, C, D, H, W]
        const float *__restrict__ weight,     // [OC, Cg, kD, kH, kW]
        const float *__restrict__ bias,       // [OC]  (can be empty)
        float *__restrict__ output,           // [N, OC, Od, Oh, Ow]
        int N, int C, int D, int H, int W,
        int OC, int kD, int kH, int kW,
        int stride_d, int stride_h, int stride_w,
        int pad_d, int pad_h, int pad_w,
        int dil_d, int dil_h, int dil_w,
        int groups,
        int outD, int outH, int outW,
        int bias_flag) {

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * OC * outD * outH * outW;
    if (idx >= total) return;

    // Decompose linear index -> coordinates
    int ow = idx % outW;
    idx /= outW;
    int oh = idx % outH;
    idx /= outH;
    int od = idx % outD;
    idx /= outD;
    int oc = idx % OC;
    int n  = idx / OC;

    int channels_per_group = C / groups;
    int group_id = oc / (OC / groups);
    int ic_start = group_id * channels_per_group;
    int ic_end   = ic_start + channels_per_group;

    // Initialize accumulator with bias (if any)
    float val = bias_flag ? bias[oc] : 0.0f;

    // Iterate over kernel volume
    for (int kd = 0; kd < kD; ++kd) {
        int in_d = od * stride_d - pad_d + kd * dil_d;
        if (in_d < 0 || in_d >= D) continue;

        for (int kh = 0; kh < kH; ++kh) {
            int in_h = oh * stride_h - pad_h + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int in_w = ow * stride_w - pad_w + kw * dil_w;
                if (in_w < 0 || in_w >= W) continue;

                // Pointer offset helpers
                long long input_base  = (((long long)n * C * D + ic_start * D + 0LL) * H + 0LL) * W; // n, ic_start, d=0, h=0, w=0
                long long weight_base = (((long long)oc * channels_per_group) * kD + kd) * kH * kW;  // oc, ic=0 will be added later

                for (int ic = ic_start; ic < ic_end; ++ic) {
                    int w_ic = ic - ic_start;

                    long long inp_idx = input_base
                        + ((long long)ic - ic_start) * D * H * W     // step through channel
                        + (long long)in_d * H * W
                        + (long long)in_h * W
                        + in_w;

                    long long w_idx  = weight_base
                        + ((long long)w_ic) * kD * kH * kW
                        + (long long)kh * kW
                        + kw;

                    val += input[inp_idx] * weight[w_idx];
                }
            }
        }
    }

    // Store result
    long long out_idx = (((long long)n * OC + oc) * outD + od) * outH * outW
                        + (long long)oh * outW + ow;
    output[out_idx] = val;
}

///////////////////////////////////////////////////////////////////
// Host launcher
///////////////////////////////////////////////////////////////////
torch::Tensor conv3d_forward(torch::Tensor input,
                             torch::Tensor weight,
                             torch::Tensor bias,        // can be empty tensor
                             int stride_d, int stride_h, int stride_w,
                             int pad_d, int pad_h, int pad_w,
                             int dil_d, int dil_h, int dil_w,
                             int groups,
                             bool bias_flag) {

    // ----------------- argument checks -----------------
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias_flag) {
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.numel() == weight.size(0), "bias shape mismatch");
    }

    TORCH_CHECK(input.dim()  == 5, "input should be NDHWC with 5 dims (N,C,D,H,W)");
    TORCH_CHECK(weight.dim() == 5, "weight should be (OC,C_per_group,kD,kH,kW)");
    int64_t N  = input.size(0);
    int64_t C  = input.size(1);
    int64_t D  = input.size(2);
    int64_t H  = input.size(3);
    int64_t W  = input.size(4);

    int64_t OC = weight.size(0);
    int64_t kD = weight.size(2);
    int64_t kH = weight.size(3);
    int64_t kW = weight.size(4);

    TORCH_CHECK(C % groups == 0, "C must be divisible by groups");
    TORCH_CHECK(OC % groups == 0, "OC must be divisible by groups");

    // Compute output sizes following PyTorch formula
    auto outD = (D + 2 * pad_d - dil_d * (kD - 1) - 1) / stride_d + 1;
    auto outH = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    auto outW = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "Output size is <= 0");

    auto output = torch::zeros({N, OC, outD, outH, outW}, input.options());

    // Grid / block
    long long total = (long long)N * OC * outD * outH * outW;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    conv3d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_flag ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        (int)N, (int)C, (int)D, (int)H, (int)W,
        (int)OC, (int)kD, (int)kH, (int)kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        groups,
        (int)outD, (int)outH, (int)outW,
        bias_flag ? 1 : 0
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv3d_forward_kernel launch failed with error: ", cudaGetErrorString(err));

    return output;
}
'''

cpp_src = r'''
torch::Tensor conv3d_forward(torch::Tensor input,
                             torch::Tensor weight,
                             torch::Tensor bias,
                             int stride_d, int stride_h, int stride_w,
                             int pad_d, int pad_h, int pad_w,
                             int dil_d, int dil_h, int dil_w,
                             int groups,
                             bool bias_flag);
'''

# ----------------------------------------------------------------------
# Inline load
# ----------------------------------------------------------------------
conv3d_cuda = load_inline(
    name='conv3d_cuda',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['conv3d_forward'],
    with_cuda=True,
    verbose=True,
    extra_cuda_cflags=['-O3', '--ptxas-options=-v'],
)

# ----------------------------------------------------------------------
# Python module that uses the CUDA kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False) -> None:
        super().__init__()

        # handle tuple / int inputs
        def _triple(v):
            if isinstance(v, int):
                return (v, v, v)
            return tuple(v)
        self.kernel_size = _triple(kernel_size)
        self.stride      = _triple(stride)
        self.padding     = _triple(padding)
        self.dilation    = _triple(dilation)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.groups       = groups

        # parameters
        kD, kH, kW = self.kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kD, kH, kW)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # initialization (same as nn.Conv3d default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * kD * kH * kW // groups
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous tensors
        x_in   = x.contiguous()
        weight = self.weight.contiguous()
        bias   = self.bias.contiguous() if self.bias is not None else torch.empty(0, device=x.device)

        out = conv3d_cuda.conv3d_forward(
            x_in,
            weight,
            bias,
            *self.stride,
            *self.padding,
            *self.dilation,
            self.groups,
            self.bias is not None
        )
        return out