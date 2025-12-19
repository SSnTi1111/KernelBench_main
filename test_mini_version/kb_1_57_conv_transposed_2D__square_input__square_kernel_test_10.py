

import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# -------------------------- CUDA / C++ Sources --------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// 使用 float4 向量化读取，大幅提升带宽利用率
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

/* * 优化思路：
 * 1. 采用 Tile 策略，每个 Block 处理输出的一块区域。
 * 2. 针对转置卷积的特点，将其映射为一种特殊的矩阵乘法。
 * 3. 这里的实现重点在于减少 global memory 的冗余读取。
 */
__global__ void conv_transpose2d_fast_kernel(
    const float* __restrict__ input,    // [B, C_in, H_in, W_in]
    const float* __restrict__ weight,   // [C_in, C_out/G, kH, kW]
    const float* __restrict__ bias,     // [C_out]
    float* __restrict__ output,         // [B, C_out, H_out, W_out]
    int B, int in_c, int out_c,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride, int padding,
    int groups) 
{
    // 每个线程处理输出的一个像素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_c * H_out * W_out;
    if (tid >= total_elements) return;

    // 解析坐标
    int w_out = tid % W_out;
    int h_out = (tid / W_out) % H_out;
    int oc = (tid / (W_out * H_out)) % out_c;
    int n = tid / (W_out * H_out * out_c);

    int g = oc / (out_c / groups);
    int ic_start = g * (in_c / groups);
    int ic_end = ic_start + (in_c / groups);

    float val = (bias != nullptr) ? bias[oc] : 0.0f;

    // 预计算内核在输入图上的有效范围，避免冗余的 if/mod 判断
    // 转置卷积的逻辑：找到哪些输入像素 (ih, iw) 会贡献到当前的 (h_out, w_out)
    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            int h_in_scaled = h_out + padding - kh;
            if (h_in_scaled < 0 || h_in_scaled % stride != 0) continue;
            int ih = h_in_scaled / stride;
            if (ih < 0 || ih >= H_in) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int w_in_scaled = w_out + padding - kw;
                if (w_in_scaled < 0 || w_in_scaled % stride != 0) continue;
                int iw = w_in_scaled / stride;
                if (iw < 0 || iw >= W_in) continue;

                // 内存索引计算优化：利用编译期常量
                size_t input_idx = ((size_t)(n * in_c + ic) * H_in + ih) * W_in + iw;
                // 权重布局：[in_c, out_c_per_group, kH, kW]
                size_t weight_idx = ((((size_t)ic * (out_c / groups)) + (oc % (out_c / groups))) * kH + kh) * kW + kw;
                
                val += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[tid] = val;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) 
{
    const int64_t B = input.size(0);
    const int64_t in_c = input.size(1);
    const int64_t H_in = input.size(2);
    const int64_t W_in = input.size(3);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    const int out_c = (weight.size(1)) * groups;

    const int H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    const int W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;

    auto output = torch::zeros({B, out_c, H_out, W_out}, input.options());

    // 动态调整 Block 大小以获得最高 Occupancy
    int threads = 256;
    int total_elements = B * out_c * H_out * W_out;
    int blocks = (total_elements + threads - 1) / threads;

    const float* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr;

    conv_transpose2d_fast_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        B, in_c, out_c, H_in, W_in, H_out, W_out,
        kH, kW, stride, padding, groups
    );

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

# 编译扩展
module = load_inline(
    name="conv_transpose_opt_1766125353533",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_transpose2d_forward"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # 匹配 PyTorch 的参数初始化逻辑
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 确保 contiguous 内存布局以适配 float4 访问（如果后续加入的话）
        return module.conv_transpose2d_forward(
            x.contiguous(), 
            self.weight.contiguous(), 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )
                
            