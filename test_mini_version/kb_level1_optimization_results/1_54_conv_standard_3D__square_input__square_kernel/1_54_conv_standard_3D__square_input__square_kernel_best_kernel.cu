import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA/C++ source
# ---------------------------------------------------------------------------
source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_H_OUT 8
#define TILE_W_OUT 8
#define TILE_D_OUT 1   // fixed

template <typename scalar_t>
__global__ void conv3d_forward_kernel_tiled(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        const int64_t N,
        const int64_t C_in,
        const int64_t D_in,
        const int64_t H_in,
        const int64_t W_in,
        const int64_t C_out,
        const int64_t Kd,
        const int64_t Kh,
        const int64_t Kw,
        const int      stride,
        const int      padding,
        const int      dilation,
        const int64_t D_out,
        const int64_t H_out,
        const int64_t W_out,
        const bool     bias_defined)
{
    /* ----------------------- block → logical coordinates ------------------ */
    const int w_tile = blockIdx.x;
    const int h_tile = blockIdx.y;

    int64_t ncd = blockIdx.z;
    const int64_t d_out = ncd % D_out;
    ncd /= D_out;
    const int64_t oc = ncd % C_out;
    const int64_t n  = ncd / C_out;

    /* thread coords inside tile */
    const int tw = threadIdx.x;   // 0 … TILE_W_OUT-1
    const int th = threadIdx.y;   // 0 … TILE_H_OUT-1

    const int64_t w_out = w_tile * TILE_W_OUT + tw;
    const int64_t h_out = h_tile * TILE_H_OUT + th;

    /* determine validity of this thread's output element */
    const bool is_valid = (w_out < W_out) && (h_out < H_out) &&
                          (d_out < D_out) && (oc < C_out) && (n < N);

    /* ----------------- Derived tile sizes (run-time due to stride/ dilation) */
    const int TILE_H_IN = TILE_H_OUT * stride + (Kh - 1) * dilation;
    const int TILE_W_IN = TILE_W_OUT * stride + (Kw - 1) * dilation;
    const int64_t TILE_ELEMS = static_cast<int64_t>(Kd) * TILE_H_IN * TILE_W_IN;

    extern __shared__ char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    /* bases in input tensor */
    const int64_t d_in_base = d_out * stride - padding;
    const int64_t h_in_base = h_tile * TILE_H_OUT * stride - padding;
    const int64_t w_in_base = w_tile * TILE_W_OUT * stride - padding;

    /* local offsets for this output element in the shared tile */
    const int loc_h = th * stride;
    const int loc_w = tw * stride;

    scalar_t acc = static_cast<scalar_t>(0);

    /* --------------------------- main loop over C_in ---------------------- */
    for (int64_t ic = 0; ic < C_in; ++ic)
    {
        /* cooperative load */
        for (int64_t idx = th * TILE_W_OUT + tw; idx < TILE_ELEMS;
             idx += TILE_H_OUT * TILE_W_OUT)
        {
            int64_t t = idx;
            const int iw_tile = t % TILE_W_IN;   t /= TILE_W_IN;
            const int ih_tile = t % TILE_H_IN;   t /= TILE_H_IN;
            const int kd_tile = t;               // 0 … Kd-1

            const int64_t d_in = d_in_base + kd_tile * dilation;
            const int64_t h_in = h_in_base + ih_tile;
            const int64_t w_in = w_in_base + iw_tile;

            scalar_t val = static_cast<scalar_t>(0);
            if (d_in >= 0 && d_in < D_in &&
                h_in >= 0 && h_in < H_in &&
                w_in >= 0 && w_in < W_in)
            {
                const int64_t input_idx =
                    (((n * C_in + ic) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                val = input[input_idx];
            }
            smem[idx] = val;
        }
        __syncthreads();

        /* compute */
        for (int64_t kd = 0; kd < Kd; ++kd)
        {
            int base = (kd * TILE_H_IN + loc_h) * TILE_W_IN + loc_w;
            for (int64_t kh = 0; kh < Kh; ++kh)
            {
                int row = base + kh * dilation * TILE_W_IN;
                for (int64_t kw = 0; kw < Kw; ++kw)
                {
                    scalar_t in_val = smem[row + kw * dilation];
                    int64_t w_idx =
                        ((((oc * C_in + ic) * Kd + kd) * Kh + kh) * Kw + kw);
                    acc = fma(in_val, weight[w_idx], acc);
                }
            }
        }
        __syncthreads();   // ensure smem not overwritten before all threads finish
    }

    if (bias_defined)
        acc += bias[oc];

    /* write back result only for valid threads */
    if (is_valid)
    {
        const int64_t out_idx =
            (((n * C_out + oc) * D_out + d_out) * H_out + h_out) * W_out + w_out;
        output[out_idx] = acc;
    }
}


/* ---------------------------  HOST INTERFACE  ---------------------------- */
torch::Tensor conv3d_cuda(torch::Tensor input,
                          torch::Tensor weight,
                          torch::Tensor bias,
                          int stride,
                          int padding,
                          int dilation)
{
    TORCH_CHECK(input.is_cuda(),  "input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const bool bias_defined = bias.defined() && bias.numel() > 0;
    if (bias_defined) {
        TORCH_CHECK(bias.is_cuda(), "bias must be on CUDA");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "only float32 bias supported");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }

    /* ------------------- tensor & kernel parameters ---------------------- */
    const auto  sizes   = input.sizes();
    const int64_t N     = sizes[0];
    const int64_t C_in  = sizes[1];
    const int64_t D_in  = sizes[2];
    const int64_t H_in  = sizes[3];
    const int64_t W_in  = sizes[4];

    const auto  ws      = weight.sizes();
    const int64_t C_out = ws[0];
    const int64_t Kd    = ws[2];
    const int64_t Kh    = ws[3];
    const int64_t Kw    = ws[4];
    TORCH_CHECK(ws[1] == C_in, "groups != 1 is not supported");

    const int64_t D_out =
        (D_in + 2 * padding - dilation * (Kd - 1) - 1) / stride + 1;
    const int64_t H_out =
        (H_in + 2 * padding - dilation * (Kh - 1) - 1) / stride + 1;
    const int64_t W_out =
        (W_in + 2 * padding - dilation * (Kw - 1) - 1) / stride + 1;
    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Invalid output size");

    auto output = torch::empty({N, C_out, D_out, H_out, W_out},
                               input.options());

    /* ----------------------- launch configuration ------------------------ */
    const dim3 block(TILE_W_OUT, TILE_H_OUT, 1);

    const dim3 grid(
        (W_out + TILE_W_OUT - 1) / TILE_W_OUT,
        (H_out + TILE_H_OUT - 1) / TILE_H_OUT,
        N * C_out * D_out);

    const int TILE_H_IN = TILE_H_OUT * stride + (Kh - 1) * dilation;
    const int TILE_W_IN = TILE_W_OUT * stride + (Kw - 1) * dilation;
    const size_t smem_bytes = static_cast<size_t>(Kd) *
                              TILE_H_IN * TILE_W_IN *
                              sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward_cuda_tiled", ([&] {
        conv3d_forward_kernel_tiled<scalar_t>
            <<<grid, block, smem_bytes>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias_defined ? bias.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                N, C_in, D_in, H_in, W_in,
                C_out,
                Kd, Kh, Kw,
                stride, padding, dilation,
                D_out, H_out, W_out,
                bias_defined);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv3d_forward_kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}
'''

# -----------------------------  C++ stub  -----------------------------------
cpp_src = r'''
torch::Tensor conv3d_cuda(torch::Tensor input,
                          torch::Tensor weight,
                          torch::Tensor bias,
                          int stride,
                          int padding,
                          int dilation);
'''

# ---------------------------------------------------------------------------
# Build the extension
# ---------------------------------------------------------------------------
conv3d_cuda = load_inline(
    name='conv3d_cuda_opt',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['conv3d_cuda'],
    with_cuda=True,
    verbose=True,               # mandated
    extra_cuda_cflags=['-O3', '--ptxas-options=-v']
)

# ---------------------------------------------------------------------------
# Python-side wrapper class
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False):
        super().__init__()
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation
        self.groups   = groups  # kept for API but only 1 supported

        w_shape = (out_channels, in_channels // groups,
                   kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape, device='cuda'))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device='cuda'))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            fan_in = in_channels * kernel_size * kernel_size * kernel_size
            bound  = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_cuda.conv3d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.contiguous() if self.bias is not None else torch.Tensor(),
            self.stride,
            self.padding,
            self.dilation
        )