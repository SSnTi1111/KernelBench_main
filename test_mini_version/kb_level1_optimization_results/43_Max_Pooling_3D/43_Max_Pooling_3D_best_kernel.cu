#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_43_Max_Pooling_3D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <limits>
// [!!! 关键 !!!] 
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// CUDA 内核实现: 3D Max Pooling（支持 stride、padding、dilation，ceil_mode=false）
__global__ void maxpool3d_forward_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    long long N, long long C,
    long long D, long long H, long long W,
    int k, int stride, int pad, int dilation,
    long long D_out, long long H_out, long long W_out
) {
    // 动态共享内存指针（当以平铺方式启动时使用）
    // 注意：s_tile 的实际大小由 kernel 启动时的动态共享内存参数决定，
    // 若未分配（例如 shared_bytes=0），则不得进入共享内存路径。
    extern __shared__ float s_tile[];
    int threads_w = blockDim.x;

    // 预计算有效核大小以减少重复乘法
    long long eff_k = ((long long)dilation * ((long long)k - 1LL)) + 1LL;

    // 当以特殊平铺配置启动（例如 threads_w == 32）时，使用共享内存优化路径；
    // 否则回退到原始的每线程独立计算路径。
    if (threads_w == 32) {
        // 计算水平切片数（每个 block 处理 threads_w 个输出宽度）
        int slices = (int)((W_out + threads_w - 1) / threads_w);
        long long bidx = (long long)blockIdx.x;
        int slice_id = (int)(bidx % slices);
        long long outer_bidx = bidx / slices;

        // 反解 outer_bidx -> (n, c, od, oh)，不含 ow
        long long tmp = outer_bidx;
        long long oh = tmp % H_out; tmp /= H_out;
        long long od = tmp % D_out; tmp /= D_out;
        long long c  = tmp % C;     tmp /= C;
        long long n  = tmp;

        // 该 block 的 ow 起点与当前线程的 ow
        long long ow_start = (long long)slice_id * threads_w;
        long long ow = ow_start + threadIdx.x;
        if (ow >= W_out) return;

        // 对应输出窗口的起始位置（考虑 padding 与 stride）
        long long start_d = od * (long long)stride - (long long)pad;
        long long start_h = oh * (long long)stride - (long long)pad;
        long long start_w = ow * (long long)stride - (long long)pad;

        // 需要加载的输入 slab 边界（clip 到合法范围）
        long long min_iw = ow_start * (long long)stride - (long long)pad;
        long long max_iw = (ow_start + threads_w - 1) * (long long)stride - (long long)pad + (eff_k - 1LL);

        long long lw_start_ll = min_iw > 0 ? min_iw : 0LL;
        long long lw_end_ll   = max_iw + 1LL;
        if (lw_end_ll < 0) lw_end_ll = 0LL;
        if (lw_end_ll > W) lw_end_ll = W;
        int load_w_start = (int)lw_start_ll;
        int load_w_end   = (int)lw_end_ll;
        int load_width   = load_w_end - load_w_start;

        long long min_id = start_d;
        long long max_id = start_d + (eff_k - 1LL);
        long long ld_start_ll = min_id > 0 ? min_id : 0LL;
        long long ld_end_ll   = max_id + 1LL;
        if (ld_end_ll < 0) ld_end_ll = 0LL;
        if (ld_end_ll > D) ld_end_ll = D;
        int load_d_start = (int)ld_start_ll;
        int load_d_end   = (int)ld_end_ll;
        int load_depth   = load_d_end - load_d_start;

        long long min_ih = start_h;
        long long max_ih = start_h + (eff_k - 1LL);
        long long lh_start_ll = min_ih > 0 ? min_ih : 0LL;
        long long lh_end_ll   = max_ih + 1LL;
        if (lh_end_ll < 0) lh_end_ll = 0LL;
        if (lh_end_ll > H) lh_end_ll = H;
        int load_h_start = (int)lh_start_ll;
        int load_h_end   = (int)lh_end_ll;
        int load_height  = load_h_end - load_h_start;

        // 协同加载输入 slab 到共享内存，沿 W 方向合并访存
        if (load_depth > 0 && load_height > 0 && load_width > 0) {
            for (int ld_idx = 0; ld_idx < load_depth; ++ld_idx) {
                long long id = (long long)load_d_start + (long long)ld_idx;
                for (int lh_idx = 0; lh_idx < load_height; ++lh_idx) {
                    long long ih = (long long)load_h_start + (long long)lh_idx;
                    int offset_base = (ld_idx * load_height + lh_idx) * load_width;
                    int num_passes = (load_width + threads_w - 1) / threads_w;
                    for (int pass = 0; pass < num_passes; ++pass) {
                        int gw = load_w_start + pass * threads_w + threadIdx.x;
                        if (gw < load_w_end) {
                            long long in_index = (((n * C + c) * D + id) * H + ih) * W + (long long)gw;
                            int sw_idx = gw - load_w_start;
                            s_tile[offset_base + sw_idx] = in[in_index];
                        }
                    }
                }
            }
        }
        __syncthreads();

        // 在共享内存中计算当前线程的池化窗口最大值
        float maxval = -FLT_MAX;
        for (int kd = 0; kd < k; ++kd) {
            long long id = start_d + (long long)kd * (long long)dilation;
            if (id < 0 || id >= D) continue;
            int ld_idx = (int)(id - (long long)load_d_start);
            if (ld_idx < 0 || ld_idx >= load_depth) continue;

            for (int kh = 0; kh < k; ++kh) {
                long long ih = start_h + (long long)kh * (long long)dilation;
                if (ih < 0 || ih >= H) continue;
                int lh_idx = (int)(ih - (long long)load_h_start);
                if (lh_idx < 0 || lh_idx >= load_height) continue;

                // 预先计算 base 偏移以减少乘法次数
                int base = (ld_idx * load_height + lh_idx) * load_width;

                for (int kw = 0; kw < k; ++kw) {
                    long long iw = start_w + (long long)kw * (long long)dilation;
                    if (iw < 0 || iw >= W) continue;
                    int lw_idx = (int)(iw - (long long)load_w_start);
                    if (lw_idx < 0 || lw_idx >= load_width) continue;

                    float val = s_tile[base + lw_idx];
                    if (val > maxval) maxval = val;
                }
            }
        }

        long long out_index = (((n * C + c) * D_out + od) * H_out + oh) * W_out + ow;
        out[out_index] = maxval;
        return;
    }

    // 回退路径：原始线性索引实现（不使用共享内存）
    long long total = N * C * D_out * H_out * W_out;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // 反解线性索引 -> (n, c, od, oh, ow)
    long long ow = idx % W_out;
    long long tmp2 = idx / W_out;
    long long oh2 = tmp2 % H_out; tmp2 /= H_out;
    long long od2 = tmp2 % D_out; tmp2 /= D_out;
    long long c2  = tmp2 % C;     tmp2 /= C;
    long long n2  = tmp2;

    // 计算池化窗口起始位置（考虑padding与stride）
    long long start_d2 = od2 * (long long)stride - (long long)pad;
    long long start_h2 = oh2 * (long long)stride - (long long)pad;
    long long start_w2 = ow  * (long long)stride - (long long)pad;

    float maxval2 = -FLT_MAX;

    // 遍历核窗口（含dilation）
    for (int kd = 0; kd < k; ++kd) {
        long long id = start_d2 + (long long)kd * (long long)dilation;
        if (id < 0 || id >= D) continue;
        for (int kh = 0; kh < k; ++kh) {
            long long ih = start_h2 + (long long)kh * (long long)dilation;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < k; ++kw) {
                long long iw = start_w2 + (long long)kw * (long long)dilation;
                if (iw < 0 || iw >= W) continue;

                long long in_index = (((n2 * C + c2) * D + id) * H + ih) * W + iw;
                float val = in[in_index];
                if (val > maxval2) {
                    maxval2 = val;
                }
            }
        }
    }

    long long out_index2 = (((n2 * C + c2) * D_out + od2) * H_out + oh2) * W_out + ow;
    out[out_index2] = maxval2;
}

// C++ Wrapper 实现
torch::Tensor kb_43_Max_Pooling_3D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4) {
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "Only float32 dtype is supported");
    TORCH_CHECK(arg0.dim() == 5, "Input must be a 5D tensor [N, C, D, H, W]");

    // 参数
    int64_t k = arg1;        // kernel_size
    int64_t stride = (arg2 > 0) ? arg2 : arg1; // 若为None，stride=kernel_size
    int64_t pad = arg3;      // padding
    int64_t dilation = arg4; // dilation
    TORCH_CHECK(k > 0, "kernel_size must be > 0");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(pad >= 0, "padding must be >= 0");
    TORCH_CHECK(dilation > 0, "dilation must be > 0");

    // 保证连续
    auto x = arg0.contiguous();

    // 尺寸
    long long N = x.size(0);
    long long C = x.size(1);
    long long D = x.size(2);
    long long H = x.size(3);
    long long W = x.size(4);

    // 输出尺寸 (ceil_mode = False)
    long long eff_k = dilation * (k - 1) + 1; // effective kernel
    long long D_out = (D + 2 * pad - eff_k) >= 0 ? ((D + 2 * pad - eff_k) / stride + 1) : 0;
    long long H_out = (H + 2 * pad - eff_k) >= 0 ? ((H + 2 * pad - eff_k) / stride + 1) : 0;
    long long W_out = (W + 2 * pad - eff_k) >= 0 ? ((W + 2 * pad - eff_k) / stride + 1) : 0;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Calculated output size is non-positive. Check your parameters.");

    // 分配输出
    auto out = torch::empty({N, C, D_out, H_out, W_out}, x.options());

    // 指针
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    // 启动配置
    long long total = N * C * D_out * H_out * W_out;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    // 调用内核
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    maxpool3d_forward_kernel<<<blocks, threads, 0, stream>>>(
        in_ptr, out_ptr,
        N, C, D, H, W,
        static_cast<int>(k),
        static_cast<int>(stride),
        static_cast<int>(pad),
        static_cast<int>(dilation),
        D_out, H_out, W_out
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "maxpool3d_forward_kernel launch failed: ", cudaGetErrorString(err));

    return out;
}