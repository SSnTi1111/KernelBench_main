#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_43_Max_Pooling_3D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4);

#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_43_Max_Pooling_3D_wrapper(torch::Tensor arg0,
                                           int64_t       arg1,
                                           int64_t       arg2,
                                           int64_t       arg3,
                                           int64_t       arg4);

#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <cfloat>
#include <climits>

// ------------------------------------------------------------
// Tiled-kernel compile-time knobs  (≤ 2 KB/block guarantee)
// ------------------------------------------------------------
#define TILE_OH            2
#define TILE_OW            2
#define MAX_SHARED_BYTES 2048

// ------------------------------------------------------------
// Constant-memory block that keeps all invariant hyper-parameters
// ------------------------------------------------------------
struct Pool3DConst {
    int N, C, D, H, W;      // input tensor sizes
    int OD, OH, OW;         // output tensor sizes
    int k, stride, pad, dil;
    int total;              // total number of output elements (≤ INT_MAX)
};
__constant__ Pool3DConst gP;

// ------------------------------------------------------------
// (Optional) helper – unused in max-pool kernel but kept from
// the original file to satisfy build dependencies.
// ------------------------------------------------------------
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的第一个线程写入 shared
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 第一个 warp 完成最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// ------------------------------------------------------------
// NEW: Tiled 3-D MaxPooling kernel (≤2 KB dyn. shared-mem)
// ------------------------------------------------------------
__global__ __launch_bounds__(32, 8) __attribute__((maxrregcount(32)))
void MaxPool3dKernelTile(const float* __restrict__ input,
                         float*       __restrict__ output) {
    extern __shared__ float sPatch[];

    const int ip_h = gP.k + (TILE_OH - 1) * gP.stride;
    const int ip_w = gP.k + (TILE_OW - 1) * gP.stride;
    const int ip_d = gP.k;
    const int patch_elems = ip_d * ip_h * ip_w;

    const int tid = threadIdx.x;

    // --- Decode blockIdx ---------------------------------------------------
    const int ow_tile = blockIdx.x;             // tiles in width  dimension
    const int oh_tile = blockIdx.y;             // tiles in height dimension
    int tmp          = blockIdx.z;              // pack (n,c,od)
    const int od     = tmp % gP.OD;  tmp /= gP.OD;
    const int  c     = tmp % gP.C;   tmp /= gP.C;
    const int  n     = tmp;                     // remaining

    const int ow_base = ow_tile * TILE_OW;
    const int oh_base = oh_tile * TILE_OH;

    // -----------------------------------------------------------------------
    // Cooperative patch load (all 32 threads)
    // -----------------------------------------------------------------------
    for (int idx = tid; idx < patch_elems; idx += blockDim.x) {
        int t = idx;
        const int iw = t % ip_w; t /= ip_w;
        const int ih = t % ip_h; t /= ip_h;
        const int id = t;                       // 0 .. k-1

        const int g_w = ow_base * gP.stride - gP.pad + iw;
        const int g_h = oh_base * gP.stride - gP.pad + ih;
        const int g_d = od       * gP.stride - gP.pad + id * gP.dil;

        float v = -FLT_MAX;
        if ((unsigned)g_w < (unsigned)gP.W &&
            (unsigned)g_h < (unsigned)gP.H &&
            (unsigned)g_d < (unsigned)gP.D) {
            int gidx = ((((n * gP.C + c) * gP.D + g_d) * gP.H + g_h) * gP.W + g_w);
            v = input[gidx];
        }
        sPatch[idx] = v;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // First 4 threads compute the TILE_OH×TILE_OW outputs
    // -----------------------------------------------------------------------
    const int t_h = tid & 1;          //      0 / 1
    const int t_w = (tid >> 1) & 1;   //      0 / 1

    if (t_h >= TILE_OH || t_w >= TILE_OW) return; // remaining threads exit

    const int oh = oh_base + t_h;
    const int ow = ow_base + t_w;
    if (oh >= gP.OH || ow >= gP.OW) return;       // boundary guard

    float maxVal = -FLT_MAX;

    for (int kd = 0; kd < gP.k; ++kd) {
        for (int kh = 0; kh < gP.k; ++kh) {
            for (int kw = 0; kw < gP.k; ++kw) {
                int ip_h_off = kh + t_h * gP.stride;
                int ip_w_off = kw + t_w * gP.stride;
                int sidx = ((kd * ip_h) + ip_h_off) * ip_w + ip_w_off;
                maxVal = fmaxf(maxVal, sPatch[sidx]);
            }
        }
    }

    int out_idx = ((((n * gP.C + c) * gP.OD + od) * gP.OH + oh) * gP.OW + ow);
    output[out_idx] = maxVal;
}

// ------------------------------------------------------------
// Reference 3-D MaxPooling kernel (unchanged)
// ------------------------------------------------------------
__global__ __launch_bounds__(256, 4) __attribute__((maxrregcount(32)))
void MaxPool3dKernel(const float* __restrict__ input,
                     float*       __restrict__ output) {
    const int total = gP.total;                      // constant-mem fetch
    const int strideGrid = blockDim.x * gridDim.x;   // 32-bit
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 32-bit

    for (int linear = tid; linear < total; linear += strideGrid) {
        int tmp = linear;

        // De-linearise (OW,OH,OD,C,N)
        const int ow = tmp % gP.OW; tmp /= gP.OW;
        const int oh = tmp % gP.OH; tmp /= gP.OH;
        const int od = tmp % gP.OD; tmp /= gP.OD;
        const int  c = tmp % gP.C;  tmp /= gP.C;
        const int  n = tmp;                    // remaining

        const int nc = n * gP.C + c;           // reused composite
        const int hStart = oh * gP.stride - gP.pad;
        const int wStart = ow * gP.stride - gP.pad;
        const int dStart = od * gP.stride - gP.pad;

        float maxVal = -FLT_MAX;

        // depth loop
        for (int kd = 0, id = dStart; kd < gP.k; ++kd, id += gP.dil) {
            if (static_cast<unsigned>(id) >= static_cast<unsigned>(gP.D)) continue;

            int dOffset = (nc * gP.D + id) * gP.H; // ((n*C+c)*D + id) * H

            // height loop
            for (int kh = 0, ih = hStart; kh < gP.k; ++kh, ih += gP.dil) {
                if (static_cast<unsigned>(ih) >= static_cast<unsigned>(gP.H)) continue;

                int hOffset = (dOffset + ih) * gP.W; // ... + ih) * W

                // width loop
                for (int kw = 0, iw = wStart; kw < gP.k; ++kw, iw += gP.dil) {
                    if (static_cast<unsigned>(iw) >= static_cast<unsigned>(gP.W)) continue;

                    float v = input[hOffset + iw];
                    if (v > maxVal) maxVal = v;
                }
            }
        }

        output[linear] = maxVal;
    }
}

// ------------------------------------------------------------
// C++ Wrapper – interface remains unchanged
// ------------------------------------------------------------
torch::Tensor kb_43_Max_Pooling_3D_wrapper(torch::Tensor arg0,
                                           int64_t       arg1,
                                           int64_t       arg2,
                                           int64_t       arg3,
                                           int64_t       arg4) {
    // ---- Sanity checks ----------------------------------------------------
    TORCH_CHECK(arg0.is_cuda(),     "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(arg0.dim()  == 5,   "Input must be a 5D tensor in NCDHW format");
    TORCH_CHECK(arg1 > 0,           "kernel_size must be > 0");
    TORCH_CHECK(arg2 > 0,           "stride must be > 0");
    TORCH_CHECK(arg3 >= 0,          "padding must be >= 0");
    TORCH_CHECK(arg4 > 0,           "dilation must be > 0");

    auto x = arg0.contiguous();
    TORCH_CHECK(x.numel() < INT_MAX, "Tensor is too large for 32-bit indexing");

    // ---- Gather dimensions ------------------------------------------------
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    const int64_t kernel_size = arg1;
    const int64_t stride      = arg2;
    const int64_t padding     = arg3;
    const int64_t dilation    = arg4;

    // ---- Compute output size (ceil_mode = false) --------------------------
    const int64_t k_eff = 1 + (kernel_size - 1) * dilation;
    auto out_size = [&](int64_t in) {
        int64_t out = (in + 2 * padding - k_eff) / stride + 1;
        return std::max<int64_t>(out, 0);
    };
    const int64_t OD = out_size(D);
    const int64_t OH = out_size(H);
    const int64_t OW = out_size(W);
    TORCH_CHECK(OD > 0 && OH > 0 && OW > 0,
                "Computed output dimensions must be > 0");

    // ---- Allocate output --------------------------------------------------
    auto out = torch::empty({N, C, OD, OH, OW}, x.options());

    // ---- Prepare constant-memory payload ----------------------------------
    Pool3DConst hostP;
    hostP.N  = static_cast<int>(N);
    hostP.C  = static_cast<int>(C);
    hostP.D  = static_cast<int>(D);
    hostP.H  = static_cast<int>(H);
    hostP.W  = static_cast<int>(W);

    hostP.OD = static_cast<int>(OD);
    hostP.OH = static_cast<int>(OH);
    hostP.OW = static_cast<int>(OW);

    hostP.k      = static_cast<int>(kernel_size);
    hostP.stride = static_cast<int>(stride);
    hostP.pad    = static_cast<int>(padding);
    hostP.dil    = static_cast<int>(dilation);

    int64_t total64 = N * C * OD * OH * OW;
    TORCH_CHECK(total64 < INT_MAX,
                "Total number of output elements exceeds 32-bit range");
    hostP.total = static_cast<int>(total64);

    // copy to device constant memory
    cudaMemcpyToSymbolAsync(gP, &hostP, sizeof(Pool3DConst), 0,
                            cudaMemcpyHostToDevice,
                            at::cuda::getCurrentCUDAStream());

    // ---- Launch kernel ----------------------------------------------------
    const int threads = 256;
    int blocks = static_cast<int>((hostP.total + threads - 1) / threads);
    blocks = std::min(blocks, 65535); // stay within grid-size limit

    auto stream = at::cuda::getCurrentCUDAStream();
    MaxPool3dKernel<<<blocks, threads, 0, stream.stream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>());

    // ---- Error check ------------------------------------------------------
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "MaxPool3dKernel launch failed: ",
                cudaGetErrorString(err));

    return out;
}