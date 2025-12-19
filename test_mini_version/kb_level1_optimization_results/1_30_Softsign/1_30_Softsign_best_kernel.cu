import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ source --------------------------------------------------------------
source = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdint>

// ================================================================
// 128-bit vector traits
// ------------------------------------------------
template<typename T> struct VecTraits;

template<> struct VecTraits<float>         { using VecT = float4;  static constexpr int kValues = 4; };
template<> struct VecTraits<double>        { using VecT = double2; static constexpr int kValues = 2; };
template<> struct VecTraits<__half>        { using VecT = uint4;   static constexpr int kValues = 8; };
template<> struct VecTraits<__nv_bfloat16> { using VecT = uint4;   static constexpr int kValues = 8; };

// ================================================================
// helpers
// ------------------------------------------------
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v){
    uint32_t tmp = static_cast<uint32_t>(
        *reinterpret_cast<const unsigned short*>(&v)) << 16;
    return __uint_as_float(tmp);
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t softsign_scalar(scalar_t v){
    scalar_t av = v >= scalar_t(0) ? v : -v;
    return v / (scalar_t(1) + av);
}

template<>
__device__ __forceinline__ __half softsign_scalar(__half v){
    float fv  = __half2float(v);
    float res = fv / (1.f + fabsf(fv));
    return __float2half(res);
}
template<>
__device__ __forceinline__ __nv_bfloat16 softsign_scalar(__nv_bfloat16 v){
    float fv  = bf16_to_float(v);
    float res = fv / (1.f + fabsf(fv));
    return __float2bfloat16(res);
}

// ================================================================
//  main kernel: can optionally add a 2nd tensor before Softsign
// ------------------------------------------------
template<typename scalar_t, bool DoAdd>
__global__ void softsign_vec_kernel_fused(const scalar_t* __restrict__ a,
                                          const scalar_t* __restrict__ b,   // can be nullptr
                                          scalar_t*       __restrict__ y,
                                          const size_t N){
    using Traits          = VecTraits<scalar_t>;
    using VecT            = typename Traits::VecT;
    constexpr int VecSize = Traits::kValues;

    const VecT* __restrict__ a_vec = reinterpret_cast<const VecT*>(a);
    const VecT* __restrict__ b_vec = reinterpret_cast<const VecT*>(b);
    VecT*       __restrict__ y_vec = reinterpret_cast<VecT*>(y);

    const size_t vec_elems = N / VecSize;
    const size_t tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride    = gridDim.x * blockDim.x;

    // ---------------- Vector loop -----------------------------
    for (size_t idx = tid; idx < vec_elems; idx += stride){
        VecT inA  = a_vec[idx];
        VecT sumV;

        if constexpr (DoAdd){
            VecT inB = b_vec[idx];
            // ---- float --------------------------------------
            if constexpr (std::is_same<scalar_t,float>::value){
                float*  ap = reinterpret_cast<float*>(&inA);
                float*  bp = reinterpret_cast<float*>(&inB);
                float*  sp = reinterpret_cast<float*>(&sumV);
                #pragma unroll
                for (int k=0;k<VecSize;++k) sp[k] = ap[k] + bp[k];
            }
            // ---- double -------------------------------------
            else if constexpr (std::is_same<scalar_t,double>::value){
                double* ap = reinterpret_cast<double*>(&inA);
                double* bp = reinterpret_cast<double*>(&inB);
                double* sp = reinterpret_cast<double*>(&sumV);
                #pragma unroll
                for (int k=0;k<VecSize;++k) sp[k] = ap[k] + bp[k];
            }
            // ---- half ---------------------------------------
            else if constexpr (std::is_same<scalar_t,__half>::value){
                uint32_t* ap = reinterpret_cast<uint32_t*>(&inA);
                uint32_t* bp = reinterpret_cast<uint32_t*>(&inB);
                uint32_t* sp = reinterpret_cast<uint32_t*>(&sumV);
                #pragma unroll
                for (int word=0; word<4; ++word){
                    __half2 va = *reinterpret_cast<__half2*>(&ap[word]);
                    __half2 vb = *reinterpret_cast<__half2*>(&bp[word]);
                    __half2 vs = __hadd2(va, vb);
                    sp[word]   = *reinterpret_cast<uint32_t*>(&vs);
                }
            }
            // ---- bf16 ---------------------------------------
            else{
                __nv_bfloat16* ap = reinterpret_cast<__nv_bfloat16*>(&inA);
                __nv_bfloat16* bp = reinterpret_cast<__nv_bfloat16*>(&inB);
                __nv_bfloat16* sp = reinterpret_cast<__nv_bfloat16*>(&sumV);
                #pragma unroll
                for (int k=0;k<VecSize;++k) sp[k] = __hadd(ap[k], bp[k]); // __hadd works for bf16 too
            }
        }

        VecT outV;

        // Apply softsign (vectorised)
        VecT* src = nullptr;
        if constexpr (DoAdd) src = &sumV;
        else                 src = &inA;

        // ---- float ------------------------------------------
        if constexpr (std::is_same<scalar_t,float>::value){
            float*  in  = reinterpret_cast<float*>(src);
            float*  out = reinterpret_cast<float*>(&outV);
            #pragma unroll
            for (int k=0;k<VecSize;++k){
                float v  = in[k];
                float av = fabsf(v);
                out[k]   = v / (1.f + av);
            }
        }
        // ---- double -----------------------------------------
        else if constexpr (std::is_same<scalar_t,double>::value){
            double* in  = reinterpret_cast<double*>(src);
            double* out = reinterpret_cast<double*>(&outV);
            #pragma unroll
            for (int k=0;k<VecSize;++k){
                double v  = in[k];
                double av = fabs(v);
                out[k]    = v / (1.0 + av);
            }
        }
        // ---- half -------------------------------------------
        else if constexpr (std::is_same<scalar_t,__half>::value){
            uint32_t* in  = reinterpret_cast<uint32_t*>(src);
            uint32_t* out = reinterpret_cast<uint32_t*>(&outV);
            #pragma unroll
            for (int word=0; word<4; ++word){
                __half2 val    = *reinterpret_cast<__half2*>(&in[word]);
                __half2 abs_v  = __habs2(val);
                __half2 denom  = __hadd2(abs_v, __half2half2(__float2half(1.f)));
                __half2 res    = __h2div(val, denom);
                out[word]      = *reinterpret_cast<uint32_t*>(&res);
            }
        }
        // ---- bf16 -------------------------------------------
        else{
            __nv_bfloat16* in  = reinterpret_cast<__nv_bfloat16*>(src);
            __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(&outV);
            #pragma unroll
            for (int k=0;k<VecSize;++k){
                out[k] = softsign_scalar<__nv_bfloat16>(in[k]);
            }
        }
        y_vec[idx] = outV;
    }

    // ---------------- Tail elements -------------------------
    const size_t base = vec_elems * VecSize;
    for (size_t i = base + tid; i < N; i += stride){
        scalar_t val = a[i];
        if constexpr (DoAdd){
            // Handle addition for different dtypes
            if constexpr (std::is_same<scalar_t,float>::value ||
                          std::is_same<scalar_t,double>::value){
                val = val + b[i];
            }
            else if constexpr (std::is_same<scalar_t,__half>::value){
                val = __hadd(val, b[i]);
            }
            else{ // bf16
                val = __hadd(val, b[i]);   // __hadd works for bf16 too
            }
        }
        y[i] = softsign_scalar<scalar_t>(val);
    }
}

// ================================================================
//  Host API
// ------------------------------------------------
namespace {

// NOTE:
// Using data_ptr<__half>() or data_ptr<__nv_bfloat16>() requires
// PyTorch to provide explicit instantiations for those template
// arguments.  Older / some builds of PyTorch do not export them,
// leading to an "undefined symbol" linker error.  To avoid that,
// we fetch an untyped (void*) pointer via data_ptr() and cast it
// to the desired CUDA scalar type.

// ----------------------------------------------------------------
template<typename scalar_t>
void launch_softsign(torch::Tensor a, torch::Tensor out){
    const size_t N = a.numel();
    constexpr int threads = 256;
    const int blocks      = std::min<int>((N + threads - 1) / threads, 65535);
    auto stream = at::cuda::getCurrentCUDAStream();

    softsign_vec_kernel_fused<scalar_t,false><<<blocks,threads,0,stream>>>(
        reinterpret_cast<const scalar_t*>(a.data_ptr()),
        nullptr,
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        N);
}

template<typename scalar_t>
void launch_softsign_add(torch::Tensor a, torch::Tensor b, torch::Tensor out){
    const size_t N = a.numel();
    constexpr int threads = 256;
    const int blocks      = std::min<int>((N + threads - 1) / threads, 65535);
    auto stream = at::cuda::getCurrentCUDAStream();

    softsign_vec_kernel_fused<scalar_t,true><<<blocks,threads,0,stream>>>(
        reinterpret_cast<const scalar_t*>(a.data_ptr()),
        reinterpret_cast<const scalar_t*>(b.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        N);
}

} // anonymous namespace

// single-input variant (compat)
torch::Tensor softsign_cuda(torch::Tensor x){
    auto out = torch::empty_like(x);

    switch (x.scalar_type()){
        case torch::kFloat:    launch_softsign<float>(x,out);                      break;
        case torch::kDouble:   launch_softsign<double>(x,out);                     break;
        case torch::kHalf:     launch_softsign<__half>(x,out);                     break;
        case torch::kBFloat16: launch_softsign<__nv_bfloat16>(x,out);              break;
        default: AT_ERROR("softsign_cuda: unsupported dtype");
    }
    return out;
}

// fused add + softsign  (optional)
torch::Tensor softsign_add_cuda(torch::Tensor a, torch::Tensor b){
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same shape");
    auto out = torch::empty_like(a);

    switch (a.scalar_type()){
        case torch::kFloat:    launch_softsign_add<float>(a,b,out);                break;
        case torch::kDouble:   launch_softsign_add<double>(a,b,out);               break;
        case torch::kHalf:     launch_softsign_add<__half>(a,b,out);               break;
        case torch::kBFloat16: launch_softsign_add<__nv_bfloat16>(a,b,out);        break;
        default: AT_ERROR("softsign_add_cuda: unsupported dtype");
    }
    return out;
}
'''

# Tiny C++ declaration stub ----------------------------------------------------
cpp_src = r'''
torch::Tensor softsign_cuda(torch::Tensor x);
torch::Tensor softsign_add_cuda(torch::Tensor a, torch::Tensor b);
'''

# Build / load extension -------------------------------------------------------
softsign_mod = load_inline(
    name         = 'softsign_opt_fused',
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ['softsign_cuda', 'softsign_add_cuda'],
    with_cuda    = True,
    verbose      = True,                          # <-- keep verbose build
    extra_cuda_cflags=['-O3', '--ptxas-options=-v', '-std=c++17'],  # <-- ptxas flags
    extra_cflags=['-O3', '-std=c++17'],
)

# ----------------------------------------------------------------------------- 
# PyTorch Module wrapper
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = softsign_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod.softsign_cuda(x)