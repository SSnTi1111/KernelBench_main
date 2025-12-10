#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Helpers for Morton (Z-order) decoding
__device__ __forceinline__ uint32_t compact1by1(uint32_t x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

__device__ __forceinline__ uint32_t morton_decode_x(uint32_t code) {
    return compact1by1(code);
}

__device__ __forceinline__ uint32_t morton_decode_y(uint32_t code) {
    return compact1by1(code >> 1);
}

__device__ __forceinline__ uint32_t highest_pow2_leq(uint32_t n) {
    uint32_t p = 1u;
    while ((p << 1u) <= n) { p <<= 1u; }
    return p;
}

// ------------------------------------------------------------------
// KERNEL: gemm_kernel 
// ------------------------------------------------------------------
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // 寄存器分块(每线程计算4x4输出) + 共享内存双缓冲 + 软件流水
    // 本版本对全局内存加载进行了float4向量化：
    //  - A的共享内存布局重构为 [BLOCK_SIZE][4*BLOCK_SIZE]，第一维为K子块(0..15)，第二维为tile内的行(0..63)
    //    每个线程沿K维一次加载连续4个元素(float4)并存入对应的K位置与行位置。
    //  - B的共享内存维持为 [BLOCK_SIZE][4*BLOCK_SIZE]，第一维为K子块行(0..15)，第二维为tile内的列(0..63)
    //    每个线程对B的同一K行加载连续4个列元素(float4)。
    __shared__ float Asub_ping[BLOCK_SIZE][4 * BLOCK_SIZE];
    __shared__ float Asub_pong[BLOCK_SIZE][4 * BLOCK_SIZE];
    __shared__ float Bsub_ping[BLOCK_SIZE][4 * BLOCK_SIZE];
    __shared__ float Bsub_pong[BLOCK_SIZE][4 * BLOCK_SIZE];

    int tid_y = threadIdx.y; // [0, BLOCK_SIZE)
    int tid_x = threadIdx.x; // [0, BLOCK_SIZE)

    // -------------------------------
    // Z-Order (Morton) block mapping
    // -------------------------------
    const int TILE_SIZE = 4 * BLOCK_SIZE; // 64
    // 需要的tile数量（每tile覆盖64x64输出）
    uint32_t num_tiles_x = (N + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t num_tiles_y = (N + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t total_tiles  = num_tiles_x * num_tiles_y;

    // 将二维block索引按行主序压成一维id
    uint32_t block_id = static_cast<uint32_t>(blockIdx.y) * static_cast<uint32_t>(gridDim.x)
                      + static_cast<uint32_t>(blockIdx.x);

    // 若当前block超出有效tile数量，提前返回（避免重复/越界计算）
    if (block_id >= total_tiles) {
        return;
    }

    // 在矩形区域内实现高效的Z-order映射：
    // 1) 先对能容纳的最大2^k方形区域（q x q）使用Morton解码
    // 2) 对右侧条带(q..w-1, 0..q-1)和底部条带(0..w-1, q..h-1)使用线性映射
    uint32_t w = num_tiles_x;
    uint32_t h = num_tiles_y;
    uint32_t q = highest_pow2_leq(min(w, h)); // 最大适配的方形尺寸(2^k)

    uint32_t tile_z_x, tile_z_y;
    uint32_t square_count = q * q;
    uint32_t right_strip_count = q * (w - q);

    if (block_id < square_count) {
        // 方形区域内的Morton解码
        uint32_t code = block_id;
        tile_z_x = morton_decode_x(code); // [0, q)
        tile_z_y = morton_decode_y(code); // [0, q)
    } else if (block_id < square_count + right_strip_count) {
        // 右侧条带：x in [q, w-1], y in [0, q-1]
        uint32_t i = block_id - square_count;
        uint32_t rs_w = w - q; // 可能为0
        // 当rs_w为0时，此分支不会进入（因为right_strip_count为0）
        tile_z_x = q + (i % rs_w);
        tile_z_y = (i / rs_w);
    } else {
        // 底部条带：x in [0, w-1], y in [q, h-1]
        uint32_t i = block_id - square_count - right_strip_count;
        uint32_t bs_h = h - q; // 可能为0
        // 当bs_h为0时，此分支不会进入（因为剩余数量为0）
        tile_z_y = q + (i % bs_h);
        tile_z_x = (i / bs_h);
    }

    // 当前块对应的输出tile起始坐标（每块覆盖64x64输出）
    int tile_row_base = static_cast<int>(tile_z_y) * TILE_SIZE;
    int tile_col_base = static_cast<int>(tile_z_x) * TILE_SIZE;

    // 若整个tile已完全越界，则直接返回
    if (tile_row_base >= N || tile_col_base >= N) {
        return;
    }

    // 每线程负责的4x4输出位置（相对于tile起始）
    int a_r0_local = tid_y * 4 + 0;
    int a_r1_local = tid_y * 4 + 1;
    int a_r2_local = tid_y * 4 + 2;
    int a_r3_local = tid_y * 4 + 3;

    int b_c0_local = tid_x * 4 + 0;
    int b_c1_local = tid_x * 4 + 1;
    int b_c2_local = tid_x * 4 + 2;
    int b_c3_local = tid_x * 4 + 3;

    int r0 = tile_row_base + a_r0_local;
    int r1 = tile_row_base + a_r1_local;
    int r2 = tile_row_base + a_r2_local;
    int r3 = tile_row_base + a_r3_local;

    int c0 = tile_col_base + b_c0_local;
    int c1 = tile_col_base + b_c1_local;
    int c2 = tile_col_base + b_c2_local;
    int c3 = tile_col_base + b_c3_local;

    // 寄存器中的4x4累加器
    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    // 指向当前用于计算的共享内存缓冲区
    float (*Asub_curr)[4 * BLOCK_SIZE] = Asub_ping;
    float (*Bsub_curr)[4 * BLOCK_SIZE] = Bsub_ping;
    float (*Asub_next)[4 * BLOCK_SIZE] = Asub_pong;
    float (*Bsub_next)[4 * BLOCK_SIZE] = Bsub_pong;

    // 预取第一个K维tile到ping缓冲区（向量化加载）
    {
        // --- A: 每线程加载一个float4，覆盖某一行在K维的4个连续元素 ---
        // 将tid_x划分为(行选择, K向量段选择)：tid_x / 4 对应本线程加载的行索引(相对于其4行组)，tid_x % 4 对应K子块内的段(0,4,8,12)
        int a_row_sel = tid_x / 4;      // 0..3，选择本线程的四行中的哪一行
        int a_vec_seg = tid_x % 4;      // 0..3，对应K内的4元素段
        int a_local_row = tid_y * 4 + a_row_sel; // 0..63
        int g_row = tile_row_base + a_local_row;
        int local_k_base = a_vec_seg * 4; // 0,4,8,12

        float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (g_row < N) {
            int gk0 = 0 + local_k_base + 0;
            int gk1 = 0 + local_k_base + 1;
            int gk2 = 0 + local_k_base + 2;
            int gk3 = 0 + local_k_base + 3;
            const float* a_ptr = &A[g_row * N + gk0];
            uintptr_t aline = reinterpret_cast<uintptr_t>(a_ptr);
            if (gk3 < N && (aline % 16 == 0)) {
                a4 = *reinterpret_cast<const float4*>(a_ptr);
            } else {
                a4.x = (gk0 < N) ? A[g_row * N + gk0] : 0.0f;
                a4.y = (gk1 < N) ? A[g_row * N + gk1] : 0.0f;
                a4.z = (gk2 < N) ? A[g_row * N + gk2] : 0.0f;
                a4.w = (gk3 < N) ? A[g_row * N + gk3] : 0.0f;
            }
        }

        Asub_curr[local_k_base + 0][a_local_row] = a4.x;
        Asub_curr[local_k_base + 1][a_local_row] = a4.y;
        Asub_curr[local_k_base + 2][a_local_row] = a4.z;
        Asub_curr[local_k_base + 3][a_local_row] = a4.w;

        // --- B: 每线程加载一个float4，覆盖同一K行上的4个连续列 ---
        int kB = 0 + tid_y;
        float4 b4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (kB < N) {
            const float* b_ptr = &B[kB * N + c0];
            uintptr_t bline = reinterpret_cast<uintptr_t>(b_ptr);
            if (c3 < N && (bline % 16 == 0)) {
                b4 = *reinterpret_cast<const float4*>(b_ptr);
            } else {
                b4.x = (c0 < N) ? B[kB * N + c0] : 0.0f;
                b4.y = (c1 < N) ? B[kB * N + c1] : 0.0f;
                b4.z = (c2 < N) ? B[kB * N + c2] : 0.0f;
                b4.w = (c3 < N) ? B[kB * N + c3] : 0.0f;
            }
        }

        Bsub_curr[tid_y][b_c0_local] = b4.x;
        Bsub_curr[tid_y][b_c1_local] = b4.y;
        Bsub_curr[tid_y][b_c2_local] = b4.z;
        Bsub_curr[tid_y][b_c3_local] = b4.w;
    }

    __syncthreads();

    // K维遍历，以BLOCK_SIZE为步长（软件流水+双缓冲）
    for (int k_tile = 0; k_tile < N; k_tile += BLOCK_SIZE) {

        // 内层k循环，UNROLL_FACTOR=4（重排FMA以提升指令级并行度）
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            // k + 0
            {
                float a0 = Asub_curr[k + 0][a_r0_local];
                float a1 = Asub_curr[k + 0][a_r1_local];
                float a2 = Asub_curr[k + 0][a_r2_local];
                float a3 = Asub_curr[k + 0][a_r3_local];

                float b0 = Bsub_curr[k + 0][b_c0_local];
                float b1 = Bsub_curr[k + 0][b_c1_local];
                float b2 = Bsub_curr[k + 0][b_c2_local];
                float b3 = Bsub_curr[k + 0][b_c3_local];

                acc00 += a0 * b0;
                acc11 += a1 * b1;
                acc22 += a2 * b2;
                acc33 += a3 * b3;

                acc01 += a0 * b1;
                acc12 += a1 * b2;
                acc23 += a2 * b3;
                acc30 += a3 * b0;

                acc02 += a0 * b2;
                acc13 += a1 * b3;
                acc20 += a2 * b0;
                acc31 += a3 * b1;

                acc03 += a0 * b3;
                acc10 += a1 * b0;
                acc21 += a2 * b1;
                acc32 += a3 * b2;
            }

            // k + 1
            {
                float a0 = Asub_curr[k + 1][a_r0_local];
                float a1 = Asub_curr[k + 1][a_r1_local];
                float a2 = Asub_curr[k + 1][a_r2_local];
                float a3 = Asub_curr[k + 1][a_r3_local];

                float b0 = Bsub_curr[k + 1][b_c0_local];
                float b1 = Bsub_curr[k + 1][b_c1_local];
                float b2 = Bsub_curr[k + 1][b_c2_local];
                float b3 = Bsub_curr[k + 1][b_c3_local];

                acc00 += a0 * b0;
                acc11 += a1 * b1;
                acc22 += a2 * b2;
                acc33 += a3 * b3;

                acc01 += a0 * b1;
                acc12 += a1 * b2;
                acc23 += a2 * b3;
                acc30 += a3 * b0;

                acc02 += a0 * b2;
                acc13 += a1 * b3;
                acc20 += a2 * b0;
                acc31 += a3 * b1;

                acc03 += a0 * b3;
                acc10 += a1 * b0;
                acc21 += a2 * b1;
                acc32 += a3 * b2;
            }

            // k + 2
            {
                float a0 = Asub_curr[k + 2][a_r0_local];
                float a1 = Asub_curr[k + 2][a_r1_local];
                float a2 = Asub_curr[k + 2][a_r2_local];
                float a3 = Asub_curr[k + 2][a_r3_local];

                float b0 = Bsub_curr[k + 2][b_c0_local];
                float b1 = Bsub_curr[k + 2][b_c1_local];
                float b2 = Bsub_curr[k + 2][b_c2_local];
                float b3 = Bsub_curr[k + 2][b_c3_local];

                acc00 += a0 * b0;
                acc11 += a1 * b1;
                acc22 += a2 * b2;
                acc33 += a3 * b3;

                acc01 += a0 * b1;
                acc12 += a1 * b2;
                acc23 += a2 * b3;
                acc30 += a3 * b0;

                acc02 += a0 * b2;
                acc13 += a1 * b3;
                acc20 += a2 * b0;
                acc31 += a3 * b1;

                acc03 += a0 * b3;
                acc10 += a1 * b0;
                acc21 += a2 * b1;
                acc32 += a3 * b2;
            }

            // k + 3
            {
                float a0 = Asub_curr[k + 3][a_r0_local];
                float a1 = Asub_curr[k + 3][a_r1_local];
                float a2 = Asub_curr[k + 3][a_r2_local];
                float a3 = Asub_curr[k + 3][a_r3_local];

                float b0 = Bsub_curr[k + 3][b_c0_local];
                float b1 = Bsub_curr[k + 3][b_c1_local];
                float b2 = Bsub_curr[k + 3][b_c2_local];
                float b3 = Bsub_curr[k + 3][b_c3_local];

                acc00 += a0 * b0;
                acc11 += a1 * b1;
                acc22 += a2 * b2;
                acc33 += a3 * b3;

                acc01 += a0 * b1;
                acc12 += a1 * b2;
                acc23 += a2 * b3;
                acc30 += a3 * b0;

                acc02 += a0 * b2;
                acc13 += a1 * b3;
                acc20 += a2 * b0;
                acc31 += a3 * b1;

                acc03 += a0 * b3;
                acc10 += a1 * b0;
                acc21 += a2 * b1;
                acc32 += a3 * b2;
            }
        }

        // 预取下一个K维tile到next缓冲区（软件流水，向量化加载），只有在存在下一tile时进行
        if (k_tile + BLOCK_SIZE < N) {
            // A next tile
            int a_row_sel = tid_x / 4;          // 0..3
            int a_vec_seg = tid_x % 4;          // 0..3
            int a_local_row = tid_y * 4 + a_row_sel; // 0..63
            int g_row = tile_row_base + a_local_row;
            int local_k_base = a_vec_seg * 4;

            float4 a4n = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < N) {
                int gk0 = k_tile + BLOCK_SIZE + local_k_base + 0;
                int gk1 = k_tile + BLOCK_SIZE + local_k_base + 1;
                int gk2 = k_tile + BLOCK_SIZE + local_k_base + 2;
                int gk3 = k_tile + BLOCK_SIZE + local_k_base + 3;
                const float* a_ptr_n = &A[g_row * N + gk0];
                uintptr_t aline_n = reinterpret_cast<uintptr_t>(a_ptr_n);
                if (gk3 < N && (aline_n % 16 == 0)) {
                    a4n = *reinterpret_cast<const float4*>(a_ptr_n);
                } else {
                    a4n.x = (gk0 < N) ? A[g_row * N + gk0] : 0.0f;
                    a4n.y = (gk1 < N) ? A[g_row * N + gk1] : 0.0f;
                    a4n.z = (gk2 < N) ? A[g_row * N + gk2] : 0.0f;
                    a4n.w = (gk3 < N) ? A[g_row * N + gk3] : 0.0f;
                }
            }

            Asub_next[local_k_base + 0][a_local_row] = a4n.x;
            Asub_next[local_k_base + 1][a_local_row] = a4n.y;
            Asub_next[local_k_base + 2][a_local_row] = a4n.z;
            Asub_next[local_k_base + 3][a_local_row] = a4n.w;

            // B next tile
            int next_kB = k_tile + BLOCK_SIZE + tid_y;
            float4 b4n = make_float4(0.f, 0.f, 0.f, 0.f);
            if (next_kB < N) {
                const float* b_ptr_n = &B[next_kB * N + c0];
                uintptr_t bline_n = reinterpret_cast<uintptr_t>(b_ptr_n);
                if (c3 < N && (bline_n % 16 == 0)) {
                    b4n = *reinterpret_cast<const float4*>(b_ptr_n);
                } else {
                    b4n.x = (c0 < N) ? B[next_kB * N + c0] : 0.0f;
                    b4n.y = (c1 < N) ? B[next_kB * N + c1] : 0.0f;
                    b4n.z = (c2 < N) ? B[next_kB * N + c2] : 0.0f;
                    b4n.w = (c3 < N) ? B[next_kB * N + c3] : 0.0f;
                }
            }

            Bsub_next[tid_y][b_c0_local] = b4n.x;
            Bsub_next[tid_y][b_c1_local] = b4n.y;
            Bsub_next[tid_y][b_c2_local] = b4n.z;
            Bsub_next[tid_y][b_c3_local] = b4n.w;
        }

        // 等待所有线程完成当前计算和下一tile的加载
        __syncthreads();

        // 交换缓冲区（仅当确实预取了下一tile时）
        if (k_tile + BLOCK_SIZE < N) {
            float (*tmpA)[4 * BLOCK_SIZE] = Asub_curr;
            Asub_curr = Asub_next;
            Asub_next = tmpA;

            float (*tmpB)[4 * BLOCK_SIZE] = Bsub_curr;
            Bsub_curr = Bsub_next;
            Bsub_next = tmpB;
        }
    }

    // 写回最终结果（4x4输出，需边界检查）
    if (r0 < N && c0 < N) C[r0 * N + c0] = acc00;
    if (r0 < N && c1 < N) C[r0 * N + c1] = acc01;
    if (r0 < N && c2 < N) C[r0 * N + c2] = acc02;
    if (r0 < N && c3 < N) C[r0 * N + c3] = acc03;

    if (r1 < N && c0 < N) C[r1 * N + c0] = acc10;
    if (r1 < N && c1 < N) C[r1 * N + c1] = acc11;
    if (r1 < N && c2 < N) C[r1 * N + c2] = acc12;
    if (r1 < N && c3 < N) C[r1 * N + c3] = acc13;

    if (r2 < N && c0 < N) C[r2 * N + c0] = acc20;
    if (r2 < N && c1 < N) C[r2 * N + c1] = acc21;
    if (r2 < N && c2 < N) C[r2 * N + c2] = acc22;
    if (r2 < N && c3 < N) C[r2 * N + c3] = acc23;

    if (r3 < N && c0 < N) C[r3 * N + c0] = acc30;
    if (r3 < N && c1 < N) C[r3 * N + c1] = acc31;
    if (r3 < N && c2 < N) C[r3 * N + c2] = acc32;
    if (r3 < N && c3 < N) C[r3 * N + c3] = acc33;
}

// ------------------------------------------------------------------
// WRAPPER: gemm_cuda (这是PyTorch和CUDA之间的桥梁)
// ------------------------------------------------------------------
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    
    // --- 输入验证 ---
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(M == N && K == N, "This naive example assumes square N=M=K matrices");
    auto C = torch::zeros({M, N}, A.options());

    // --- 内核启动配置 ---
    const int block_dim_x = BLOCK_SIZE;
    const int block_dim_y = BLOCK_SIZE;
    const int grid_dim_x = (N + block_dim_x - 1) / block_dim_x;
    const int grid_dim_y = (N + block_dim_y - 1) / block_dim_y;
    dim3 blocks(grid_dim_x, grid_dim_y);
    dim3 threads(block_dim_x, block_dim_y);

    // --- 启动内核 ---
    gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // --- 错误检查 ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in gemm_kernel: " + std::string(cudaGetErrorString(err)));
    }
    return C;
}