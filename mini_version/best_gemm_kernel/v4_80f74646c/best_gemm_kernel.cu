#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// ------------------------------------------------------------------
// KERNEL: gemm_kernel 
// ------------------------------------------------------------------
__global__ void gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    // 使用双缓冲共享内存的分块 (tiled) CUDA 矩阵乘法 (GEMM) 内核

    // 1. 定义双缓冲共享内存块
    __shared__ float Asub[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[2][BLOCK_SIZE][BLOCK_SIZE];

    // 2. 线程在块内的位置以及块在矩阵中的起始位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowBlock = blockIdx.y * BLOCK_SIZE;
    int colBlock = blockIdx.x * BLOCK_SIZE;

    // 全局行/列索引（用于最终写回）
    int row = rowBlock + ty;
    int col = colBlock + tx;

    // 3. 初始化累加器
    float sum = 0.0f;

    // 4. 计算需要的tile数量（覆盖K维）
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 如果没有tile，直接写0并返回（不过对于N>0这一分支通常不会触发）
    if (numTiles == 0) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    // 5. 初始化缓冲区索引：compute_buffer 持有当前用于计算的tile（预加载），load_buffer 用于加载下一个tile
    int load_buffer = 1;
    int compute_buffer = 0;

    // 6. 预加载第一个tile到 compute_buffer，并同步以确保可见
    {
        int t0 = 0;
        int aCol0 = t0 * BLOCK_SIZE + tx; // A 的列
        int bRow0 = t0 * BLOCK_SIZE + ty; // B 的行
        int aRow = rowBlock + ty;         // A 的行（全局）
        int bCol = colBlock + tx;         // B 的列（全局）

        Asub[compute_buffer][ty][tx] = (aRow < N && aCol0 < N) ? A[aRow * N + aCol0] : 0.0f;
        Bsub[compute_buffer][ty][tx] = (bRow0 < N && bCol < N) ? B[bRow0 * N + bCol] : 0.0f;
    }
    __syncthreads();

    // 7. 主循环：对于每个tile，先发起对下一个tile的加载（如果存在），再对当前compute_buffer执行计算
    for (int t = 0; t < numTiles; ++t) {
        // 发起对下一个tile的加载（写入 load_buffer），但不在这里同步，以便和当前计算重叠
        if (t < numTiles - 1) {
            int next = t + 1;
            int aCol_n = next * BLOCK_SIZE + tx;
            int bRow_n = next * BLOCK_SIZE + ty;
            int aRow = rowBlock + ty;
            int bCol = colBlock + tx;

            Asub[load_buffer][ty][tx] = (aRow < N && aCol_n < N) ? A[aRow * N + aCol_n] : 0.0f;
            Bsub[load_buffer][ty][tx] = (bRow_n < N && bCol < N) ? B[bRow_n * N + bCol] : 0.0f;
        }

        // 对当前compute_buffer执行计算
        // 使用单次连续遍历 k 方向（0..BLOCK_SIZE-1），每次处理 4 个元素（手工展开）
        // 以减少循环开销并直接将乘加累到主累加器 sum 中
        {
            int k = 0;
            // 主循环：每次处理 4 个元素
            for (; k + 3 < BLOCK_SIZE; k += 4) {
                // 限定作用域以便寄存器复用最小化活跃变量数
                {
                    float a0 = Asub[compute_buffer][ty][k + 0];
                    float a1 = Asub[compute_buffer][ty][k + 1];
                    float a2 = Asub[compute_buffer][ty][k + 2];
                    float a3 = Asub[compute_buffer][ty][k + 3];

                    float b0 = Bsub[compute_buffer][k + 0][tx];
                    float b1 = Bsub[compute_buffer][k + 1][tx];
                    float b2 = Bsub[compute_buffer][k + 2][tx];
                    float b3 = Bsub[compute_buffer][k + 3][tx];

                    sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                }
            }
            // 处理剩余（如果 BLOCK_SIZE 不是 4 的倍数）
            for (; k < BLOCK_SIZE; ++k) {
                float a = Asub[compute_buffer][ty][k];
                float b = Bsub[compute_buffer][k][tx];
                sum += a * b;
            }
        }

        // 在交换缓冲区之前需要同步，确保下一次迭代中读取的缓冲区已经被完整加载
        __syncthreads();

        // 交换缓冲区索引，为下一次迭代准备
        int temp = load_buffer;
        load_buffer = compute_buffer;
        compute_buffer = temp;
    }

    // 8. 将结果写回全局内存（带边界检查）
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
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