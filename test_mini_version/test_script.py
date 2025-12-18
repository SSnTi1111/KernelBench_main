import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import torch 
import re
import tempfile 
import importlib.util 
import contextlib
import traceback     
import inspect # [!!! 新增 !!!] 用于从 .py 文件中抓取源代码
import signal # [!!! 新增 !!!]
import gc
import io
import warnings
import copy

data_type_info = ""
# --- 1. 设置项目路径 ---
KERNELBENCH_PATH = "/home/lxt/KernelBench/KernelBench_main"
MINI_VERSION_PATH = "/home/lxt/KernelBench/KernelBench_main/mini_version"

# [!!! 已更新 !!!] 
# 我们只添加 mini_version 
# 路径，不再需要 KERNELBENCH_PATH 
# (因为我们不导入 src)
if MINI_VERSION_PATH not in sys.path:
    sys.path.append(MINI_VERSION_PATH)

# --- 2. 导入依赖的项目模块 ---
try:
    # [!!! 已移除 !!!] 
    # 不再导入 KernelBench (kb_eval, kb_utils, Problem)
    pass
except ImportError as e:
    pass

try:
    # 导入 mini_version 模块 (已重构)
    import config as mv_config
    import llm_api as mv_llm_api
    import prompts as mv_prompts
    import main as mv_main 
    import cuda_utils as mv_cuda_utils 
except ImportError as e:
    print(f"Error: 无法从 {MINI_VERSION_PATH} 导入 mini_version 模块。")
    print(f"ImportError: {e}")
    sys.exit(1)



def get_pytorch_to_cuda_prompt(pytorch_code, inputs, ref_outputs):
    # input_specs = []
    # input_params = []
    

    # for i, item in enumerate(inputs):
    #     if isinstance(item, torch.Tensor):
    #         input_specs.append(f"  Input {i} (arg{i}): shape={item.shape}, dtype={item.dtype}")
    #         input_params.append(f"torch::Tensor arg{i}")
    #     elif isinstance(item, float):
    #         input_specs.append(f"  Input {i} (arg{i}): type=float, value={item}")
    #         input_params.append(f"double arg{i}") # C++ 中 PyTorch float 对应 double
    #     elif isinstance(item, int):
    #         input_specs.append(f"  Input {i} (arg{i}): type=int, value={item}")
    #         input_params.append(f"int64_t arg{i}") # C++ 中 PyTorch int 对应 int64_t
    #     else:
    #         print(f"Warning: get_pytorch_to_cuda_prompt 中未知的输入类型: {type(item)}")

    # outputs_list = ref_outputs if isinstance(ref_outputs, (list, tuple)) else [ref_outputs]
    # output_specs = []
    # output_return_type = "torch::Tensor" # 默认为单个输出
    # if len(outputs_list) > 1:
    #     output_return_type = "std::vector<torch::Tensor>"
        
    # for i, tensor in enumerate(outputs_list):
    #     output_specs.append(f"  Output {i}: shape={tensor.shape}, dtype={tensor.dtype}")

    # input_specs_str = "\n".join(input_specs)
    # output_specs_str = "\n".join(output_specs)
    # input_params_str = ", ".join(input_params)

    prompt = """
Task
----
Generate **hand‑written CUDA kernels** that replace *all* PyTorch operator(s)
inside the original `class Model` (shown later).  You may fuse multiple
operators into a single kernel if that yields better performance.  Leave any
non‑replaced parts of the model unchanged.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple‑quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.
3. '--ptxas-options=-v'option must be added
4. 'verbose=True' option must be added

Few‑shot example (reference only – do **not** echo):
**Original**
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, Q, K, V):
        att = (Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(K.size(-1))))
        att = F.softmax(att, dim=-1)
        y = att @ V
        return y

batch_size = 32
n_head = 12
seq_len = 64
head_embd = 32

def get_inputs():
    # randomly generate input tensors based on the model architecture
    Q = torch.randn(batch_size, n_head, seq_len, head_embd)
    K = torch.randn(batch_size, n_head, seq_len, head_embd)
    V = torch.randn(batch_size, n_head, seq_len, head_embd)
    return [Q, K, V]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
```
**Converted CUDA Version** 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

source = '''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

__global__
void attention_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    attention_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}
'''
cpp_src = '''torch::Tensor attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);'''

attention = torch.utils.cpp_extension.load_inline(
    'attention',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['attention'],
    with_cuda=True,
    verbose=True,
    extra_cuda_cflags=['-O2','--ptxas-options=-v'],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, Q, K, V):
        return self.attention.attention(Q, K, V)

```
""" + f"""
Currently, the complete pytorch code you need to convert is:
```python
{pytorch_code}
```
You do not need get_inputs and get_init_inputs functions in the generated code. After that, I will directly use the data obtained by get_init_inputs in pytorch to create your generated modelnew. Therefore, you need to ensure that the same parameters are directly received in the __init__ function. At the same time, I will also use the input obtained from get_inputs in pytorch directly for the calculation of modelnew, so it is necessary to ensure that the forward generated by you can be directly received.
Your output format should be:
### FINAL_CUDA_CODE_START
```python
[Complete CUDA code]
```
### FINAL_CUDA_CODE_END

For the generated CUDA code, please pay attention to the following points:
    1. Do not use statements such as C10_CUDA_KERNEL_LAUNCH_CHECK, which have different availability between different versions！！！！. The current valid environment is Python version 3.10.12
        Torch version: 2.4.0
        Triton version: 3.0.0
        CUDA version: 12.4
    2. Boundary checks:
        Guard all memory access with boundary conditions (e.g., if (i < size)). 
    3. Reductions and shared memory:
        Use dynamic shared memory (extern __shared__). Implement 3-phase reduction (local → shared → global), with __syncthreads() for synchronization.
    4. Synchronization:
        Synchronize after shared writes and within reductions. Cross-block communication must use global memory.
    5. Error handling:
        Check cudaGetLastError() after launch. Use TORCH_CHECK for input tensor validation.
    6. Boolean masks:
        Use conditional logic or ternary operations. Never use float casts or comparisons for boolean masks.
    7. Do not use C++ macros (such as TORCH_CHECK) in the Python layer.
        All TORCH_CHECK-based type checking logic must appear only in the C++ layer.
"""
    return prompt

def final_extract(content):
    # 方法1：使用特殊标记提取（推荐）
    def extract_final_code_method1(content):
        start_marker = "### FINAL_CUDA_CODE_START"
        end_marker = "### FINAL_CUDA_CODE_END"
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return None
            
        # 找到起始标记后的第一个```python
        start_idx = content.find("```python", start_idx)
        if start_idx == -1:
            start_idx = content.find("```", start_idx)
            if start_idx == -1:
                return None
            start_idx += 3
        else:
            start_idx += 9  # len("```cpp")
        
        # 找到结束标记前的最后一个```
        end_idx = content.find(end_marker)
        if end_idx == -1:
            return None
            
        # 在结束标记前查找```
        temp_content = content[start_idx:end_idx]
        code_end = temp_content.rfind("```")
        if code_end == -1:
            code_end = temp_content.rfind('"""')
        if code_end == -1:
            return None
            
        return temp_content[:code_end].strip()
    
    extracted_code = extract_final_code_method1(content)
    if extracted_code:
        return extracted_code
    else:
        print("Warning: Could not extract code properly. Returning full response for debugging.")
        return content

def generate_initial_cuda_kernel(pytorch_code, inputs, ref_outputs):
    """
    调用 LLM API 来生成初始的 C++ Wrapper 和 CUDA 内核代码。
    """
    prompt = get_pytorch_to_cuda_prompt(pytorch_code, inputs, ref_outputs) 
    system_prompt = "You are a professional CUDA programmer who is good at converting pytorch to CUDA c++."
    
    try:
        response_text = mv_llm_api.call_llm(
            agent_name="initial_generator", 
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        if not response_text:
            raise Exception("LLM 响应为空")

        cuda_code = final_extract(response_text)

        return cuda_code
        
    except Exception as e:
        print(f"Error during initial kernel generation: {e}")
        return None, None, str(e)
def get_cuda_correction_prompt(pytorch_code, cuda_code, errMessage):
    if errMessage['exeErr'] != "":
        modifyCudaCodesWithFeedbackPrompt = f'''
        You are a professional CUDA engineer. A student has been assigned the task of converting a Pytorch kernel into CUDA code, ensuring that the CUDA code has the same inputs, outputs, and functionality as the original Pytorch code. I will provide you with both the original Pytorch code and the student's CUDA code.
        The original Pytorch code:
        ```python
        {pytorch_code}
        ```
        The student's CUDA code:
        ```python
        {cuda_code}
        ```
        There is an execution issue when verifying the generated CUDA code:
        > {errMessage['exeErr']}
        Remove all assert operations!!!
        Please modify the original CUDA code according to the error message.
        Only the error-reporting part needs to be modified, and the remaining parts should remain unchanged as much as possible. The final output should be the complete CUDA code.
        ```
        Your output format should be:
        ### FINAL_CUDA_CODE_START
        ```python
        [Complete CUDA code here]
        ```
        ### FINAL_CUDA_CODE_END
        '--ptxas-options=-v'option must be added
        'verbose=True' option must be added
        '''
    else:
        modifyCudaCodesWithFeedbackPrompt = f'''
        You are a professional CUDA engineer. A student has been assigned the task of converting a Pytorch kernel into CUDA code, ensuring that the CUDA code has the same inputs, outputs, and functionality as the original Pytorch code. I will provide you with both the original Pytorch code and the student's CUDA code.
        The original Pytorch code:
        ```python
        {pytorch_code}
        ```
        The student's CUDA code:
        ```python
        {cuda_code}
        ```
        There is an IO issue when verifying the generated CUDA code:
        > {errMessage['ioErr']}
        Remove all assert operations!!!
        Please modify the original CUDA code according to the error message.
        During the modification process, it is essential to ensure that the code format remains unchanged, and only make functional modifications based on the input and output.
        ```
        Your output format should be:
        ### FINAL_CUDA_CODE_START
        ```python
        [Complete CUDA code here]
        ```
        ### FINAL_CUDA_CODE_END
        '--ptxas-options=-v'option must be added
        'verbose=True' option must be added
        '''
    
    # print(modifyCudaCodesWithFeedbackPrompt)    
    return modifyCudaCodesWithFeedbackPrompt

def correct_cuda_kernel(pytorch_code, cuda_code, error_message): 
    prompt = get_cuda_correction_prompt(pytorch_code, cuda_code, error_message) 
    system_prompt = "You are a professional CUDA programmer, good at debugging c++extension compilation and runtime errors."
    try:
        response_text = mv_llm_api.call_llm(
            agent_name="planner", 
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        if not response_text:
            raise Exception("LLM 响应为空")

        cuda_code = final_extract(response_text)

        return cuda_code

    except Exception as e:
        print(f"Error during kernel correction: {e}")
        return None, None, str(e)

def benchmark_torch_model(model, gpu_inputs, warmup_runs=10, benchmark_runs=50):
    # 预热
    for _ in range(warmup_runs):
        _ = model(*gpu_inputs)
    torch.cuda.synchronize()

    # 测量
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(benchmark_runs):
        _ = model(*gpu_inputs)
    end.record()

    torch.cuda.synchronize()
    avg_time_ms = start.elapsed_time(end) / benchmark_runs
    return avg_time_ms

def load_problem_module_from_file(problem_name: str, file_path: str):
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"Problem file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(problem_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {file_path}")
        
    problem_module = importlib.util.module_from_spec(spec)
    # 将此模块添加到 sys.modules，以防它内部有导入
    sys.modules[problem_name] = problem_module 
    spec.loader.exec_module(problem_module)

    return problem_module

# [!!! 新增 1 !!!]
# 定义超时异常和处理函数
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    # (7200秒)
    raise TimeoutException("该测试用例处理超时 (超过 2 小时)")


def move_to_cuda(item):
    if isinstance(item, torch.Tensor):
        return item.cuda()
    elif isinstance(item, (list, tuple)):
        # 递归处理列表或元组，并保持原有类型
        return type(item)(move_to_cuda(x) for x in item)
    elif isinstance(item, dict):
        return {k: move_to_cuda(v) for k, v in item.items()}
    else:
        return item

import subprocess
def extract_error_and_next_line(text):
    # 按行分割
    lines = text.splitlines()
    results = []
    for i, line in enumerate(lines):
        if "error:" in line:
            results.append(line)
            if i + 1 < len(lines):
                results.append(lines[i + 1])
    return "\n".join(results)

# def validate_extracted_code(cuda_code, init_inputs, test_inputs=None, test_outputs=None):

#     TEST_NN_MODEL_NAME = 'ModelNew'

#     try:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             temp_file = os.path.join(temp_dir, "cuda_code.py")
#             with open(temp_file, "w") as f:
#                 f.write(cuda_code)

#             spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, temp_file)
#             if spec is None:
#                 return False, {"exeErr": "Error: Failed to get module spec.", "ioErr": ""}

#             module = importlib.util.module_from_spec(spec)
#             sys.modules[TEST_NN_MODEL_NAME] = module
#             # ---------- 执行模块 & 捕获所有输出 ----------
#             try:
#                 # 优先使用 exec_module（能更快给出语法错误）
#                 spec.loader.exec_module(module)
#             except Exception:
#                 # exec_module 失败后，再用 subprocess 捕获详细错误
#                 proc = subprocess.run(
#                     [sys.executable, temp_file],
#                     capture_output=True,
#                     text=True
#                 )

#                 stderr = proc.stderr.strip()
#                 stdout = proc.stdout.strip()

#                 err_msg = ""
#                 if stdout:
#                     err_msg += "\n[stdout]\n" + stdout
#                 if stderr:
#                     err_msg += "\n[stderr]\n" + stderr

#                 if proc.returncode != 0:
#                     return False, {
#                         "exeErr": f"Execution failed with return code {proc.returncode}.{extract_error_and_next_line(err_msg)}",
#                         "ioErr": ""
#                     }


#             # try:
#             #     # 创建 StringIO 对象来捕获 stdout 和 stderr
#             #     stdout_capture = io.StringIO()
#             #     stderr_capture = io.StringIO()

#             #     # 重定向 stdout 和 stderr
#             #     with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
#             #         # 尝试执行模块
#             #         try:
#             #             spec.loader.exec_module(module)
#             #         except Exception as e:
#             #             # 如果直接执行失败，尝试通过 subprocess 运行以捕获更多输出
#             #             try:
#             #                 result = subprocess.run(
#             #                     [sys.executable, temp_file],
#             #                     capture_output=True,
#             #                     text=True,
#             #                     check=True
#             #                 )
#             #                 # 如果 subprocess 成功运行，附加其输出
#             #                 stdout_capture.write(result.stdout or "")
#             #                 stderr_capture.write(result.stderr or "")
#             #             except subprocess.CalledProcessError as sub_e:
#             #                 # 如果 subprocess 失败，捕获其输出
#             #                 stdout_capture.write(sub_e.stdout or "")
#             #                 stderr_capture.write(sub_e.stderr or "")
#             #                 raise sub_e  # 继续抛出异常以进入外层 except

#             #     # 如果没有异常，清理捕获的输出（根据需求可保留）
#             #     captured_stdout = stdout_capture.getvalue().strip()
#             #     captured_stderr = stderr_capture.getvalue().strip()
#             #     additional_output = ""
#             #     if captured_stdout:
#             #         additional_output += f"\nCaptured stdout:\n{captured_stdout}"
#             #     if captured_stderr:
#             #         additional_output += f"\nCaptured stderr:\n{captured_stderr}"

#             # except Exception as e:
#             #     # 捕获异常并附加所有输出
#             #     captured_stdout = stdout_capture.getvalue().strip()
#             #     captured_stderr = stderr_capture.getvalue().strip()
#             #     additional_output = ""
#             #     if captured_stdout:
#             #         additional_output += f"\nCaptured stdout:\n{captured_stdout}"
#             #     if captured_stderr:
#             #         additional_output += f"\nCaptured stderr:\n{captured_stderr}"
#             #     additional_output = extract_error_and_next_line(additional_output)
#             #     return False, {
#             #         "exeErr": f"Error: Failed to execute module: {str(e)}{additional_output}",
#             #         "ioErr": ""
#             #     }

#             # 检查 ModelNew 类是否存在
#             model_class = getattr(module, TEST_NN_MODEL_NAME, None)
#             if model_class is None:
#                 return False, {"exeErr": f"Error: Class '{TEST_NN_MODEL_NAME}' not found in the module.", "ioErr": ""}

#             # 实例化模型
#             try:
#                 if init_inputs is not None:
#                     if isinstance(init_inputs, (list, tuple)):
#                         model_instance = model_class(*init_inputs)
#                     elif isinstance(init_inputs, dict):
#                         model_instance = model_class(**init_inputs)
#                     else:
#                         # 单值初始化
#                         model_instance = model_class(init_inputs)
#                 else:
#                     # 无初始化参数
#                     model_instance = model_class()
#             except Exception as e:
#                 return False, {"exeErr": f"Error: Failed to instantiate '{TEST_NN_MODEL_NAME}': {str(e)}", "ioErr": ""}

#             # 检查 forward 方法是否存在
#             if not hasattr(model_instance, 'forward'):
#                 return False, {"exeErr": f"Error: '{TEST_NN_MODEL_NAME}' does not have a 'forward' method.", "ioErr": ""}
#             if test_inputs is not None:
#                 # print("正在进行validate_extracted_code中的IO验证……")
#                 # if not isinstance(test_inputs, (list, tuple)): model_output = (test_inputs,)
#                 # if not isinstance(test_outputs, (list, tuple)): test_outputs = (test_outputs,)
#                 # for i, (real_input, real_output) in enumerate(zip(test_inputs, test_outputs)):
#                 output = model_instance(*test_inputs)  # CUDA 的输出
                
#                 def compare_outputs(a, b, atol=1e-2, rtol=1e-2):
#                     global data_type_info
#                     # tuple 情况
#                     if isinstance(a, tuple) and isinstance(b, tuple):
#                         if len(a) != len(b):
#                             return False
#                         return all(compare_outputs(x, y, atol, rtol) for x, y in zip(a, b))

#                     # tensor 对 tensor
#                     if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
#                         return torch.allclose(a, b, atol=atol, rtol=rtol)

#                     # # 标量对标量
#                     if isinstance(a, (int, float)) and isinstance(b, (int, float)):
#                         return abs(a - b) <= (atol + rtol * abs(b))

#                     print("输出类型不匹配：", type(a), type(b))
#                     data_type_info = f"The value type of some values in the return value is incorrect. The current value type is {type(b)} and the correct value type is f{type(a)}"
#                     return False


#                 if compare_outputs(output, test_outputs, atol=1e-2, rtol=1e-2):
#                     global data_type_info
#                     data_type_info = ""
#                     print(f"Validate_extracted_code:Test passed")
#                 else:
#                     print(f"Test failed: output mismatch")

#                     if not data_type_info:

#                         # --- [核心修改] 捕获前 5 个错误值 ---
#                         diff = torch.abs(output - test_outputs)
#                         # 计算允许的误差范围
#                         tol = 1e-2 + 1e-2 * torch.abs(test_outputs)
#                         # 找出超出误差的掩码
#                         error_mask = diff > tol
#                         # 获取错误索引
#                         error_indices = torch.nonzero(error_mask, as_tuple=False)
#                         num_errors = error_indices.size(0)
                        
#                         msg_header = f"Failed (Correctness): Output has {num_errors} mismatches (total elements: {test_outputs.numel()})."
#                         error_details = [msg_header]
#                         error_details.append("Top 5 Mismatches (Index | Reference Value | Actual Value):")
                        
#                         # 取前 5 个
#                         for j in range(min(5, num_errors)):
#                             idx = error_indices[j]
#                             idx_tuple = tuple(idx.tolist())
#                             ref_val = test_outputs[idx_tuple].item()
#                             act_val = output[idx_tuple].item()
#                             error_details.append(f"  [{j}] Index: {idx_tuple} | Ref: {ref_val:.6f} | Act: {act_val:.6f}")
                        
#                         full_msg = "\n".join(error_details)

#                         return False, {
#                             "exeErr": "",
#                             "ioErr": (
#                                 # f"{data_type_info}"
#                                 "For the current test input, the output generated by the CUDA code cannot match "
#                                 "the correct output, with a certain margin of error allowed, that is, satisfying "
#                                 "torch.allclose(cuda_output, pytorch_output, atol=1e-3, rtol=1e-3). "
#                                 # f"The current input is:{test_inputs} "
#                                 # f"The output of the CUDA code is:{output} "
#                                 # f"The output of the Triton code is:{test_outputs}"
#                                 f"{full_msg}"
#                             )
#                         }
#                     else:
#                         return False, {
#                             "exeErr": "",
#                             "ioErr": (
#                                 f"{data_type_info}"
#                                 # "For the current test input, the output generated by the CUDA code cannot match "
#                                 # "the correct output, with a certain margin of error allowed, that is, satisfying "
#                                 # "torch.allclose(triton_output, cuda_output, atol=1e-3, rtol=1e-3). "
#                                 # f"The current input is:{test_inputs} "
#                                 # f"The output of the CUDA code is:{output} "
#                                 # f"The output of the Triton code is:{test_outputs}"
#                             )
#                         }
#         return True, None
#     except Exception as e:
#         return False, {
#             "exeErr": f"Error during validation: {str(e)}",
#             "ioErr": ""
#         }

def validate_extracted_code(cuda_code, init_inputs, test_inputs=None, test_outputs=None):
    TEST_NN_MODEL_NAME = 'ModelNew'
    
    # 定义需要在 finally 中清理的变量名，防止 UnboundLocalError
    module = None
    model_instance = None
    output = None
    cloned_inputs = None
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "cuda_code.py")
            with open(temp_file, "w") as f:
                f.write(cuda_code)

            spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, temp_file)
            if spec is None:
                return False, {"exeErr": "Error: Failed to get module spec.", "ioErr": ""}

            module = importlib.util.module_from_spec(spec)
            sys.modules[TEST_NN_MODEL_NAME] = module
            
            # ---------- 执行模块 & 捕获所有输出 ----------
            try:
                spec.loader.exec_module(module)
            except Exception:
                # exec_module 失败后，再用 subprocess 捕获详细错误
                proc = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True
                )
                stderr = proc.stderr.strip()
                stdout = proc.stdout.strip()
                err_msg = ""
                if stdout: err_msg += "\n[stdout]\n" + stdout
                if stderr: err_msg += "\n[stderr]\n" + stderr

                if proc.returncode != 0:
                    return False, {
                        "exeErr": f"Execution failed with return code {proc.returncode}.{extract_error_and_next_line(err_msg)}",
                        "ioErr": ""
                    }

            # 检查 ModelNew 类是否存在
            model_class = getattr(module, TEST_NN_MODEL_NAME, None)
            if model_class is None:
                return False, {"exeErr": f"Error: Class '{TEST_NN_MODEL_NAME}' not found in the module.", "ioErr": ""}

            # 实例化模型
            try:
                if init_inputs is not None:
                    # cloned_inputs = copy.deepcopy(test_inputs)
                    if isinstance(init_inputs, (list, tuple)):
                        model_instance = model_class(*init_inputs)
                    elif isinstance(init_inputs, dict):
                        model_instance = model_class(**init_inputs)
                    else:
                        model_instance = model_class(init_inputs)
                else:
                    model_instance = model_class()
            except Exception as e:
                return False, {"exeErr": f"Error: Failed to instantiate '{TEST_NN_MODEL_NAME}': {str(e)}", "ioErr": ""}

            # 检查 forward 方法
            if not hasattr(model_instance, 'forward'):
                return False, {"exeErr": f"Error: '{TEST_NN_MODEL_NAME}' does not have a 'forward' method.", "ioErr": ""}
            
            if test_inputs is not None:
                # 运行模型获取输出
                # output = model_instance(*test_inputs)
                cloned_inputs = copy.deepcopy(test_inputs)
                
                # 只有在“喂”给模型的时候，才需要根据类型决定是否解包 (*args, **kwargs)
                if isinstance(cloned_inputs, (list, tuple)):
                    output = model_instance(*cloned_inputs)
                elif isinstance(cloned_inputs, dict):
                    output = model_instance(**cloned_inputs)
                else:
                    # 单个 Tensor 或其他对象
                    output = model_instance(cloned_inputs) 

                def compare_outputs(a, b, atol=1e-2, rtol=1e-2):# 目前这个函数的内存占用比较高，这个函数运算过程中的高内存占用避免不了，但是问题是这个函数返回时内存不会立即释放
                    global data_type_info
                    if isinstance(a, tuple) and isinstance(b, tuple):
                        if len(a) != len(b): return False
                        return all(compare_outputs(x, y, atol, rtol) for x, y in zip(a, b))
                    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                        return torch.allclose(a, b, atol=atol, rtol=rtol)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        return abs(a - b) <= (atol + rtol * abs(b))
                    data_type_info = f"Type mismatch: {type(b)} vs {type(a)}"
                    return False

                if compare_outputs(output, test_outputs, atol=1e-2, rtol=1e-2):
                    print(f"Validate_extracted_code: Test passed")
                else:
                    print(f"Test failed: output mismatch")
                    
                    # # 生成错误报告
                    # # 注意：这里计算 diff 也会产生临时 Tensor，用完要由 GC 回收
                    # diff = torch.abs(output - test_outputs)
                    # tol = 1e-2 + 1e-2 * torch.abs(test_outputs)
                    # error_mask = diff > tol
                    # error_indices = torch.nonzero(error_mask, as_tuple=False)
                    # num_errors = error_indices.size(0)
                    
                    # msg_header = f"Failed (Correctness): Output has {num_errors} mismatches."
                    # error_details = [msg_header, "Top 5 Mismatches:"]
                    
                    # for j in range(min(5, num_errors)):
                    #     idx = tuple(error_indices[j].tolist())
                    #     ref_val = test_outputs[idx].item()
                    #     act_val = output[idx].item()
                    #     error_details.append(f"  Idx: {idx} | Ref: {ref_val:.6f} | Act: {act_val:.6f}")
                    
                    # full_msg = "\n".join(error_details)

                    # 初始化变量，防止后面代码报错导致 UnboundLocalError
                    diff = None
                    tol = None
                    error_mask = None
                    error_indices = None
                    full_msg = ""

                    try:
                        # --- [1. 计算阶段：产生 GPU 临时 Tensor] ---
                        diff = torch.abs(output - test_outputs)
                        # 注意：这里 test_outputs 可能是 CPU 也可能是 GPU，确保计算在同设备
                        tol = 1e-2 + 1e-2 * torch.abs(test_outputs)
                        error_mask = diff > tol
                        error_indices = torch.nonzero(error_mask, as_tuple=False)
                        num_errors = error_indices.size(0)
                        
                        # --- [2. 字符串构建阶段：只产生 CPU 字符串] ---
                        msg_header = f"Failed (Correctness): Output has {num_errors} mismatches (total elements: {test_outputs.numel()})."
                        error_details = [msg_header]
                        error_details.append("Top 5 Mismatches (Index | Reference Value | Actual Value):")
                        
                        # 取前 5 个错误 (只取值 item()，不持有 Tensor 引用)
                        for j in range(min(5, num_errors)):
                            idx = tuple(error_indices[j].tolist())
                            ref_val = test_outputs[idx].item()
                            act_val = output[idx].item()
                            error_details.append(f"  [{j}] Index: {idx} | Ref: {ref_val:.6f} | Act: {act_val:.6f}")
                        
                        full_msg = "\n".join(error_details)

                    except Exception as e:
                        full_msg = f"Error during error reporting: {str(e)}"
                    
                    finally:
                        # --- [3. 清理阶段：在 return 之前立即销毁临时 Tensor] ---
                        # 这一步至关重要，确保在进入外层 finally 的 empty_cache 之前，
                        # 这些临时变量占用的显存标记为可释放。
                        # 这些临时变量占特别大的显存
                        if diff is not None: del diff
                        if tol is not None: del tol
                        if error_mask is not None: del error_mask
                        if error_indices is not None: del error_indices
                        
                        # 这里的局部变量清理完后，显存引用计数归零。
                        # 接下来代码执行 return，会触发函数最外层的 finally (gc.collect + empty_cache)，
                        # 此时显存就能被真正物理释放了。
                    
                    # 验证失败返回前，finally 块会自动执行清理
                    return False, {
                        "exeErr": "",
                        "ioErr": full_msg
                    }

        return True, None

    except Exception as e:
        return False, {
            "exeErr": f"Error during validation: {str(e)}",
            "ioErr": ""
        }

    finally:
        # [!!! 核心修改 !!!] 强制清理显存和引用
        
        # 1. 删除大对象引用
        if output is not None: 
            del output
        if cloned_inputs is not None: 
            del cloned_inputs
        if model_instance is not None: 
            del model_instance
        
        # if model_class is not None:
        #     del model_class

        # if test_inputs is not None:
        #     del test_inputs
        
        # if test_outputs is not None:
        #     del test_outputs
        
        # if init_inputs is not None:
        #     del init_inputs
        
        # 2. 从 sys.modules 中移除模块，防止全局污染和内存泄漏
        if TEST_NN_MODEL_NAME in sys.modules:
            del sys.modules[TEST_NN_MODEL_NAME]
        
        # 3. 删除模块对象本身
        if module is not None:
            del module
            
        # 4. 强制进行垃圾回收
        gc.collect()
        
        # 5. 强制清空 CUDA 缓存，将显存还给操作系统（给 NCU 用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()#这个函数执行完之后会释放上面compare_outputs函数占用的大内存，这种释放是安全的

def main(args):
    print("--- 优化配置 (来自 mini_version/config.py) ---")
    print(f"API URL: {mv_config.DMX_API_BASE_URL}")
    print(f"Iteration Rounds: {mv_config.ITERATION_ROUNDS}")
    print(f"Agent Models: {json.dumps(mv_config.AGENT_MODELS, indent=2)}")
    print("--------------------------------------------------")

    # [!!! 新增 1 !!!]
    # 注册 SIGALRM 信号的处理函数
    signal.signal(signal.SIGALRM, timeout_handler)
    # [!!! 新增结束 !!!]

    # --- B. 设置路径 ---
    # [!!! 已更新 !!!] 
    # KERNELBENCH_PATH 是仓库的根目录
    # KERNELBENCH_DATA_PATH 是 level 
    # 所在的子目录
    KERNELBENCH_DATA_PATH = os.path.join(KERNELBENCH_PATH, "KernelBench")
    kernelbench_level1_dir = os.path.join(KERNELBENCH_DATA_PATH, "level1") 

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    summary_results = {}
    summary_path = os.path.join(results_dir, "summary_results.json")

    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding='utf-8') as f:
                summary_results = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {summary_path} 损坏, 将重新创建。")
            summary_results = {}

    # --- C. 遍历 Level 1 的所有问题 ---
    problem_files = sorted(glob.glob(f"{kernelbench_level1_dir}/*.py"))

    if args.limit_files > 0:
        problem_files = problem_files[:args.limit_files]
        print(f"*** 限制运行，仅测试 {len(problem_files)} 个文件 ***")

    for problem_file_path in problem_files:
        problem_name = os.path.basename(problem_file_path).replace('.py', '')
        problem_name_safe = f"kb_{problem_name}"
        
        problem_results_dir = os.path.join(results_dir, problem_name)
        os.makedirs(problem_results_dir, exist_ok=True)
        
        history_file_path = os.path.join(problem_results_dir, f"{problem_name}_optimization_history.json")
        best_kernel_path = os.path.join(problem_results_dir, f"{problem_name}_best_kernel.cu")
        
        
        if problem_name in summary_results and os.path.exists(best_kernel_path) and not args.force_rerun:# 跳过已经执行的用例
            print(f"--- Skipping {problem_name} (结果已存在于 summary.json) ---")
            continue
            
        print(f"\n\n--- Processing {problem_name} ---")
        
        # [!!! 新增 1 !!!]
        # 为这个 problem 的所有处理设置 try/except/finally 
        # 以捕获超时或未知错误
        
        # [!!! 已修复 !!!] 
        # 显式将这些变量定义为 None 
        # 以确保 finally 块中的 'in locals()' 检查有效
        inputs = None
        gpu_inputs = None
        ref_outputs = None
        pytorch_kernel_module = None
        cpp_wrapper_gpu_inputs = None
        problem_module = None
        
        try:
            # 设置 2 小时 (7200 秒) 闹钟
            signal.alarm(7200) 

            # --- 1. 加载 KernelBench 问题 (不依赖 src) ---
            try:
                # [!!! 已更新 !!!] 
                # 动态加载 .py 文件
                problem_module = load_problem_module_from_file(problem_name, problem_file_path)
                
                # [!!! 已更新 !!!] 
                # 直接从模块调用 get_inputs()
                inputs = problem_module.get_inputs()
                
                # [!!! 已更新 !!!] 
                # 使用 inspect 抓取 PyTorch 
                # 源代码
                pytorch_code = inspect.getsource(problem_module.Model)# 这里不包含get_inputs和get_init_inputs
                with open(problem_file_path, "r", encoding="utf-8") as f:
                    full_pytorch_source_code = f.read()#这里是文件中的完整代码
                
                # [!!! 已修复 !!!] 
                # 使用 get_init_inputs() 来正确实例化模型
                init_inputs = problem_module.get_init_inputs()
                pytorch_kernel_module = problem_module.Model(*init_inputs).cuda()
                
                # [!!! 已更新 !!!] 
                # KernelBench 
                # 总是将 __global__ 
                # 内核命名为 'kernel'
                # kernel_name = "kernel"
                
                # [!!! 已修复 !!!] 
                # 使用 C++-safe 的名称创建 wrapper
                # wrapper_function_name = f"{problem_name_safe}_wrapper"
                
                # [!!! 已更新 !!!] 
                # 自己生成参考输出 (Ref Outputs)
                gpu_inputs = [move_to_cuda(t) for t in inputs]# 经过处理后目前gpu_inputs中的tensor已经是在GPU上了 DONE：如果inputs中存在tensor元组这里应该处理不了
                
                # (确保 ref_outputs 始终是列表)
                ref_outputs_raw = pytorch_kernel_module(*gpu_inputs)#*gpu_inputs 表示解包列表或元组，把里面的元素作为单独参数传入函数。
                # if not isinstance(ref_outputs_raw, (list, tuple)):# 如果返回值 不是 list 或 tuple，即返回单个张量。
                #     ref_outputs = [ref_outputs_raw]#把单个张量放到一个列表里。
                # else:
                #     ref_outputs = list(ref_outputs_raw)
                ref_outputs = ref_outputs_raw
                
                
                cpp_wrapper_gpu_inputs = gpu_inputs.copy()
                        
                # [!!! 通用修复 !!!]
                # 获取所有在 __init__ 中使用的参数
                # init_inputs = problem_module.get_init_inputs()
                
                # 过滤掉布尔值 (例如 'return_indices')，
                # 因为它们通常控制 *输出签名* (例如，返回一个还是两个张量)，
                # 而不是作为 *输入* 参数传递给内核。
                # runtime_init_args = [arg for arg in init_inputs if not isinstance(arg, bool)]# DONE 是不是应该删掉？？？
                
                # if runtime_init_args:
                #     print(f"为 {problem_name} 添加 {len(runtime_init_args)} 个 __init__ 参数: {runtime_init_args}")
                #     cpp_wrapper_gpu_inputs.extend(runtime_init_args)
                


            except Exception as e:
                print(f"Error: 加载 KernelBench 问题 {problem_name} 失败: {e}")
                traceback.print_exc() # 打印详细的堆栈跟踪
                summary_results[problem_name] = {"status": f"Failed to load problem: {e}", "baseline_ms": 0, "best_cuda_ms": float('inf'), "speedup": 0}
                continue # [!!!] 跳到 finally 块，然后到下一个 problem

            # --- 2. 测量 PyTorch 基线时间 (不依赖 src) ---
            try:
                # [!!! 已更新 !!!] 
                # 使用我们自己的基准测试函数
                baseline_time_ms = benchmark_torch_model(pytorch_kernel_module, gpu_inputs)
            except Exception as e:
                print(f"Warning: 测量 PyTorch 基线失败: {e}")
                traceback.print_exc()
                baseline_time_ms = float('inf')
            
            print(f"Pytorch Baseline Time: {baseline_time_ms:.4f} ms")
            try:
                pytorch_baseline_path = os.path.join(problem_results_dir, f"{problem_name}_pytorch_baseline.json")
                baseline_data = {
                    "problem_name": problem_name,
                    "pytorch_baseline_ms": baseline_time_ms
                }
                with open(pytorch_baseline_path, "w", encoding='utf-8') as f:
                    json.dump(baseline_data, f, indent=2)
                print(f"PyTorch baseline stats saved to {pytorch_baseline_path}")
            except Exception as e:
                print(f"Warning: Failed to save PyTorch baseline stats: {e}")
            # --- 3. 生成并验证初始 C++ 和 CUDA 内核 (带修正循环) ---
            initial_cuda_code = None

            if os.path.exists(best_kernel_path) and args.force_rerun:
                try:
                    print(f"Found existing best kernel file: {best_kernel_path}")
                    with open(best_kernel_path, "r", encoding='utf-8') as f:
                        initial_cuda_code = f.read()
                    
                    if initial_cuda_code and len(initial_cuda_code.strip()) > 0:
                        print("✅ Successfully loaded initial CUDA code from best_kernel.cu. Skipping Step 3 (Generation).")
                    else:
                        print("⚠️ best_kernel.cu is empty. Trying fallback...")
                        initial_cuda_code = None
                except Exception as e:
                    print(f"⚠️ Failed to load best_kernel.cu: {e}")
                    initial_cuda_code = None
            if not initial_cuda_code:
            # initial_cuda_source = None
                generation_history = []
                is_correct_and_compiled = False
                
                print(f"Attempt 1/{args.max_correction_attempts + 1}: Generating initial C++/CUDA kernel...")
                
                # 3.1. 初始生成 (Attempt 0)
                # cuda_code = generate_initial_cuda_kernel(
                #     full_pytorch_source_code, 
                #     cpp_wrapper_gpu_inputs, # [!!! 已修复 !!!] 
                #     ref_outputs, 
                # )

                #坏
                cuda_code = '''import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nsource = r\'\'\'\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n\ntemplate <typename scalar_t>\n__device__ __forceinline__ scalar_t sigmoid_func(scalar_t x) {\n    return scalar_t(1) / (scalar_t(1) + exp(-x));\n}\n\n// Kernel: element-wise Sigmoid\ntemplate <typename scalar_t>\n__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,\n                               scalar_t* __restrict__ output,\n                               const int64_t numel) {\n    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < numel) {\n        scalar_t val = input[idx];\n        output[idx] = sigmoid_func(val);\n    }\n}\n\ntorch::Tensor sigmoid_forward(torch::Tensor input) {\n    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");\n    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");\n    auto output = torch::empty_like(input);\n\n    const int64_t numel = input.numel();\n    const int threads = 256;\n    const int64_t blocks = (numel + threads - 1) / threads;\n\n    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_forward_cuda", ([&] {\n        sigmoid_kernel<scalar_t><<<blocks, threads, 0,\n                                   at::cuda::getCurrentCUDAStream()>>>(\n            input.data_ptr<scalar_t>(),\n            output.data_ptr<scalar_t>(),\n            numel);\n    }));\n\n    cudaError_t err = cudaGetLastError();\n    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed with error code ", err);\n    return output;\n}\n\'\'\'\n\ncpp_src = r\'\'\'\ntorch::Tensor sigmoid_forward(torch::Tensor input);\n\'\'\'\n\nsigmoid_module = load_inline(\n    name=\'sigmoid_cuda\',\n    cpp_sources=cpp_src,\n    cuda_sources=source,\n    functions=[\'sigmoid_forward\'],\n    with_cuda=True,\n verbose=True,\n    extra_cuda_cflags=[\'-O2\',\'--ptxas-options=-v\']\n)\n\n\nclass ModelNew(nn.Module):\n    """\n    CUDA-accelerated model that applies element-wise Sigmoid.\n    Mirrors the original Model interface.\n    """\n    def __init__(self):\n        super(ModelNew, self).__init__()\n        self.sigmoid = sigmoid_module\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.sigmoid.sigmoid_forward(x)'''
                #好
                # cuda_code = '''import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\n# ---------------------------------------------------------------------------\n# CUDA source (kernels + C++/ATen host wrappers)\n# ---------------------------------------------------------------------------\nsource = r\'\'\'\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n\ntemplate <typename scalar_t>\n__device__ __forceinline__ scalar_t sigmoid_func(scalar_t x) {\n    return scalar_t(1) / (scalar_t(1) + exp(-x));\n}\n\n/* ---------------------------------------------------------\n * Scalar fallback kernel : one-element per thread\n * ------------------------------------------------------- */\ntemplate <typename scalar_t>\n__global__ void sigmoid_kernel_scalar(const scalar_t* __restrict__ input,\n                                      scalar_t* __restrict__ output,\n                                      const int64_t numel) {\n    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < numel) {\n        output[idx] = sigmoid_func(input[idx]);\n    }\n}\n\n/* ---------------------------------------------------------\n * Vectorised kernel : VEC elements per thread\n * VEC = 4 for float (float4, 16-byte transaction)\n *     = 2 for double (double2, 16-byte transaction)\n * The last (numel % VEC) elements are processed by a\n * single thread (vec_idx == 0) inside the same kernel.\n * ------------------------------------------------------- */\ntemplate <typename scalar_t , int VEC>\n__global__ void sigmoid_kernel_vec(const scalar_t* __restrict__ input,\n                                   scalar_t*       __restrict__ output,\n                                   const int64_t   vec_elems,\n                                   const int64_t   tail_start,\n                                   const int64_t   tail_size) {\n    using VecT = typename std::conditional< (sizeof(scalar_t)==4),\n                                            float4,              // 4 x fp32 = 16 B\n                                            double2               // 2 x fp64 = 16 B\n                                          >::type;\n\n    const int64_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n    /* ---------------- Aligned, vectorised path ---------------- */\n    if (vec_idx < vec_elems) {\n        VecT v = reinterpret_cast<const VecT*>(input)[vec_idx];\n\n        scalar_t* v_elem = reinterpret_cast<scalar_t*>(&v);\n        #pragma unroll\n        for (int i = 0; i < VEC; ++i) {\n            v_elem[i] = sigmoid_func(v_elem[i]);\n        }\n\n        reinterpret_cast<VecT*>(output)[vec_idx] = v;\n    }\n\n    /* ---------------- Tail handling by one thread ------------- */\n    if (tail_size && vec_idx == 0) {\n        for (int64_t j = 0; j < tail_size; ++j) {\n            const int64_t idx = tail_start + j;\n            output[idx] = sigmoid_func(input[idx]);\n        }\n    }\n}\n\n/* ---------------------------------------------------------\n * Host launcher\n * ------------------------------------------------------- */\ntorch::Tensor sigmoid_forward(torch::Tensor input) {\n    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");\n    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");\n\n    auto output = torch::empty_like(input);\n    const int64_t numel = input.numel();\n    const int threads = 256;\n    auto stream = at::cuda::getCurrentCUDAStream();\n\n    // Fast path : fp32 / fp64 with vectorised kernel\n    if (input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble) {\n\n        if (input.scalar_type() == at::kFloat) {\n            using scalar_t = float;\n            constexpr int  VEC = 4;\n            const int64_t  vec_elems  = numel / VEC;\n            const int64_t  tail_start = vec_elems * VEC;\n            const int64_t  tail_sz    = numel - tail_start;\n            const int64_t  blocks     = (vec_elems + threads - 1) / threads;\n\n            if (blocks > 0) {\n                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>(),\n                    output.data_ptr<scalar_t>(),\n                    vec_elems,\n                    tail_start,\n                    tail_sz);\n            } else if (tail_sz) {\n                // Fallback to scalar kernel if vector part is empty\n                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;\n                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>() + tail_start,\n                    output.data_ptr<scalar_t>() + tail_start,\n                    tail_sz);\n            }\n        } else { // double\n            using scalar_t = double;\n            constexpr int  VEC = 2;\n            const int64_t  vec_elems  = numel / VEC;\n            const int64_t  tail_start = vec_elems * VEC;\n            const int64_t  tail_sz    = numel - tail_start;\n            const int64_t  blocks     = (vec_elems + threads - 1) / threads;\n\n            if (blocks > 0) {\n                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>(),\n                    output.data_ptr<scalar_t>(),\n                    vec_elems,\n                    tail_start,\n                    tail_sz);\n            } else if (tail_sz) {\n                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;\n                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>() + tail_start,\n                    output.data_ptr<scalar_t>() + tail_start,\n                    tail_sz);\n            }\n        }\n\n    } else {\n        /* Generic scalar kernel for remaining dtypes (half, bfloat16, etc.) */\n        const int64_t blocks = (numel + threads - 1) / threads;\n        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),\n                                            "sigmoid_forward_cuda_scalar", ([&] {\n            sigmoid_kernel_scalar<scalar_t><<<blocks, threads, 0, stream>>>(\n                input.data_ptr<scalar_t>(),\n                output.data_ptr<scalar_t>(),\n                numel);\n        }));\n    }\n\n    cudaError_t err = cudaGetLastError();\n    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed with error code ", err);\n\n    return output;\n}\n\'\'\'\n\n# ---------------------------------------------------------------------------\n# C++ function prototypes\n# ---------------------------------------------------------------------------\ncpp_src = r\'\'\'\ntorch::Tensor sigmoid_forward(torch::Tensor input);\n\'\'\'\n\n# ---------------------------------------------------------------------------\n# Build & load extension\n# ---------------------------------------------------------------------------\nsigmoid_module = load_inline(\n    name         = \'sigmoid_cuda_opt\',\n    cpp_sources  = cpp_src,\n    cuda_sources = source,\n    functions    = [\'sigmoid_forward\'],\n    with_cuda    = True,\n    verbose      = True,\n    extra_cuda_cflags=[\'-O3\', \'--ptxas-options=-v\']\n)\n\n# ---------------------------------------------------------------------------\n# PyTorch Module wrapper\n# ---------------------------------------------------------------------------\nclass ModelNew(nn.Module):\n    """\n    CUDA-accelerated model that applies element-wise Sigmoid.\n    Mirrors the original Model interface.\n    """\n    def __init__(self):\n        super(ModelNew, self).__init__()\n        self.sigmoid = sigmoid_module\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.sigmoid.sigmoid_forward(x)'''

                generation_history.append({
                    "attempt": 0,
                    "type": "generation",
                    "response_cuda_code": cuda_code
                    # "response": response_text,
                    # "cpp_code_extracted": bool(cpp_code),
                    # "cuda_code_extracted": bool(cuda_code),
                    # "error": ""
                })

                if not cuda_code:
                    print("The initialization phase did not produce the correct cuda_code.")
                else:
                    # initial_cpp_source = cpp_code
                    # initial_cuda_source = cuda_code
                    initial_cuda_code = cuda_code

                    # 3.2. 验证与修正循环 (最多10次，由 args.max_correction_attempts 控制)
                    for attempt in range(args.max_correction_attempts):
                        if not initial_cuda_code:
                            print(f"Cannot correct code (missing). Aborting for {problem_name}.")
                            break

                        print(f"--- Verification Attempt {attempt+1}/{args.max_correction_attempts} ---")
                        verResult, errMessage = validate_extracted_code(initial_cuda_code, init_inputs, gpu_inputs, ref_outputs)



                        # 步骤 3: 修正 (如果发生任何错误)
                        if not verResult:
                            print(f"Error captured. Attempting LLM correction (Attempt {attempt+1})...")
                            
                            # 打印一部分错误信息用于调试
                            err_str = str(errMessage)
                            print(f"--- Error Snippet ---\n{err_str[:500]}...\n---------------------")

                            # 截断过长
                            if len(err_str) > 4000:
                                err_str = err_str[:2000] + "\n...[TRUNCATED]...\n" + err_str[-2000:]
                                
                            cuda_code = correct_cuda_kernel(
                                full_pytorch_source_code,
                                initial_cuda_code,
                                errMessage
                            )
                            
                            generation_history.append({
                                "attempt": attempt + 1,
                                "type": "correction",
                                "error_sent": err_str,
                                "response_cuda_code": cuda_code,
                            })
                            
                            if cuda_code:
                                initial_cuda_code = cuda_code
                                print("Code corrected by LLM. Retrying verification...")
                            else:
                                print("LLM correction failed (did not return valid code). Aborting.")
                                break
                        else:
                            is_correct_and_compiled = True
                            break

                # 保存初始内核的生成/修正历史 (用于调试)
                init_gen_path = os.path.join(problem_results_dir, f"{problem_name}_initial_generation.json")
                with open(init_gen_path, "w", encoding='utf-8') as f:
                    json.dump(generation_history, f, indent=2, ensure_ascii=False)

                # 3.3. 检查最终结果
                if not is_correct_and_compiled:
                    print(f"FAILED to generate a correct initial C++/CUDA source after {args.max_correction_attempts + 1} attempts. Skipping optimization.")
                    summary_results[problem_name] = {
                        "baseline_ms": baseline_time_ms,
                        "best_cuda_ms": float('inf'),
                        "speedup": 0.0,
                        "status": "Failed initial generation/correction"
                    }
                    continue # [!!!] 跳到 finally 块，然后到下一个 problem

            # --- 4. 调用 mini_version 的优化循环 ---
            
            print(f"--- Calling mini_version optimization framework for {problem_name} ---")
            
            best_time_ms = float('inf')
            best_kernel_code_full = initial_cuda_code # 默认为初始代码
            status = "Failed (Unknown)"
            del pytorch_kernel_module
            torch.cuda.empty_cache()
            try:
                # [!!! 核心调用 !!!] 
                # (这部分不变，因为它依赖 mini_version, 
                # 而 mini_version 
                # 已经被重构为通用)
                best_node = mv_main.run_optimization_on_problem(
                    problem_name=problem_name_safe,
                    # cpp_source=initial_cpp_source,
                    # initial_cuda_code=initial_cuda_source,]
                    initial_cuda_code = best_kernel_code_full,
                    inputs=cpp_wrapper_gpu_inputs, # [!!! 已修复 !!!] # 不包括init_input
                    init_inputs = init_inputs, # 这个是init_inputs
                    ref_outputs=ref_outputs,# pytorch的参考输出
                    # kernel_name=kernel_name,
                    # wrapper_function_name=wrapper_function_name,
                    iteration_rounds=mv_config.ITERATION_ROUNDS,
                    history_file_path=history_file_path,
                    baseline_time_ms=baseline_time_ms, # [!!! 已更新，传入基线时间 !!!]
                    full_pytorch_source_code = full_pytorch_source_code
                )
                
                if 'error' in best_node:
                    raise Exception(best_node['error'])

                best_time_ms = best_node.get('time_ms', float('inf'))
                # best_node['code'] 只包含 __global__ 内核
                best_kernel_code_full = best_node['code']
                status = "Success"
                
            except Exception as e:
                print(f"Optimization loop failed for {problem_name}: {e}")
                traceback.print_exc()
                status = f"Failed optimization loop: {e}"

            # [!!! 移除 !!!] 
            # 我们不再需要 "5. 保存此问题的结果" 
            # 因为 mini_version/main.py 
            # 已经实时保存了

            # --- 6. 统计加速比 ---
            speedup = baseline_time_ms / best_time_ms if best_time_ms > 0 and best_time_ms < float('inf') and baseline_time_ms < float('inf') else 0.0
            
            print(f"--- Finished {problem_name} ---")
            print(f"PyTorch Baseline: {baseline_time_ms:.4f} ms")
            print(f"Best CUDA Time:   {best_time_ms:.4f} ms")
            print(f"Speedup:          {speedup:.2f}x")
            
            summary_results[problem_name] = {
                "baseline_ms": baseline_time_ms,
                "best_cuda_ms": best_time_ms,
                "speedup": speedup,
                "status": status
            }

        # [!!! 新增 1 !!!]
        # 捕获超时
        except TimeoutException as e:
            print(f"--- FAILED (Timeout) {problem_name}: {e} ---")
            summary_results[problem_name] = {
                "baseline_ms": 0, "best_cuda_ms": float('inf'), "speedup": 0,
                "status": f"Failed (Timeout: {e})"
            }
        
        # 捕获所有其他意外错误 (例如 OOM, Segfaults)
        except Exception as e:
            print(f"--- FAILED (Unknown Error) {problem_name}: {e} ---")
            traceback.print_exc()
            summary_results[problem_name] = {
                "baseline_ms": 0, "best_cuda_ms": float('inf'), "speedup": 0,
                "status": f"Failed (Unknown Error: {e})"
            }

        # [!!! 新增 1 !!!]
        # 无论成功、失败还是超时，
        # finally 块总是执行
        finally:
            signal.alarm(0) # 关闭超时闹钟
            print(f"♻️  Global Cleanup: resetting environment after {problem_name}...")
            
            try:
                # 1. 按照依赖顺序反向删除对象
                # 先删 ref_outputs (依赖 gpu_inputs/model)
                if 'ref_outputs' in locals() and ref_outputs is not None:
                    del ref_outputs
                
                # 删 inputs
                if 'gpu_inputs' in locals() and gpu_inputs is not None:
                    del gpu_inputs
                if 'inputs' in locals() and inputs is not None:
                    del inputs
                if 'cpp_wrapper_gpu_inputs' in locals() and cpp_wrapper_gpu_inputs is not None:
                    del cpp_wrapper_gpu_inputs

                # 删模型 (最占显存的部分)
                if 'pytorch_kernel_module' in locals() and pytorch_kernel_module is not None:
                    del pytorch_kernel_module
                
                # 删 Python 模块对象 (防止 sys.modules 泄漏)
                if 'problem_module' in locals() and problem_module is not None:
                    del problem_module
                if problem_name in sys.modules:
                    del sys.modules[problem_name]

                # 2. 清理全局 JIT 缓存
                # 这一步非常重要，防止上一个任务的 .so 文件干扰下一个任务
                # import cuda_utils as mv_cuda_utils # 确保能访问到
                mv_cuda_utils._gemm_module = None

                # 3. 强制 GC 和 Empty Cache
                gc.collect()
                
                # 同样包裹 try...except，防止上一个任务把 Context 搞坏了导致这里报错
                torch.cuda.empty_cache()
                
                print("✅ Cleanup complete. Memory recycled.")

            except Exception as cleanup_e:
                print(f"⚠️  Global cleanup warning for {problem_name}: {cleanup_e}")
                # 即使这里报错，也不会影响下一个 for 循环的开始
            
            # [可选] 打印当前显存占用情况，确认回收效果
            print(f"Current Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                # 即使清理失败也要继续
        
        # --- 7. 实时保存摘要 ---
        # [!!!] 注意：此块现在位于 try/except/finally 
        # 之外。
        # 这确保了无论用例是成功、失败还是超时，
        # `summary_results` 
        # 都会被更新并立即保存。
        # 这满足了你的需求 "3. 实时保存（摘要）"
        with open(summary_path, "w", encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)

    print("\n\n--- ALL DONE ---")
    print(f"Optimization summary saved to {summary_path}")


# --- 6. 命令行参数解析 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM optimization on KernelBench Level 1.")

    # 路径参数
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./kb_level1_optimization_results",
        help="保存最优内核和历史记录的目录"
    )
    
    # [!!! 已移除 !!!] 
    # 所有 LLM 参数 (api_key, api_url, model, max_tokens, 
    # temperature, max_iterations)
    # 现在都由 mini_version/config.py 
    # (例如 ITERATION_ROUNDS) 控制。

    # 循环参数
    parser.add_argument(
        "--max_correction_attempts",
        type=int,
        default=10, # [!!! 已更新 !!!] 如您所要求的10轮
        help="为生成正确的初始 C++/CUDA 内核所做的最大修正尝试次数 (0 = 仅生成，不修正)"
    )

    # 调试参数
    parser.add_argument(
        "--limit_files",
        type=int,
        default=0,
        help="限制测试文件的数量（0=全部100个）"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="如果结果文件已存在，则强制重新运行"
    )

    # 检查路径
    if not os.path.isdir(KERNELBENCH_PATH):
        print(f"Error: KERNELBENCH_PATH '{KERNELBENCH_PATH}' 不是一个有效的目录。请在脚本中编辑此路径。")
        sys.exit(1)
    if not os.path.isdir(MINI_VERSION_PATH):
        print(f"Error: MINI_VERSION_PATH '{MINI_VERSION_PATH}' 不是一个有效的目录。请在脚本中编辑此路径。")
        sys.exit(1)

    args = parser.parse_args()
    main(args)