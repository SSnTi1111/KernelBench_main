import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import torch 
import re
import importlib.util 
import traceback     
import inspect # [!!! 新增 !!!] 用于从 .py 文件中抓取源代码
import signal # [!!! 新增 !!!]
import gc

# --- 1. 设置项目路径 ---
KERNELBENCH_PATH = "/home/lxt/KernelBench/KernelBench"
MINI_VERSION_PATH = "/home/lxt/KernelBench/KernelBench/mini_version"

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

# --- 3. 辅助函数：Pytorch -> CUDA (初始生成) ---

def extract_code_block(text, lang):
    """(此函数保持不变)"""
    pattern = rf'```{lang}\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_pytorch_to_cuda_prompt(pytorch_code, inputs, ref_outputs, kernel_name, wrapper_name):
    """
    (此函数已更新，以匹配 mini_version 的编译结构)
    """
    input_specs = []
    input_params = []
    

    for i, item in enumerate(inputs):
        if isinstance(item, torch.Tensor):
            input_specs.append(f"  Input {i} (arg{i}): shape={item.shape}, dtype={item.dtype}")
            input_params.append(f"torch::Tensor arg{i}")
        elif isinstance(item, float):
            input_specs.append(f"  Input {i} (arg{i}): type=float, value={item}")
            input_params.append(f"double arg{i}") # C++ 中 PyTorch float 对应 double
        elif isinstance(item, int):
            input_specs.append(f"  Input {i} (arg{i}): type=int, value={item}")
            input_params.append(f"int64_t arg{i}") # C++ 中 PyTorch int 对应 int64_t
        else:
            print(f"Warning: get_pytorch_to_cuda_prompt 中未知的输入类型: {type(item)}")

    outputs_list = ref_outputs if isinstance(ref_outputs, (list, tuple)) else [ref_outputs]
    output_specs = []
    output_return_type = "torch::Tensor" # 默认为单个输出
    if len(outputs_list) > 1:
        output_return_type = "std::vector<torch::Tensor>"
        
    for i, tensor in enumerate(outputs_list):
        output_specs.append(f"  Output {i}: shape={tensor.shape}, dtype={tensor.dtype}")

    input_specs_str = "\n".join(input_specs)
    output_specs_str = "\n".join(output_specs)
    input_params_str = ", ".join(input_params)

    prompt = f"""
你是一位顶级的 CUDA 和 PyTorch C++ 扩展专家。
你的任务是将一个 PyTorch 内核转换为与 `torch.utils.cpp_extension.load_inline` 兼容的代码。

你必须生成 *两个* 独立的代码块：

1.  **C++ 签名 (```cpp ... ``` 块)**: 
    * **仅** 包含 `torch/extension.h` 头文件。
    * **仅** 包含 C++ wrapper 函数的 *声明* (即原型/签名)，以分号结尾。

2.  **C++/CUDA 组合实现 (```cu ... ``` 块)**:
    * 包含 *所有* 必要的头文件 (例如 `<torch/extension.h>`, `<cuda_runtime.h>`, `<cuda.h>`, `<cmath>`)。
    * 包含所有 CUDA 辅助函数（例如 `blockReduceSum`）。
    * **[重要]** 辅助函数必须在使用它们的 `__global__` 内核 *之前* 被 *定义* 或 *声明*。
    * 包含 `__global__ void {kernel_name}(...)` 的 *实现*。
    * 包含 C++ wrapper `{wrapper_name}` 的 *完整实现* (函数体)。

[问题规格]
PyTorch 内核实现:
```python
{pytorch_code}
```

输入 (已在 GPU 上):

{input_specs_str}

输出 (应在 GPU 上生成):

{output_specs_str}

[模板]

--- C++ 签名 (`cpp`) ---
```cpp
#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
{output_return_type} {wrapper_name}({input_params_str});
```

--- C++/CUDA 组合实现 (`cu`) ---

**代码段**

```cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// [!!! 关键 !!!] 
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {{
    // ... (如果需要，请实现 blockReduceSum)
    // ... (确保在 kernel 之前声明/定义)
  
    // 示例 Warp 内归约
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
  
    // Warp内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {{
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }}
  
    // 每个warp的第一个线程将结果写入共享内存
    if (lane == 0) {{
        shared[wid] = val;
    }}
    __syncthreads();
  
    // 第一个warp进行最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {{
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {{
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }}
    }}
    return val;
}}


// CUDA 内核实现
__global__ void {kernel_name}(...) {{
    // ... 内核逻辑 ...
    // (例如，调用 blockReduceSum)
}}

// C++ Wrapper 实现
{output_return_type} {wrapper_name}({input_params_str}) {{
    // ... 验证输入 ...
    // ... 分配输出张量 ...
    // ... 计算网格/块维度 ...
  
    // ... 调用内核 ...
    // {kernel_name}<<<...>>>(...);
  
    // ... 返回输出 ...
}}
```

请严格按照上述模板，分别提供 C++ 签名 (cpp ... ) 和 C++/CUDA 组合实现 (cu ... ) 两个代码块。

"""
    return prompt

# [!!! 已更新 !!!] 
# 此函数现在调用 mv_llm_api.call_llm 并提取两个代码块
def generate_initial_cuda_kernel(pytorch_code, inputs, ref_outputs, kernel_name, wrapper_name):
    """
    调用 LLM API 来生成初始的 C++ Wrapper 和 CUDA 内核代码。
    """
    prompt = get_pytorch_to_cuda_prompt(pytorch_code, inputs, ref_outputs, kernel_name, wrapper_name) 
    system_prompt = "你是一位专业的 CUDA 程序员，擅长将 PyTorch 转换为 CUDA C++ 扩展。"
    
    try:
        response_text = mv_llm_api.call_llm(
            agent_name="initial_generator", 
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        if not response_text:
            raise Exception("LLM 响应为空")

        cpp_code = extract_code_block(response_text, "cpp")
        cuda_code = extract_code_block(response_text, "cu")

        if not cpp_code:
            print("Error: 未能在 LLM 响应中找到 ```cpp ... ``` 块。")
            return None, None, response_text
        
        if not cuda_code:
            print("Error: 未能在 LLM 响应中找到 ```cu ... ``` 块。")
            return None, None, response_text

        return cpp_code, cuda_code, response_text
        
    except Exception as e:
        print(f"Error during initial kernel generation: {e}")
        return None, None, str(e)
def get_cuda_correction_prompt(pytorch_code, failing_cpp, failing_cuda, error_message):
    prompt = f"""
你是一位顶级的 CUDA 和 PyTorch C++ 扩展专家。
你的任务是修复一个编译或运行时失败的 C++/CUDA 内核。
[原始 PyTorch 代码 (上下文)]
```python
{pytorch_code}
```
[失败的 C++ 签名 (`cpp`)]
```cpp
{failing_cpp}
```
[失败的 C++/CUDA 实现 (`cu`)]
```cu
{failing_cuda}
```
[!!! 编译/运行 错误信息 !!!]

```
{error_message}
```
[任务]
请仔细分析错误信息 (例如 "error: namespace 'c10::cuda' has no member 'getCurrentCUDAStream'" 或 "Failed (Correctness)") 并修复 C++/CUDA 代码。
[重要]

1. **修复错误**: 错误可能在 .cu 文件的 include、函数实现、或 C++ wrapper 中。
   * (提示: 常见的错误 `c10::cuda::getCurrentCUDAStream` 在 PyTorch 2.1+ 中应替换为 `at::cuda::getCurrentCUDAStream()`，并且需要 `#include <ATen/cuda/CUDAContext.h>`)
2. **保持结构**: 严格按照原始 C++ 签名 (cpp) 和 C++/CUDA 组合实现 (cu) 的格式返回 *两个* 完整的代码块。
3. **返回完整代码**: 即使只修改了 .cu 文件，也请返回 *两个* 代码块 (cpp 和 cu)。
[模板]

--- C++ 签名 (cpp) ---
```cpp
// ... (返回完整的、可能是修复后的 C++ 签名) ...
```

--- C++/CUDA 组合实现 (`cu`) ---
```cu
// ... (返回完整的、修复后的 C++/CUDA 实现) ...
```
"""
    return prompt

def correct_cuda_kernel(pytorch_code, failing_cpp, failing_cuda, error_message): 
    prompt = get_cuda_correction_prompt(pytorch_code, failing_cpp, failing_cuda, error_message) 
    system_prompt = "你是一位专业的 CUDA 程序员，擅长调试 C++ 扩展编译和运行时错误。"
    try:
        response_text = mv_llm_api.call_llm(
            # [!!! 注意 !!!] 
            # 确保 "initial_corrector" 在 config.py 中已定义!
            agent_name="initial_corrector", 
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        if not response_text:
            raise Exception("LLM 响应为空")

        cpp_code = extract_code_block(response_text, "cpp")
        cuda_code = extract_code_block(response_text, "cu")

        if not cpp_code or not cuda_code:
            print("Error: 修正 LLM 响应中未找到 'cpp' 或 'cu' 块。")
            return None, None, response_text

        return cpp_code, cuda_code, response_text

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
# [!!! 新增结束 !!!]

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
        
        
        if problem_name in summary_results and os.path.exists(best_kernel_path) and not args.force_rerun:
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
                pytorch_code = inspect.getsource(problem_module.Model)
                
                # [!!! 已修复 !!!] 
                # 使用 get_init_inputs() 来正确实例化模型
                init_inputs = problem_module.get_init_inputs()
                pytorch_kernel_module = problem_module.Model(*init_inputs).cuda()
                
                # [!!! 已更新 !!!] 
                # KernelBench 
                # 总是将 __global__ 
                # 内核命名为 'kernel'
                kernel_name = "kernel"
                
                # [!!! 已修复 !!!] 
                # 使用 C++-safe 的名称创建 wrapper
                wrapper_function_name = f"{problem_name_safe}_wrapper"
                
                # [!!! 已更新 !!!] 
                # 自己生成参考输出 (Ref Outputs)
                gpu_inputs = [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in inputs]
                
                # (确保 ref_outputs 始终是列表)
                ref_outputs_raw = pytorch_kernel_module(*gpu_inputs)
                if not isinstance(ref_outputs_raw, (list, tuple)):
                    ref_outputs = [ref_outputs_raw]
                else:
                    ref_outputs = list(ref_outputs_raw)
                
                
                cpp_wrapper_gpu_inputs = gpu_inputs.copy()
                
                if problem_name == "20_LeakyReLU" and hasattr(pytorch_kernel_module, 'negative_slope'):
                    print(f"检测到 {problem_name}，添加 negative_slope={pytorch_kernel_module.negative_slope}")
                    cpp_wrapper_gpu_inputs.append(pytorch_kernel_module.negative_slope)
                


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
            initial_cpp_source = None
            initial_cuda_source = None
            generation_history = []
            is_correct_and_compiled = False
            
            print(f"Attempt 1/{args.max_correction_attempts + 1}: Generating initial C++/CUDA kernel...")
            
            # 3.1. 初始生成 (Attempt 0)
            cpp_code, cuda_code, response_text = generate_initial_cuda_kernel(
                pytorch_code, 
                cpp_wrapper_gpu_inputs, # [!!! 已修复 !!!] 
                ref_outputs, 
                kernel_name,
                wrapper_function_name
            )
            
            generation_history.append({
                "attempt": 0,
                "type": "generation",
                "response": response_text,
                "cpp_code_extracted": bool(cpp_code),
                "cuda_code_extracted": bool(cuda_code),
                "error": ""
            })

            if not cpp_code or not cuda_code:
                print("Initial generation FAILED. Skipping correction loop.")
            else:
                initial_cpp_source = cpp_code
                initial_cuda_source = cuda_code

                # 3.2. 验证与修正循环 (最多10次，由 args.max_correction_attempts 控制)
                for attempt in range(args.max_correction_attempts):
                    if not initial_cpp_source or not initial_cuda_source:
                        print(f"Cannot correct code (missing). Aborting for {problem_name}.")
                        break

                    print(f"--- Verification Attempt {attempt+1}/{args.max_correction_attempts} ---")
                    error_message = ""
                    
                    try:
                        # 步骤 1: 尝试编译
                        print("Attempting compilation...")
                        current_module_name = f"{problem_name_safe}_init_test_{attempt}"
                        
                        # [!!!] 临时设置全局模块为 None，强制重编译
                        mv_cuda_utils._gemm_module = None 
                        
                        module, stdout_log, stderr_log = mv_cuda_utils.load_gemm_module(
                            initial_cpp_source, 
                            initial_cuda_source, 
                            module_name=current_module_name,
                            wrapper_function_name=wrapper_function_name
                        )
                        print("Compilation SUCCESSFUL.")

                        # 步骤 2: 尝试检查正确性 (如果编译成功)
                        print("Checking correctness...")
                        try:
                            is_correct = mv_cuda_utils.check_correctness(
                                cpp_wrapper_gpu_inputs, # [!!! 已修复 !!!] 
                                ref_outputs, 
                                wrapper_function_name
                            )
                        except Exception as e:
                            # 捕获内核内部的运行时错误 (e.g., segfault)
                            print(f"Runtime Error during correctness check: {e}")
                            is_correct = False
                            error_message = f"Runtime Error during check_correctness: {e}\n{traceback.format_exc()}"
                            
                        if is_correct:
                            print("Correctness VERIFIED. Initial kernel is valid.")
                            is_correct_and_compiled = True
                            break # [!!! 成功，退出修正循环 !!!]
                        else:
                            print("Correctness FAILED.")
                            if not error_message: # 如果 check_correctness 只是返回 False
                                error_message = "Failed (Correctness): Kernel output does not match reference output."

                    except RuntimeError as e:
                        # 捕获 `load_gemm_module` (Compilation) 失败
                        print("Compilation FAILED.")
                        error_message = str(e)
                    except Exception as e:
                        # 捕获其他意外错误
                        print(f"An unexpected error occurred during verification: {e}")
                        error_message = f"Unexpected Error: {e}\n{traceback.format_exc()}"

                    # 步骤 3: 修正 (如果发生任何错误)
                    if error_message:
                        print(f"Error captured. Attempting LLM correction (Attempt {attempt+1})...")
                        # (截断长的错误信息)
                        if len(error_message) > 4000:
                             error_message = error_message[:2000] + "\n...[TRUNCATED]...\n" + error_message[-2000:]
                             
                        cpp_code_corr, cuda_code_corr, response_text_corr = correct_cuda_kernel(
                            pytorch_code,
                            initial_cpp_source,
                            initial_cuda_source,
                            error_message
                        )
                        
                        generation_history.append({
                            "attempt": attempt + 1,
                            "type": "correction",
                            "error_sent": error_message,
                            "response": response_text_corr,
                            "cpp_code_extracted": bool(cpp_code_corr),
                            "cuda_code_extracted": bool(cuda_code_corr)
                        })
                        
                        if cpp_code_corr and cuda_code_corr:
                            initial_cpp_source = cpp_code_corr
                            initial_cuda_source = cuda_code_corr
                            print("Code corrected by LLM. Retrying verification...")
                        else:
                            print("LLM correction failed (did not return valid code). Aborting.")
                            break
                    else:
                        # 这不应该发生，但作为保险
                        print("Verification failed but no error message was captured. Aborting.")
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
            best_kernel_code_full = initial_cpp_source + "\n\n" + initial_cuda_source # 默认为初始代码
            status = "Failed (Unknown)"
            
            try:
                # [!!! 核心调用 !!!] 
                # (这部分不变，因为它依赖 mini_version, 
                # 而 mini_version 
                # 已经被重构为通用)
                best_node = mv_main.run_optimization_on_problem(
                    problem_name=problem_name_safe,
                    cpp_source=initial_cpp_source,
                    initial_cuda_code=initial_cuda_source,
                    inputs=cpp_wrapper_gpu_inputs, # [!!! 已修复 !!!] 
                    ref_outputs=ref_outputs,
                    kernel_name=kernel_name,
                    wrapper_function_name=wrapper_function_name,
                    iteration_rounds=mv_config.ITERATION_ROUNDS,
                    history_file_path=history_file_path,
                    baseline_time_ms=baseline_time_ms # [!!! 已更新，传入基线时间 !!!]
                )
                
                if 'error' in best_node:
                    raise Exception(best_node['error'])

                best_time_ms = best_node.get('time_ms', float('inf'))
                # best_node['code'] 只包含 __global__ 内核
                best_kernel_code_full = initial_cpp_source + "\n\n" + best_node['code']
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
            # [!!!] 关键：禁用闹钟，
            # 否则它可能在下一个循环中触发
            signal.alarm(0)
            
            # [!!! 已修复 !!!] 
            # 显式删除大张量并清空 CUDA 缓存
            # 这对于防止OOM和上下文错误至关重要
            print(f"Cleaning up resources for {problem_name}...")
            try:
                # 删除在 try 块中创建的变量
                if 'inputs' in locals() and inputs is not None:
                    del inputs
                if 'gpu_inputs' in locals() and gpu_inputs is not None:
                    del gpu_inputs
                if 'ref_outputs' in locals() and ref_outputs is not None:
                    del ref_outputs
                if 'pytorch_kernel_module' in locals() and pytorch_kernel_module is not None:
                    del pytorch_kernel_module
                if 'cpp_wrapper_gpu_inputs' in locals() and cpp_wrapper_gpu_inputs is not None:
                    del cpp_wrapper_gpu_inputs
                
                # 强制Python垃圾回收
                gc.collect()
                
                # 强制PyTorch清空CUDA缓存
                torch.cuda.empty_cache()
                
                # 重置全局编译模块
                mv_cuda_utils._gemm_module = None
                print("Cleanup complete.")
            except Exception as e:
                print(f"Warning: Resource cleanup failed for {problem_name}: {e}")
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