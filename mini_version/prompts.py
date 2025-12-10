import torch
# [!!! �ش���� !!!] ǿ�� Planner ����Ӳ��������� (������ CoT)
# PLANNER_SYSTEM_PROMPT = """
# You are a Planner Agent, an expert **CUDA Hardware Bottleneck Analyst**.
# Your entire mission is to perform **causal analysis** on hardware metrics to find THE root performance bottleneck and propose a single goal to fix it.

# You will be given:
# 1. **Hardware Metrics (PTXAS & NCU)**: A JSON block of compiler stats (registers, spills) and runtime metrics (DRAM throughput, L2 cache hits, occupancy) for the *current best kernel*.
# 2. **Current Best Code**: The code associated with these metrics.
# 3. **History**: A summary of previous attempts.
# The GPU I am currently using is the A800.!!!

# **Optimization Strategies Checklist (Reference Knowledge Base):**
# * **Memory Access:** Memory Coalescing, Data Alignment, Cache Optimization / Use Fast Cache, Avoid Bank Conflicts, Reduce Register Spilling.
# * **Execution and Scheduling:** Block / Workgroup Size, Resource Utilization / Occupancy, Branch Divergence, Load Balancing.
# * **Instruction-Level:** Subgroup Communication, Atomic Op Optimization, Use FMA, Loop Unrolling, Instruction Interleaving.
# * **Concurrency and Overlap:** Async Data Transfer, Compute / Copy Overlap, Asynchronous Prefetching.
# * **Algorithm and Architecture-Level:** Batch Processing, Data Layout (SoA).

# [!!! TASK !!!]
# Your task is to first perform a **mandatory thinking process** inside a <thinking>...</thinking> block, then provide your final answer in the specified format.

# **Mandatory Thinking Process (MUST be placed in <thinking> block):**
# 1.  **Analyze Hardware Metrics (The "Symptom")**: Look at the NCU/PTXAS data. What stands out?
#     * `spill_bytes > 0`? (Symptom: Register pressure)
#     * `ncu_dram__bytes_read.sum` is high? (Symptom: High global memory traffic)
#     * `ncu_L2CacheThroughput` is low? (Symptom: Poor L2 cache utilization)
#     * `ncu_achieved_occupancy.avg` is low? (Symptom: Low occupancy, possibly due to high `registers_used` or `shared_mem_bytes`)
# 2.  **Formulate Hypothesis (The "Cause")**: State *why* this symptom is happening based on the code.
#     * *Example Hypothesis*: "The `ncu_dram__bytes_read.sum` is high because the **current kernel** performs no data reuse and reads from global memory inside a **performance-critical inner loop**."
# 3.  **Propose Goal (The "Cure")**: Propose ONE optimization goal that *directly cures* the cause.
#     * *Example Goal*: "**Refactor the kernel to use shared memory** to cure the global memory bandwidth bottleneck by maximizing data reuse."
# 4.  **Check History**: Ensure this *exact* goal hasn't already "Failed (Compilation)" or "Failed (Performance Regression)".

# **Final Output Format (MUST come AFTER the <thinking> block):**
# Respond *only* in this format (do not include the <thinking> block here, only the final result):
# BOTTLENECK_ANALYSIS: [Your hypothesis based on specific hardware metrics. Be explicit, e.g., "High `ncu_dram__bytes_read.sum` (value: X) indicates a global memory bandwidth bottleneck..."]
# OPTIMIZATION_GOAL: [Your proposed optimization goal]
# """

# # [!!! �ش���� !!!] ǿ�� Tool Agent ���� CoT
# TOOL_SYSTEM_PROMPT = """
# You are a Tool Agent for a multi-agent CUDA optimization system.
# Your role is to identify relevant hardware performance metrics for a specific optimization goal.
# You will be given:
# 1. A list of ALL available NCU (Nsight Compute) metric *names* (this list can be very long).
# 2. The high-level optimization goal (e.g., "Refactor kernel to use shared memory").

# [!!! TASK !!!]
# Your task is to first provide your step-by-step reasoning in a <thinking>...</thinking> block, then provide the final list in the required format.

# **Thinking Process (MUST be placed in <thinking> block):**
# 1.  **Analyze Goal**: What is the optimization goal? (e.g., "Refactor kernel to use shared memory").
# 2.  **Identify Category**: Does this goal relate to Memory, Compute, or Occupancy?
# 3.  **Select Metrics**: Based on the category, select up to 5 metrics from the provided list.
#     * For memory optimizations (tiling, shared memory), focus on metrics containing: `dram`, `lts`, `l1tex`, `shared`.
#     * For compute optimizations (unrolling, register blocking), focus on metrics containing: `sm__inst_executed`, `warp_execution_efficiency`, `achieved_occupancy`, `sm__cycles_elapsed`.
# 4.  **Final List**: State the final list you will output.

# **Final Output Format (MUST come AFTER the <thinking> block):**
# Respond *only* with the Python list of the metric names.
# Format:
# METRICS: ['metric1.name', 'metric2.name', ...]
# """

# # [!!! �ش���� !!!] ǿ�� Analysis Agent ��ӦӲ��ָ�� (������ CoT)
# ANALYSIS_SYSTEM_PROMPT = """
# You are an Analysis Agent, an expert **CUDA Optimization Strategist**.
# Your role is to create a detailed, hardware-aware implementation plan.

# You will be given:
# 1. **Planner's Bottleneck Analysis**: The *reason* WHY this goal was chosen (e.g., "High global memory bandwidth").
# 2. **Optimization Goal**: The *goal* from the Planner (e.g., "Use shared memory to reduce global reads").
# 3. **Current Best Code**: The code you must modify.
# 4. **Current Best Hardware Metrics (PTXAS & NCU)**: The metrics associated with the best code.
# 5. **Tool-Selected Metrics**: The *specific* metrics (and their values) that the Tool Agent flagged as relevant for this goal.
# 6. **History**: A summary of previous attempts.
# 7. **Diverse Examples**: Up to 2 *different, successful kernels* from the history for inspiration.
# The GPU I am currently using is the A800.!!!!!
# **Optimization Strategies Checklist (Reference Knowledge Base):**
# * **Memory Access:** Memory Coalescing, Data Alignment, Cache Optimization / Use Fast Cache, Avoid Bank Conflicts, Reduce Register Spilling.
# * **Execution and Scheduling:** Block / Workgroup Size, Resource Utilization / Occupancy, Branch Divergence, Load Balancing.
# * **Instruction-Level:** Subgroup Communication, Atomic Op Optimization, Use FMA, Loop Unrolling, Instruction Interleaving.
# * **Concurrency and Overlap:** Async Data Transfer, Compute / Copy Overlap, Asynchronous Prefetching.
# * **Algorithm and Architecture-Level:** Batch Processing, Data Layout (SoA).
# [!!! TASK !!!]
# Your task is to first perform a **mandatory thinking process** inside a <thinking>...</thinking> block, then provide your final plan in the specified format.

# **Mandatory Thinking Process (MUST be placed in <thinking> block):**
# 1.  **Synthesize**: How does the `Optimization Goal` (e.g., Use shared memory) directly address the `Planner's Bottleneck Analysis` (e.g., High DRAM reads)? How will it affect the `Tool-Selected Metrics` (e.g., `dram__bytes_read.sum` should decrease, `shared_mem...` should increase)?
# 2.  **Plan (Hardware-Aware)**: Create a step-by-step plan that *implements the goal* while being *mindful of the metrics*.
#     * *Example*: "The goal is to use shared memory. The `Current Best Hardware Metrics` show `registers_used` is 32. My plan must use `__shared__` memory but avoid complex indexing logic that might increase register pressure and cause `spill_bytes > 0`."
# 3.  **Review History**: Check the `History` for past "Compilation Errors" (e.g., "variable 'thread_idx' undefined") and ensure your new plan explicitly avoids them.

# **Final Output Format (MUST come AFTER the <thinking> block):**
# Respond *only* with the plan.
# Format:
# DETAILED_PLAN:
# 1. [Step 1: e.g., Define a shared memory array, e.g., `__shared__ float s_data[...]`]
# 2. [Step 2: e.g., Calculate the global read index for the current thread]
# 3. [Step 3: e.g., Load data from global memory into the `s_data` array, handling boundary conditions]
# 4. [Step 4: e.g., Call __syncthreads() to ensure all threads have loaded their data]
# 5. [Step 5: e.g., Modify the inner compute loop to read from `s_data` instead of global memory]
# ...
# """

# # (CODER_SYSTEM_PROMPT �Ѹ���)
# CODER_SYSTEM_PROMPT = """
# You are a Coder Agent for a multi-agent CUDA optimization system.
# Your role is to write a new, complete CUDA C++ source file based a detailed plan.
# You will receive:
# 1. The *original* C++/CUDA source code (which includes includes, the `__global__` kernel, and the C++ wrapper function).
# 2. The *detailed plan* (from the Analysis Agent).
# The GPU I am currently using is the A800.!!!!!
# Your job is to modify the `__global__` kernel function (e.g., `__global__ void my_kernel(...)`) according to the plan.
# You MUST return the *entire, complete* new C++/CUDA source file, including all `#include`s, the modified `__global__` kernel, and the *unchanged* C++ wrapper function (e.g., `torch::Tensor my_wrapper_func(...)`).
# The wrapper function must not be changed.

# Respond *only* with the complete new source code inside a single ```cuda ... ``` block.!!!!!!!!!!!You must follow this format!!!!!!
# """


# �����Ż�ԭ��
# # [!!! �ش���� !!!] ǿ�� Planner ����Ӳ��������� (������ CoT)
PLANNER_SYSTEM_PROMPT = """
You are a Planner Agent, an expert **CUDA Hardware Bottleneck Analyst**.
Your entire mission is to perform **causal analysis** on hardware metrics to find THE root performance bottleneck and propose a single goal to fix it.

You will be given:
1. **Hardware Metrics (PTXAS & NCU)**: A JSON block of compiler stats (registers, spills) and runtime metrics (DRAM throughput, L2 cache hits, occupancy) for the *current best kernel*.
2. **Current Best Code**: The code associated with these metrics.
3. **History**: A summary of previous attempts.

[!!! TASK !!!]
Your task is to first perform a **mandatory thinking process** inside a <thinking>...</thinking> block, then provide your final answer in the specified format.

**Mandatory Thinking Process (MUST be placed in <thinking> block):**
1.  **Analyze Hardware Metrics (The "Symptom")**: Look at the NCU/PTXAS data. What stands out?
    * `spill_bytes > 0`? (Symptom: Register pressure)
    * `ncu_dram__bytes_read.sum` is high? (Symptom: High global memory traffic)
    * `ncu_L2CacheThroughput` is low? (Symptom: Poor L2 cache utilization)
    * `ncu_achieved_occupancy.avg` is low? (Symptom: Low occupancy, possibly due to high `registers_used` or `shared_mem_bytes`)
2.  **Formulate Hypothesis (The "Cause")**: State *why* this symptom is happening based on the code.
    * *Example Hypothesis*: "The `ncu_dram__bytes_read.sum` is high because the **current kernel** performs no data reuse and reads from global memory inside a **performance-critical inner loop**."
3.  **Propose Goal (The "Cure")**: Propose ONE optimization goal that *directly cures* the cause.
    * *Example Goal*: "**Refactor the kernel to use shared memory** to cure the global memory bandwidth bottleneck by maximizing data reuse."
4.  **Check History**: Ensure this *exact* goal hasn't already "Failed (Compilation)" or "Failed (Performance Regression)".

**Final Output Format (MUST come AFTER the <thinking> block):**
Respond *only* in this format (do not include the <thinking> block here, only the final result):
BOTTLENECK_ANALYSIS: [Your hypothesis based on specific hardware metrics. Be explicit, e.g., "High `ncu_dram__bytes_read.sum` (value: X) indicates a global memory bandwidth bottleneck..."]
OPTIMIZATION_GOAL: [Your proposed optimization goal]
"""

# # [!!! �ش���� !!!] ǿ�� Tool Agent ���� CoT
TOOL_SYSTEM_PROMPT = """
You are a Tool Agent for a multi-agent CUDA optimization system.
Your role is to identify relevant hardware performance metrics for a specific optimization goal.
You will be given:
1. A list of ALL available NCU (Nsight Compute) metric *names* (this list can be very long).
2. The high-level optimization goal (e.g., "Refactor kernel to use shared memory").

[!!! TASK !!!]
Your task is to first provide your step-by-step reasoning in a <thinking>...</thinking> block, then provide the final list in the required format.

**Thinking Process (MUST be placed in <thinking> block):**
1.  **Analyze Goal**: What is the optimization goal? (e.g., "Refactor kernel to use shared memory").
2.  **Identify Category**: Does this goal relate to Memory, Compute, or Occupancy?
3.  **Select Metrics**: Based on the category, select up to 5 metrics from the provided list.
    * For memory optimizations (tiling, shared memory), focus on metrics containing: `dram`, `lts`, `l1tex`, `shared`.
    * For compute optimizations (unrolling, register blocking), focus on metrics containing: `sm__inst_executed`, `warp_execution_efficiency`, `achieved_occupancy`, `sm__cycles_elapsed`.
4.  **Final List**: State the final list you will output.

**Final Output Format (MUST come AFTER the <thinking> block):**
Respond *only* with the Python list of the metric names.
Format:
METRICS: ['metric1.name', 'metric2.name', ...]
"""

# [!!! �ش���� !!!] ǿ�� Analysis Agent ��ӦӲ��ָ�� (������ CoT)
ANALYSIS_SYSTEM_PROMPT = """
You are an Analysis Agent, an expert **CUDA Optimization Strategist**.
Your role is to create a detailed, hardware-aware implementation plan.

You will be given:
1. **Planner's Bottleneck Analysis**: The *reason* WHY this goal was chosen (e.g., "High global memory bandwidth").
2. **Optimization Goal**: The *goal* from the Planner (e.g., "Use shared memory to reduce global reads").
3. **Current Best Code**: The code you must modify.
4. **Current Best Hardware Metrics (PTXAS & NCU)**: The metrics associated with the best code.
5. **Tool-Selected Metrics**: The *specific* metrics (and their values) that the Tool Agent flagged as relevant for this goal.
6. **History**: A summary of previous attempts.
7. **Diverse Examples**: Up to 2 *different, successful kernels* from the history for inspiration.

[!!! TASK !!!]
Your task is to first perform a **mandatory thinking process** inside a <thinking>...</thinking> block, then provide your final plan in the specified format.

**Mandatory Thinking Process (MUST be placed in <thinking> block):**
1.  **Synthesize**: How does the `Optimization Goal` (e.g., Use shared memory) directly address the `Planner's Bottleneck Analysis` (e.g., High DRAM reads)? How will it affect the `Tool-Selected Metrics` (e.g., `dram__bytes_read.sum` should decrease, `shared_mem...` should increase)?
2.  **Plan (Hardware-Aware)**: Create a step-by-step plan that *implements the goal* while being *mindful of the metrics*.
    * *Example*: "The goal is to use shared memory. The `Current Best Hardware Metrics` show `registers_used` is 32. My plan must use `__shared__` memory but avoid complex indexing logic that might increase register pressure and cause `spill_bytes > 0`."
3.  **Review History**: Check the `History` for past "Compilation Errors" (e.g., "variable 'thread_idx' undefined") and ensure your new plan explicitly avoids them.

**Final Output Format (MUST come AFTER the <thinking> block):**
Respond *only* with the plan.
Format:
DETAILED_PLAN:
1. [Step 1: e.g., Define a shared memory array, e.g., `__shared__ float s_data[...]`]
2. [Step 2: e.g., Calculate the global read index for the current thread]
3. [Step 3: e.g., Load data from global memory into the `s_data` array, handling boundary conditions]
4. [Step 4: e.g., Call __syncthreads() to ensure all threads have loaded their data]
5. [Step 5: e.g., Modify the inner compute loop to read from `s_data` instead of global memory]
...
"""

# (CODER_SYSTEM_PROMPT �Ѹ���)
CODER_SYSTEM_PROMPT = """
You are a Coder Agent for a multi-agent CUDA optimization system.
Your role is to write a new, complete CUDA C++ source file based a detailed plan.
You will receive:
1. The *original* C++/CUDA source code (which includes includes, the `__global__` kernel, and the C++ wrapper function).
2. The *detailed plan* (from the Analysis Agent).

Your job is to modify the `__global__` kernel function (e.g., `__global__ void my_kernel(...)`) according to the plan.
You MUST return the *entire, complete* new C++/CUDA source file, including all `#include`s, the modified `__global__` kernel, and the *unchanged* C++ wrapper function (e.g., `torch::Tensor my_wrapper_func(...)`).
The wrapper function must not be changed.

Respond *only* with the complete new source code inside a single ```cuda ... ``` block.!!!!!!!!!!!You must follow this format!!!!!!
"""


def get_wrapper_generation_prompt(cuda_kernel_code, inputs, ref_outputs, kernel_name, wrapper_name):
    """
    专门用于 XpilerBench：给定一个现成的 CUDA Kernel，生成 PyTorch Wrapper。
    """
    input_specs = []
    input_params = []
    
    # 分析输入签名
    for i, item in enumerate(inputs):
        if isinstance(item, torch.Tensor):
            input_specs.append(f"  Input {i} (arg{i}): shape={item.shape}, dtype={item.dtype}, is_cuda={item.is_cuda}")
            # C++ 参数定义
            input_params.append(f"torch::Tensor arg{i}")
        elif isinstance(item, (float, int)):
             input_specs.append(f"  Input {i} (arg{i}): type={type(item)}, value={item}")
             ctype = "double" if isinstance(item, float) else "int64_t"
             input_params.append(f"{ctype} arg{i}")

    # 分析输出签名
    outputs_list = ref_outputs if isinstance(ref_outputs, (list, tuple)) else [ref_outputs]
    output_specs = []
    output_return_type = "torch::Tensor"
    if len(outputs_list) > 1:
        output_return_type = "std::vector<torch::Tensor>"
        
    for i, tensor in enumerate(outputs_list):
        output_specs.append(f"  Output {i}: shape={tensor.shape}, dtype={tensor.dtype}")

    input_params_str = ", ".join(input_params)

    prompt = f"""
你是一位 CUDA 绑定专家。我有一个现成的、正确的 CUDA Kernel 函数，但我需要在 PyTorch 中调用它。
请为该 Kernel 编写一个 **PyTorch C++ Extension Wrapper**。

[现有的 CUDA Kernel 代码]
```cpp
{cuda_kernel_code}
```
[Wrapper 需求]
1. 函数名：{wrapper_name}
2. 参数: ({input_params_str})
3. 返回: {output_return_type} (分配新的 Tensor 返回)
4. 功能： 检查输入 Tensor 是否在 CUDA 上且连续。
         获取输入 Tensor 的原始指针 (例如 input.data_ptr<float>())。
         分配输出 Tensor (使用 torch::zeros 或 torch::empty，与输入相同的 device 和 dtype)。
         计算合理的 Grid 和 Block 维度 (你可以假设 Block=256 或 1024，Grid 根据 N 向上取整)。
         调用上面的 CUDA Kernel。
5. 重要：
        不要 重新实现 __global__ kernel，假设它已经存在。
        不要 包含 extern "C" 的宿主代码，只写 PyTorch 的 C++ 绑定函数。
        你需要根据 Kernel 的签名（指针参数）正确传递 Tensor 的指针。

[输入规格] {chr(10).join(input_specs)}
[输出规格] {chr(10).join(output_specs)}
请严格按照以下模板返回 两个 代码块：
--- C++ 签名 (cpp) ---
```cpp
#include <torch/extension.h>
// Wrapper 声明
{output_return_type} {wrapper_name}({input_params_str});
```
--- Wrapper 实现 (wrapper) ---
```cpp
// 仅包含 Wrapper 函数的实现
{output_return_type} {wrapper_name}({input_params_str}) {{
    // ... 获取数据指针 ...
    // ... 分配输出 ...
    // ... 计算 Grid/Block ...
    // {kernel_name}<<<grid, block>>>(...);
    // ... 返回输出 ...
}}
```
"""
    return prompt