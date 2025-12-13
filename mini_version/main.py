import config
import kernels
import cuda_utils
import llm_api as agents 
import prompts            
import torch
from tqdm import tqdm
import os
import re
import ast
import sys
import json 
from typing import List, Dict # [!!! 新增 !!!]
import gc

def extract_code(response_text):
    """(此函数保持不变)"""
    if not response_text: return None 
    match = re.search(r'```cuda\n(.*?)```', response_text, re.DOTALL)
    if not match:
        if "torch::Tensor gemm_cuda" in response_text: 
             return response_text
        print("[Coder Agent] Error: No CUDA code block found in response.")
        return None
            
    return match.group(1).strip()

def extract_metrics(response_text):
    """(此函数保持不变)"""
    if not response_text: return None 
    try:
        metrics_list_str = response_text.split("METRICS:")[1].strip()
        metrics_list = ast.literal_eval(metrics_list_str) 
        return metrics_list
    except Exception as e:
        print(f"[Tool Agent] Error parsing metrics list: {e}\nResponse was: {response_text}")
        return None

# [!!! 已更新 !!!] 解决了 TODO 问题 5（根据你的最新要求）
def get_diverse_champions(history: list, current_best_code: str, num_kernels=2) -> str:
    """
    (此函数保持不变)
    """
    
    # 1. 查找所有成功的条目 (不包括 Round 0)
    success_entries = [
        h for h in history 
        if "Success" in h['status'] and h['round'] > 0 and h.get('code')
    ]
    
    # 2. 按性能排序
    success_entries.sort(key=lambda x: x['time_ms'])
    
    diverse_str = "--- Diverse Successful Kernel Examples (Best first) ---\n"
    count = 0
    
    # 3. 提取代码 (确保它与当前最佳代码 *不同*)
    for entry in success_entries:
        if entry['code'] == current_best_code:
            continue # 跳过与当前最佳完全相同的代码
            
        diverse_str += f"\n\n--- Example {count+1} (From Round {entry['round']}) ---\n"
        diverse_str += f"// Goal: {entry['goal']}\n"
        diverse_str += f"// Performance: {entry['time_ms']:.3f} ms\n"
        
        # 添加 PTXAS 指标
        ptxas = entry.get('ptxas_metrics', {})
        diverse_str += f"// Registers: {ptxas.get('registers_used', 'N/A')}\n"
        diverse_str += f"// Shared Mem: {ptxas.get('shared_mem_bytes', 'N/A')}\n"
        
        # [!!! 新增 !!!] 仅添加该轮选择的 NCU 指标
        selected_metrics = entry.get('selected_ncu_metrics')
        all_ncu = entry.get('all_ncu_metrics')
        
        if isinstance(selected_metrics, list) and isinstance(all_ncu, dict) and selected_metrics:
            diverse_str += f"// Selected NCU Metrics (for Goal):\n"
            for metric_name in selected_metrics:
                value = all_ncu.get(metric_name, 'N/A')
                diverse_str += f"//  - {metric_name}: {value}\n"
        # [!!! 结束新增 !!!]

        diverse_str += entry['code']
        count += 1

        if count >= num_kernels:
            break
            
    if count == 0:
        return "No other diverse successful examples available in history."
    return diverse_str

# [!!! 已更新 !!!] 解决了 TODO 问题 5（根据你的最新要求）
def summarize_history(history: list) -> str:
    """
    (此函数保持不变)
    """
    if not history:
        return "No previous attempts."
    
    summary = "Previous Optimization Attempts:\n"
    for i, entry in enumerate(history):
        summary += f"  Round {entry['round']}:\n"
        summary += f"    Goal: {entry['goal']}\n"
        summary += f"    Status: {entry['status']}\n"
        
        perf_str = "N/A"
        if entry['time_ms'] is not None:
            perf_str = f"{entry['time_ms']:.3f} ms"
        summary += f"    Performance: {perf_str}\n"

        # 添加 PTXAS 指标
        if entry.get('ptxas_metrics'):
            summary += f"    Registers: {entry['ptxas_metrics'].get('registers_used', 'N/A')}\n"
            summary += f"    Shared Mem: {entry['ptxas_metrics'].get('shared_mem_bytes', 'N/A')} bytes\n"

        # [!!! 新增 !!!] 仅添加该轮选择的 NCU 指标
        selected_metrics = entry.get('selected_ncu_metrics')
        all_ncu = entry.get('all_ncu_metrics')
        
        # 检查 'selected_metrics' 是否是列表，'all_ncu' 是否是字典，并且 'selected_metrics' 不为空
        if isinstance(selected_metrics, list) and isinstance(all_ncu, dict) and selected_metrics:
            summary += f"    Selected NCU Metrics (for Goal):\n"
            for metric_name in selected_metrics:
                value = all_ncu.get(metric_name, 'N/A')
                summary += f"      - {metric_name}: {value}\n"
        # [!!! 结束新增 !!!]

        elif "Error" in entry['status'] or "Failed" in entry['status']:
            details = entry.get('details', 'No details')
            if len(details) > 200:
                details = details[:200] + "..."
            summary += f"    Error Details: {details}\n"
    return summary

# [!!! 已更新 !!!] 解决了 TODO 问题 7 (来自上一个请求)
def format_metrics_for_llm(ptxas_metrics: dict, ncu_metrics: dict) -> str:
    """
    (此函数保持不变)
    """
    if not ncu_metrics:
        return "Hardware metrics are not yet available."
        
    summary = "=== PTXAS Compiler Metrics ===\n"
    summary += json.dumps(ptxas_metrics, indent=2)
    
    # [!!! 更改 !!!] 直接使用完整的 ncu_metrics 字典，并将标题更改为 "Full Set"
    summary += "\n\n=== NCU Hardware Metrics (Full Set) ===\n" 
    summary += json.dumps(ncu_metrics, indent=2)
    
    return summary


# [!!! 已更新 !!!] main() 已重构为通用函数
# def run_optimization_on_problem(
#     problem_name: str,
#     cpp_source: str, 
#     initial_cuda_code: str, 
#     inputs: List[torch.Tensor], 
#     ref_outputs: List[torch.Tensor],
#     kernel_name: str,           # __global__ 函数名
#     wrapper_function_name: str, # C++ wrapper 函数名
#     iteration_rounds: int,
#     history_file_path: str,
#     baseline_time_ms: float = float('inf') # [!!! 已修改 !!!] 接收Pytorch基准
# ):
def run_optimization_on_problem(
    problem_name,
    initial_cuda_code,
    inputs, 
    init_inputs,
    ref_outputs,
    # kernel_name: str,           # __global__ 函数名
    # wrapper_function_name: str, # C++ wrapper 函数名
    iteration_rounds,
    history_file_path,
    baseline_time_ms # [!!! 已修改 !!!] 接收Pytorch基准
):
    """
    运行通用的、线性的多智能体优化循环。
    """
    
    print(f"Starting optimization for problem: {problem_name}")
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 CUDA。无法进行本地测试。")
        return {"error": "No CUDA detected"}
        
    print(f"Running on device: {config.DEVICE}")
    print(f"Total iteration rounds: {iteration_rounds}")
    if config.MOCK_LLM_CALLS:
        print("--- 警告: MOCK LLM CALLS ARE ENABLED (in config.py) ---")
    
    # 1. 初始化
    device = torch.device(config.DEVICE)
    
    best_kernel_code_cuda = initial_cuda_code # <--- [!!! 已更新 !!!]
    best_time_ms = float('inf')
    best_ptxas_metrics = {}
    best_ncu_metrics = {}
    current_ncu_metrics = {}
    
    optimization_history = []
    
    # (加载历史记录的代码已更新)
    if os.path.exists(history_file_path): # <--- [!!! 已更新 !!!]
        print(f"Loading existing history from {history_file_path}")
        try:
            with open(history_file_path, 'r', encoding='utf-8') as f: # <--- [!!! 已更新 !!!]
                optimization_history = json.load(f)
            
            found_best = False
            for entry in sorted(optimization_history, key=lambda x: x.get('time_ms') if x.get('time_ms') is not None else float('inf')):
                 if ("Success" in entry['status']) and entry.get('code'):
                    best_time_ms = entry['time_ms']
                    best_ptxas_metrics = entry['ptxas_metrics']
                    best_kernel_code_cuda = entry['code'] # <--- 从历史中恢复代码
                    best_ncu_metrics = entry.get('all_ncu_metrics', {}) 
                    
                    print(f"Restored best kernel from history (Round {entry['round']}, Time: {best_time_ms:.3f} ms)")
                    found_best = True
                    break
            if not found_best:
                 print("No successful kernel found in history, starting from baseline.")
                 optimization_history = [] 
        except json.JSONDecodeError:
            print(f"Warning: Corrupt history file {history_file_path}. Starting from scratch.")
            optimization_history = []
             
    # 2. 获取基线性能 (Round 0)
    if not optimization_history: 
        print("\n--- Round 0: Compiling and analyzing baseline (naive) kernel ---")
        current_module_name = f"{problem_name}_0" # <--- [!!! 已更新 !!!]
        try:
            module, stdout_log, stderr_log = cuda_utils.load_module(
                # cpp_source, 
                best_kernel_code_cuda, 
                current_module_name,
                init_inputs
                # wrapper_function_name=wrapper_function_name # <--- [!!! 已更新 !!!]
            )
            print("Baseline kernel compiled successfully.")
            best_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log)
            
            # [!!! 已更新 !!!]
            is_correct = cuda_utils.check_correctness(inputs, ref_outputs, module)
            if not is_correct:
                print("❌ Baseline kernel is INCORRECT. Exiting.")
                return {"error": "Baseline kernel incorrect."} # <--- [!!! 已更新 !!!]
                
            print("Baseline kernel is correct. Benchmarking...")
            # [!!! 已更新 !!!]
            best_time_ms = cuda_utils.benchmark_kernel(inputs, module)
            
            print("Analyzing baseline kernel with NCU (this may take a while)...")
            # [!!! 已更新 !!!]
            best_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, 
                current_module_name, 
                # kernel_name,           # <--- [!!! 已更新 !!!]
                # wrapper_function_name, # <--- [!!! 已更新 !!!]
                inputs                 # <--- [!!! 已更新 !!!]
            )
            current_ncu_metrics = best_ncu_metrics # liuxitai:到这里目前已经改好了
            
            history_entry = {
                "round": 0, "goal": "Baseline", "status": "Success",
                "time_ms": best_time_ms, 
                "ptxas_metrics": best_ptxas_metrics,
                "all_ncu_metrics": best_ncu_metrics,
                "selected_ncu_metrics": [],
                "details": "Initial baseline measurement",
                "code": best_kernel_code_cuda 
            }
            optimization_history.append(history_entry)
            print(f"Baseline performance: {best_time_ms:.3f} ms")

            try:
                # 1. 保存 History
                with open(history_file_path, 'w', encoding='utf-8') as f:
                    json.dump(optimization_history, f, indent=2)

                # 2. 保存 Best Kernel
                # 既然 Round 0 成功了且是当前唯一的内核，它就是 Best Kernel
                best_kernel_path = history_file_path.replace(
                    "_optimization_history.json", 
                    "_best_kernel.cu"
                )
                with open(best_kernel_path, "w", encoding='utf-8') as f:
                    f.write(cpp_source + "\n\n" + best_kernel_code_cuda)
                print(f"✅ Real-time save (Round 0): Initial kernel saved to {best_kernel_path}")

                # 3. 保存 Best Stats
                # 如果传入了 baseline_time_ms，则计算加速比并保存
                if baseline_time_ms != float('inf') and best_time_ms > 0:
                    speedup = baseline_time_ms / best_time_ms
                    stats = {
                        "problem_name": problem_name,
                        "baseline_ms": baseline_time_ms,
                        "best_cuda_ms": best_time_ms,
                        "speedup": speedup,
                        "last_updated_round": 0
                    }
                    stats_file_path = history_file_path.replace(
                        "_optimization_history.json",
                        "_best_stats.json"
                    )
                    with open(stats_file_path, "w", encoding='utf-8') as f:
                        json.dump(stats, f, indent=2)
                    print(f"✅ Real-time save (Round 0): Initial stats saved to {stats_file_path} (Speedup: {speedup:.2f}x)")
            
            except Exception as e:
                print(f"[Warning] Round 0: Failed to real-time save initial results: {e}")

        except Exception as e:
            print(f"❌ Baseline kernel failed compilation or runtime. Exiting. \n{e}")
            return {"error": f"Baseline failed: {e}"} # <--- [!!! 已更新 !!!]
    
    if not current_ncu_metrics: 
        current_ncu_metrics = best_ncu_metrics if best_ncu_metrics else {}


    # 3. 开始优化循环
    # [!!! 已更新 !!!]
    for i in tqdm(range(len(optimization_history), iteration_rounds + 1), desc="Optimization Rounds"):
        if i == 0: continue # Round 0 已经完成
        
        print(f"\n--- Round {i}/{iteration_rounds} ---")
        
        history_summary = summarize_history(optimization_history)
        metrics_summary = format_metrics_for_llm(best_ptxas_metrics, best_ncu_metrics)
        
        print("------------------LXT:metrics_summary (to Planner)----------------------")
        print(metrics_summary)
        print("------------------LXT:metrics_summary (to Planner)----------------------")
        
        opt_goal = "N/A"
        bottleneck_analysis = "N/A" 
        detailed_plan = "N/A"
        new_kernel_code_full = None # [!!! 已更新 !!!]
        new_kernel_code_cuda_only = None # [!!! 已更新 !!!]
        status = "Failed (Unknown)"
        details = ""
        new_time_ms = float('inf')
        new_ptxas_metrics = {}
        new_ncu_metrics = {}
        relevant_metric_names = [] 
        
        try:
            # 1. Planner Agent
            print("[Planner Agent] Analyzing hardware metrics and history...")
            
            # [!!! 已更新 !!!] 合并 C++ 和 CUDA 以获取完整上下文
            parent_kernel_code = cpp_source + "\n\n" + best_kernel_code_cuda
            
            planner_response = agents.call_llm(
                "planner", 
                prompts.PLANNER_SYSTEM_PROMPT,
                f"Optimization History:\n{history_summary}\n\n"
                f"=== Hardware Metrics for Current Best Kernel ===\n{metrics_summary}\n\n"
                f"Current Best C++/CUDA Source (Time: {best_time_ms:.3f} ms):\n{parent_kernel_code}" # <--- [!!! 已更新 !!!]
            )
            if not planner_response or "OPTIMIZATION_GOAL:" not in planner_response:
                status, details = "Failed (Planner)", "Planner did not return a valid goal."
                print(f"❌ {status} {details}")
                continue 
            
            if "BOTTLENECK_ANALYSIS:" in planner_response:
                 bottleneck_analysis = planner_response.split("BOTTLENECK_ANALYSIS:")[1].split("OPTIMIZATION_GOAL:")[0].strip()
                 print(f"[Planner Agent] Bottleneck identified: {bottleneck_analysis}")
            else:
                status, details = "Failed (Planner)", "Planner did not output BOTTLENECK_ANALYSIS."
                print(f"❌ {status} {details}")
                continue
                 
            opt_goal = planner_response.split("OPTIMIZATION_GOAL:")[1].strip()
            print(f"[Planner Agent] Goal: {opt_goal}")
            print("-----------------------LXT:planner_response----------------------")
            print(planner_response)
            print("-----------------------LXT:planner_response----------------------")
            
            # 2. Tool Agent
            print("[Tool Agent] Selecting metrics...")
            all_metric_names = list(current_ncu_metrics.keys())
            print("-----------------------LXT:all_metric_names----------------------")
            print(all_metric_names)
            print("-----------------------LXT:all_metric_names----------------------")
            if not all_metric_names:
                all_metric_names = config.BASE_NCU_METRICS_LIST_EXAMPLE
                
            tool_response = agents.call_llm(
                "tool", 
                prompts.TOOL_SYSTEM_PROMPT,
                f"All Available NCU Metric Names ({len(all_metric_names)}): {all_metric_names}\n\nOptimization Goal: {opt_goal}"
            )
            print("-----------------------LXT:tool_response----------------------")
            print(tool_response)
            print("-----------------------LXT:tool_response----------------------")
            
            relevant_metric_names = extract_metrics(tool_response)
            
            if not relevant_metric_names:
                status, details = "Failed (Tool)", "Tool Agent did not return a valid metric list."
                print(f"❌ {status} {details}")
                continue 
            print(f"[Tool Agent] Selected {len(relevant_metric_names)} metrics: {relevant_metric_names}")
            
            relevant_metrics_dict = {
                metric: current_ncu_metrics.get(metric, 0.0) 
                for metric in relevant_metric_names
            }
            
            diverse_kernels_str = get_diverse_champions(optimization_history, best_kernel_code_cuda)
            
            # 3. Analysis Agent [!!! 已更新 !!!]
            print("[Analysis Agent] Formulating plan...")
            analysis_response = agents.call_llm(
                "analysis", 
                prompts.ANALYSIS_SYSTEM_PROMPT,
                f"Planner's Bottleneck Analysis: {bottleneck_analysis}\n\n"
                f"Optimization Goal: {opt_goal}\n\n"
                f"Optimization History:\n{history_summary}\n\n"
                f"Diverse Successful Kernel Examples:\n{diverse_kernels_str}\n\n"
                f"Current Best C++/CUDA Source:\n{parent_kernel_code}\n\n" # <--- [!!! 已更新 !!!]
                f"Current Best Hardware Metrics (Full Set): {metrics_summary}\n\n"
                f"Tool-Selected Metrics from *Previous* Run (Values): {relevant_metrics_dict}" 
            )
            print("-----------------------LXT:analysis_response----------------------")
            print(analysis_response)
            print("-----------------------LXT:analysis_response----------------------")
            if not analysis_response or "DETAILED_PLAN:" not in analysis_response:
                status, details = "Failed (Analysis)", "Analysis Agent did not return a valid plan."
                print(f"❌ {status} {details}")
                continue 
            detailed_plan = analysis_response.split("DETAILED_PLAN:")[1].strip()

            # 4. Coder Agent
            print("[Coder Agent] Generating new kernel...")
            coder_response = agents.call_llm(
                "coder", 
                prompts.CODER_SYSTEM_PROMPT,
                f"Original C++/CUDA Source:\n{parent_kernel_code}\n\nDetailed Plan:\n{detailed_plan}" # <--- [!!! 已更新 !!!]
            )
            print("-----------------------LXT:coder_response----------------------")
            print(coder_response)
            print("-----------------------LXT:coder_response----------------------")
            
            new_kernel_code_full = extract_code(coder_response) # <--- [!!! 已更新 !!!]
            
            if not new_kernel_code_full: # <--- [!!! 已更新 !!!]
                status, details = "Failed (Coder)", "Coder Agent did not produce valid code."
                print(f"❌ {status} {details}")
                continue 
            print("[Coder Agent] New kernel source generated.")
                
            # 5. 验证和分析
            current_module_name = f"{problem_name}_{i}" # <--- [!!! 已更新 !!!]
            print(f"Compiling new kernel (module: {current_module_name})...")
            
            try:
                # [!!! 已更新 !!!] 假设 Coder 返回 C++ 和 CUDA
                module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
                    cpp_source, 
                    new_kernel_code_full, # <--- [!!! 已更新 !!!]
                    module_name=current_module_name,
                    wrapper_function_name=wrapper_function_name # <--- [!!! 已更新 !!!]
                )
                print("Compilation successful.")
                new_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
                
                # [!!! 已更新 !!!]
                is_correct = cuda_utils.check_correctness(inputs, ref_outputs, wrapper_function_name)
                if not is_correct:
                    status, details = "Failed (Correctness)", "New kernel is INCORRECT."
                    print(f"❌ {status}")
                    continue 
                    
            except Exception as e:
                status, details = "Failed (Compilation)", str(e)
                print(f"❌ {status}")
                continue 
                
            print("New kernel is CORRECT. Benchmarking...")
            
            # [!!! 已更新 !!!]
            new_time_ms = cuda_utils.benchmark_kernel(inputs, wrapper_function_name)
            print("Analyzing new kernel with NCU...")
            
            # [!!! 已更新 !!!]
            new_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, 
                current_module_name, 
                kernel_name,           # <--- [!!! 已更新 !!!]
                wrapper_function_name, # <--- [!!! 已更新 !!!]
                inputs                 # <--- [!!! 已更新 !!!]
            )
            
            if new_time_ms < best_time_ms:
                status = "Success (New Best)"
                details = f"Performance improved from {best_time_ms:.3f} ms to {new_time_ms:.3f} ms."
                print(f"✅ {status} {details}")
                
                best_time_ms = new_time_ms
                
                # [!!! 已更新 !!!] 提取 CUDA-only 代码以供下次迭代
                if new_kernel_code_full.startswith(cpp_source):
                    new_kernel_code_cuda_only = new_kernel_code_full[len(cpp_source):].strip()
                else:
                    match = re.search(r'__global__\s+void\s+' + kernel_name + r'\(.*?\)\s*\{.*\}', new_kernel_code_full, re.DOTALL)
                    if match:
                        new_kernel_code_cuda_only = match.group(0)
                    else:
                        new_kernel_code_cuda_only = new_kernel_code_full # 无法分离
                        print(f"[Warning] Round {i}: Could not auto-extract kernel. Saving full code.")

                best_kernel_code_cuda = new_kernel_code_cuda_only # <--- [!!! 已更新 !!!]
                best_ptxas_metrics = new_ptxas_metrics
                best_ncu_metrics = new_ncu_metrics

                # [!!! 已修改 !!!] (来自你之前的请求) 实时保存最佳内核
                try:
                    # 从 history_file_path 推导最佳内核路径
                    # (例如: ".../100_HingeLoss_optimization_history.json" 
                    # 变为 ".../100_HingeLoss_best_kernel.cu")
                    best_kernel_path = history_file_path.replace(
                        "_optimization_history.json", 
                        "_best_kernel.cu"
                    )
                    
                    # 保存完整的 C++ 包装器 + 优化的 CUDA 内核
                    with open(best_kernel_path, "w", encoding='utf-8') as f:
                        f.write(cpp_source + "\n\n" + best_kernel_code_cuda)
                    print(f"✅ Real-time save: New best kernel saved to {best_kernel_path}")
                
                except Exception as e:
                    print(f"[Warning] Round {i}: Failed to real-time save best kernel: {e}")
                # [!!! 结束修改 !!!]

                # [!!! 新增 3 !!!] 实时保存最优统计数据 (ms 和 speedup)
                if baseline_time_ms != float('inf') and best_time_ms > 0:
                    try:
                        speedup = baseline_time_ms / best_time_ms
                        stats = {
                            "problem_name": problem_name,
                            "baseline_ms": baseline_time_ms,
                            "best_cuda_ms": best_time_ms,
                            "speedup": speedup,
                            "last_updated_round": i
                        }
                        
                        # 从 history_file_path 推导
                        stats_file_path = history_file_path.replace(
                            "_optimization_history.json",
                            "_best_stats.json"
                        )
                        
                        with open(stats_file_path, "w", encoding='utf-8') as f:
                            json.dump(stats, f, indent=2)
                        print(f"✅ Real-time save: New best stats saved to {stats_file_path} (Speedup: {speedup:.2f}x)")
                            
                    except Exception as e:
                        print(f"[Warning] Round {i}: Failed to real-time save stats: {e}")
                # [!!! 新增结束 !!!]

            else:
                status = "Failed (Performance Regression)"
                details = f"New time {new_time_ms:.3f} ms is not better than best time {best_time_ms:.3f} ms."
                print(f"❌ {status} {details}")
            
            current_ncu_metrics = new_ncu_metrics

        except Exception as e:
            status, details = "Failed (Unhandled Exception)", str(e)
            print(f"❌ {status} {details}")
            
        finally:
            # ================= [!!! 重点修改：强力回收当前轮次资源 !!!] =================
            print(f"--- Cleaning up resources for Round {i} ---")
            try:
                # 1. 清除本轮编译生成的 JIT 模块缓存 (关键)
                # 这一步会释放加载的 .so 文件句柄和相关显存引用
                cuda_utils._gemm_module = None
                
                # 2. 显式删除本轮产生的中间大对象
                # 使用 locals().get() 或检查变量是否存在，防止因报错导致变量未定义
                if 'module' in locals() and module is not None:
                    del module
                if 'new_kernel_code_full' in locals(): 
                    # 注意：如果 history_entry 需要用到代码字符串，请确保先保存到 history_entry 再 del
                    # 或者依赖 Python 的引用计数（只要 history_entry 引用了，del 局部变量也没事）
                    pass 

                # 3. 强制 Python 垃圾回收 (回收 CPU 内存对象)
                gc.collect()
                
                # 4. 强制 PyTorch 清空 CUDA 缓存 (回收 GPU 显存)
                # [核心逻辑]：这里包裹 try...except，防止 "misaligned address" 等严重错误
                # 导致 empty_cache() 抛出异常从而中断循环。
                torch.cuda.empty_cache()
                
            except Exception as cleanup_err:
                # 吞掉清理过程中的错误，保证绝对能进入下一轮
                print(f"[Warning] Cleanup failed in Round {i} (Ignored to continue loop): {cleanup_err}")
            
            # [!!! 已更新 !!!] 提取 CUDA-only 代码以保存到历史
            code_to_save = ""
            if new_kernel_code_full: # 仅当 Coder 成功时
                if new_kernel_code_cuda_only: # 如果在 'Success' 块中已提取
                    code_to_save = new_kernel_code_cuda_only
                # 否则，再次尝试提取（以防失败）
                elif new_kernel_code_full.startswith(cpp_source):
                    code_to_save = new_kernel_code_full[len(cpp_source):].strip()
                else:
                    match = re.search(r'__global__\s+void\s+' + kernel_name + r'\(.*?\)\s*\{.*\}', new_kernel_code_full, re.DOTALL)
                    if match: 
                        code_to_save = match.group(0)
                    else: 
                        code_to_save = new_kernel_code_full # 无法分离，保存全部
            
            history_entry = {
                "round": i,
                "goal": opt_goal,
                "status": status,
                "time_ms": new_time_ms if new_time_ms != float('inf') else None,
                "ptxas_metrics": new_ptxas_metrics,
                "all_ncu_metrics": new_ncu_metrics,
                "selected_ncu_metrics": relevant_metric_names,
                "details": details,
                "code": code_to_save if code_to_save else (new_kernel_code_full if new_kernel_code_full else "") 
            }
            optimization_history.append(history_entry)

            # [!!! 已修改 !!!] (来自你之前的请求) 实时保存历史
            try:
                with open(history_file_path, 'w', encoding='utf-8') as f:
                    json.dump(optimization_history, f, indent=2)
            except Exception as e:
                print(f"[Warning] Round {i}: Failed to real-time save history: {e}")
            # [!!! 结束修改 !!!]

    # 4. 最终报告
    print("\n--- Optimization Finished ---")
    if optimization_history:
        print(f"Baseline performance (Round 0): {optimization_history[0].get('time_ms', 0.0):.3f} ms")
    print(f"Best kernel performance: {best_time_ms:.3f} ms")
    
    # [!!! 注意 !!!] 
    # 最终保存 .cu 和 .json 的代码保留在此处，
    # 作为"最终"状态的保证，
    # 即使实时保存失败。
    
    final_kernel_path = history_file_path.replace(
        "_optimization_history.json", 
        "_best_kernel.cu"
    )
    with open(final_kernel_path, "w", encoding='utf-8') as f:
        f.write(cpp_source + "\n\n" + best_kernel_code_cuda) # <--- [!!! 已更新 !!!]
    print(f"Best kernel C++/CUDA source saved to {final_kernel_path}")
    
    with open(history_file_path, 'w') as f: # <--- [!!! 已更新 !!!]
        json.dump(optimization_history, f, indent=2)
    print(f"Optimization history saved to {history_file_path}")
    
    # 5. [!!! 已更新 !!!] 返回最佳节点
    best_entry = None
    if best_time_ms != float('inf'):
         best_entry = next((h for h in reversed(optimization_history) if h.get('time_ms') == best_time_ms), None)
    
    if not best_entry: # 如果没有成功的，返回 Baseline
         best_entry = optimization_history[0] if optimization_history else {"error": "No history found."}
         
    return best_entry

# [!!! 新增 !!!] 
# 添加新的 main() 和 if __name__ == "__main__": 
# 以调用通用循环，实现后向兼容
# def main():
#     """
#     为原始 GEMM 问题设置参数，并调用通用的优化器。
#     """
#     print("--- Running Original GEMM Problem (Backward-Compatibility) ---")
    
#     # 1. 设置原始的 GEMM 问题参数
#     N = 8192 # <--- [!!! 新增 !!!] 为 GEMM 定义 N
#     # (如果 config.py 中存在 MATRIX_N，则覆盖)
#     if hasattr(config, 'MATRIX_N'): 
#         N = config.MATRIX_N
            
#     device = torch.device(config.DEVICE)
#     torch.manual_seed(42)
#     A_torch = torch.randn((N, N), dtype=torch.float32, device=device)
#     B_torch = torch.randn((N, N), dtype=torch.float32, device=device)
#     C_ref_torch = torch.matmul(A_torch, B_torch) 
    
#     inputs = [A_torch, B_torch]
#     ref_outputs = [C_ref_torch]
    
#     problem_name = "gemm_N" + str(N)
#     cpp_source = kernels.CPP_SOURCE
#     initial_cuda_code = kernels.NAIVE_CUDA_SOURCE # <--- 使用 kernels.py 中的基线
#     kernel_name = "gemm_kernel" # __global__ name
#     wrapper_function_name = "gemm_cuda" # C++ wrapper name
#     iteration_rounds = config.ITERATION_ROUNDS
#     history_file_path = config.HISTORY_FILE # 使用 config 中的默认历史文件

#     # 2. 调用通用优化器
#     # [!!! 注意 !!!] 
#     # 此处未传递 baseline_time_ms，
#     # 因为旧的 main() 
#     # 在最后才计算它。
#     # 这意味着 _best_stats.json 
#     # 不会在此模式下生成，这是正常的。
#     best_node = run_optimization_on_problem(
#         problem_name=problem_name,
#         cpp_source=cpp_source,
#         initial_cuda_code=initial_cuda_code,
#         inputs=inputs,
#         ref_outputs=ref_outputs,
#         kernel_name=kernel_name,
#         wrapper_function_name=wrapper_function_name,
#         iteration_rounds=iteration_rounds,
#         history_file_path=history_file_path
#         # baseline_time_ms 
#         # 使用默认的 float('inf')
#     )
    
#     if 'error' in best_node:
#         print(f"GEMM optimization failed: {best_node.get('error', 'Unknown')}")
#         return
    
#     if best_node.get('time_ms') is None or best_node['time_ms'] == float('inf'):
#          print("Optimization finished, but no successful kernel was found.")
#          return

#     # 3. 运行原始的最终基准测试 (来自旧 main())
#     print("\n--- Running Final Benchmark (GEMM vs PyTorch) ---")
    
#     try:
#         pytorch_time_ms = cuda_utils.get_pytorch_performance(A_torch, B_torch)
#         print(f"PyTorch (torch.matmul) performance: {pytorch_time_ms:.3f} ms")
#         print(f"Our best LLM-optimized kernel: {best_node['time_ms']:.3f} ms")
        
#         speedup = pytorch_time_ms / best_node['time_ms']
#         if best_node['time_ms'] < pytorch_time_ms:
#             print(f"SUCCESS: Optimized kernel is {speedup:.2f}x faster than PyTorch!")
#         else:
#             print(f"Result: PyTorch is {1/speedup:.2f}x faster.")
#     except AttributeError:
#          print("Warning: cuda_utils.get_pytorch_performance not found. Skipping final comparison.")
#     except Exception as e:
#          print(f"Error during final benchmark: {e}")


# if __name__ == "__main__":
#     main()