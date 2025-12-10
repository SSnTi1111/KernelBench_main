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
    从历史中提取最多 N 个不同的、成功的内核代码。
    现在在注释中包含该轮 *Tool Agent 选择* 的 NCU 指标及其值。
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
    (此函数已更新)
    现在包含 PTXAS 指标以及 *Tool Agent 当轮选择* 的 NCU 指标及其值。
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
    [!!! 已更新 !!!] 解决了 TODO 问题 7。
    此函数现在动态地将 *所有* 捕获的 NCU 指标传递给 Planner Agent，
    而不是硬编码一个固定的 "Key" 列表。
    """
    if not ncu_metrics:
        return "Hardware metrics are not yet available."
        
    summary = "=== PTXAS Compiler Metrics ===\n"
    summary += json.dumps(ptxas_metrics, indent=2)
    
    # [!!! 更改 !!!] 直接使用完整的 ncu_metrics 字典，并将标题更改为 "Full Set"
    summary += "\n\n=== NCU Hardware Metrics (Full Set) ===\n" 
    summary += json.dumps(ncu_metrics, indent=2)
    
    return summary


def main():
    print(f"Starting GEMM optimization for {config.MATRIX_N}x{config.MATRIX_N} matrix.")
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 CUDA。无法进行本地测试。")
        sys.exit(1)
        
    print(f"Running on device: {config.DEVICE}")
    print(f"Total iteration rounds: {config.ITERATION_ROUNDS}")
    if config.MOCK_LLM_CALLS:
        print("--- 警告: MOCK LLM CALLS ARE ENABLED (in config.py) ---")
    
    # 1. 初始化
    N = config.MATRIX_N
    device = torch.device(config.DEVICE)
    print("Initializing Tensors...")
    torch.manual_seed(42)
    A_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    B_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    print("Running PyTorch baseline (torch.matmul) for reference...")
    C_ref_torch = torch.matmul(A_torch, B_torch) 
    
    cpp_source = kernels.CPP_SOURCE 
    best_kernel_code_cuda = kernels.NAIVE_CUDA_SOURCE
    best_time_ms = float('inf')
    best_ptxas_metrics = {}
    best_ncu_metrics = {}
    current_ncu_metrics = {}
    
    optimization_history = []
    # (加载历史记录的代码保持不变)
    if os.path.exists(config.HISTORY_FILE):
        print(f"Loading existing history from {config.HISTORY_FILE}")
        with open(config.HISTORY_FILE, 'r') as f:
            optimization_history = json.load(f)
        
        found_best = False
        for entry in sorted(optimization_history, key=lambda x: x.get('time_ms', float('inf'))):
             # [!!! 已更新 !!!] 修复了 'code' 键的查找
             if ("Success" in entry['status']) and entry.get('code'):
                best_time_ms = entry['time_ms']
                best_ptxas_metrics = entry['ptxas_metrics']
                best_kernel_code_cuda = entry['code'] # <--- 从历史中恢复代码
                
                # [!!! 已更新 !!!] 从历史中恢复完整的 NCU 指标
                best_ncu_metrics = entry.get('all_ncu_metrics', {}) 
                
                print(f"Restored best kernel from history (Round {entry['round']}, Time: {best_time_ms:.3f} ms)")
                found_best = True
                break
        if not found_best:
             print("No successful kernel found in history, starting from baseline.")
             optimization_history = [] 
             
    # 2. 获取基线性能 (Round 0)
    if not optimization_history: 
        print("\n--- Round 0: Compiling and analyzing baseline (naive) kernel ---")
        current_module_name = "gemm_evolved_0"
        
        try:
            module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
                cpp_source, 
                best_kernel_code_cuda, 
                module_name=current_module_name
            )
            print("Baseline kernel compiled successfully.")
            best_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
            
            is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
            if not is_correct:
                print("❌ Baseline kernel is INCORRECT. Exiting.")
                return
                
            print("Baseline kernel is correct. Benchmarking...")
            best_time_ms = cuda_utils.benchmark_kernel(A_torch, B_torch)
            
            print("Analyzing baseline kernel with NCU (this may take a while)...")
            best_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, current_module_name, N
            )
            current_ncu_metrics = best_ncu_metrics 
            
            # [!!! 已更新 !!!] 解决了 TODO 问题 5
            history_entry = {
                "round": 0, "goal": "Baseline", "status": "Success",
                "time_ms": best_time_ms, 
                "ptxas_metrics": best_ptxas_metrics,
                "all_ncu_metrics": best_ncu_metrics,      # <--- 保存完整的 NCU 指标
                "selected_ncu_metrics": [],                 # <--- 基线没有选择指标
                "details": "Initial baseline measurement",
                "code": best_kernel_code_cuda # 保存基线代码
            }
            optimization_history.append(history_entry)
            print(f"Baseline performance: {best_time_ms:.3f} ms")

        except Exception as e:
            print(f"❌ Baseline kernel failed compilation or runtime. Exiting. \n{e}")
            return
    
    # 确保我们有 Round 0 的指标
    if not current_ncu_metrics: # current_ncu_metrics是所有的指标
        # 如果从历史加载，我们没有“上一轮”的NCU指标，所以我们使用“最佳”的指标
        current_ncu_metrics = best_ncu_metrics if best_ncu_metrics else {}


    # 3. 开始优化循环
    for i in tqdm(range(len(optimization_history), config.ITERATION_ROUNDS + 1), desc="Optimization Rounds"):
        if i == 0: continue # Round 0 已经完成
        
        print(f"\n--- Round {i}/{config.ITERATION_ROUNDS} ---")
        
        # [!!! 已更新 !!!] summarize_history 现在会显示选定的 NCU 指标
        history_summary = summarize_history(optimization_history)
        
        # [!!! 已更新 !!!] format_metrics_for_llm 现在会显示所有 NCU 指标
        metrics_summary = format_metrics_for_llm(best_ptxas_metrics, best_ncu_metrics)
        
        print("------------------LXT:metrics_summary (to Planner)----------------------")
        print(metrics_summary)
        print("------------------LXT:metrics_summary (to Planner)----------------------")
        
        opt_goal = "N/A"
        bottleneck_analysis = "N/A" 
        detailed_plan = "N/A"
        new_kernel_code = None
        status = "Failed (Unknown)"
        details = ""
        new_time_ms = float('inf')
        new_ptxas_metrics = {}
        new_ncu_metrics = {}
        
        # [!!! 已更新 !!!] 在 try 块顶部初始化，以便 finally 块可以使用
        relevant_metric_names = [] 
        
        try:
            # 1. Planner Agent
            print("[Planner Agent] Analyzing hardware metrics and history...")
            planner_response = agents.call_llm(
                "planner", 
                prompts.PLANNER_SYSTEM_PROMPT,
                f"Optimization History:\n{history_summary}\n\n"
                f"=== Hardware Metrics for Current Best Kernel ===\n{metrics_summary}\n\n"
                f"Current Best C++/CUDA Source (Time: {best_time_ms:.3f} ms):\n{best_kernel_code_cuda}"
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
            print(all_metric_names)# 这里是所有的指标
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
            
            # [!!! 已更新 !!!] 捕获 relevant_metric_names 以便保存到历史
            relevant_metric_names = extract_metrics(tool_response)
            
            if not relevant_metric_names:
                status, details = "Failed (Tool)", "Tool Agent did not return a valid metric list."
                print(f"❌ {status} {details}")
                continue 
            print(f"[Tool Agent] Selected {len(relevant_metric_names)} metrics: {relevant_metric_names}")
            
            relevant_metrics_dict = {
                metric: current_ncu_metrics.get(metric, 0.0) 
                for metric in relevant_metric_names
            }# 获取所选五个指标的值
            
            # [!!! 已更新 !!!] get_diverse_champions 现在会显示选定的 NCU 指标
            diverse_kernels_str = get_diverse_champions(optimization_history, best_kernel_code_cuda)# 获取多样性成功案例
            
            # 3. Analysis Agent [!!! 已更新 !!!]
            print("[Analysis Agent] Formulating plan...")
            analysis_response = agents.call_llm(
                "analysis", 
                prompts.ANALYSIS_SYSTEM_PROMPT,
                f"Planner's Bottleneck Analysis: {bottleneck_analysis}\n\n" # <--- 传入瓶颈
                f"Optimization Goal: {opt_goal}\n\n"
                f"Optimization History:\n{history_summary}\n\n" # <--- history_summary 现在包含选定的 NCU 指标
                f"Diverse Successful Kernel Examples:\n{diverse_kernels_str}\n\n" # <--- diverse_kernels_str 现在包含选定的 NCU 指标
                f"Current Best C++/CUDA Source:\n{best_kernel_code_cuda}\n\n"
                f"Current Best Hardware Metrics (Full Set): {metrics_summary}\n\n" # <--- 传入完整最佳指标
                f"Tool-Selected Metrics from *Previous* Run (Values): {relevant_metrics_dict}" # <--- 传入工具选择的指标（这个确实是五个）
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
                f"Original C++/CUDA Source:\n{best_kernel_code_cuda}\n\nDetailed Plan:\n{detailed_plan}"
            )
            print("-----------------------LXT:coder_response----------------------")
            print(coder_response)
            print("-----------------------LXT:coder_response----------------------")
            new_kernel_code = extract_code(coder_response)
            if not new_kernel_code:
                status, details = "Failed (Coder)", "Coder Agent did not produce valid code."
                print(f"❌ {status} {details}")
                continue 
            print("[Coder Agent] New kernel source generated.")
                
            # 5. 验证和分析
            current_module_name = f"gemm_evolved_{i}" 
            print(f"Compiling new kernel (module: {current_module_name})...")
            
            try:
                module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
                    cpp_source, 
                    new_kernel_code, 
                    module_name=current_module_name
                )
                print("Compilation successful.")
                new_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
                
                is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
                if not is_correct:
                    status, details = "Failed (Correctness)", "New kernel is INCORRECT."
                    print(f"❌ {status}")
                    continue 
                    
            except Exception as e:
                status, details = "Failed (Compilation)", str(e)
                print(f"❌ {status}")
                continue 
                
            print("New kernel is CORRECT. Benchmarking...")
            
            new_time_ms = cuda_utils.benchmark_kernel(A_torch, B_torch)
            print("Analyzing new kernel with NCU...")
            
            # [!!! 已更新 !!!] 捕获完整的 NCU 指标
            new_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, 
                current_module_name, 
                N
            )
            
            if new_time_ms < best_time_ms:
                status = "Success (New Best)"
                details = f"Performance improved from {best_time_ms:.3f} ms to {new_time_ms:.3f} ms."
                print(f"✅ {status} {details}")
                
                best_time_ms = new_time_ms
                best_kernel_code_cuda = new_kernel_code
                best_ptxas_metrics = new_ptxas_metrics
                best_ncu_metrics = new_ncu_metrics
            else:
                status = "Failed (Performance Regression)"
                details = f"New time {new_time_ms:.3f} ms is not better than best time {best_time_ms:.3f} ms."
                print(f"❌ {status} {details}")
            
            # [!!! 已更新 !!!] 存储当轮的指标，供下一轮的 Tool Agent 和 Analysis Agent 使用
            current_ncu_metrics = new_ncu_metrics

        except Exception as e:
            status, details = "Failed (Unhandled Exception)", str(e)
            print(f"❌ {status} {details}")
            
        finally:
            # [!!! 已更新 !!!] 解决了 TODO 问题 5
            # 保存所有 NCU 指标和选定的指标名称
            history_entry = {
                "round": i,
                "goal": opt_goal,
                "status": status,
                "time_ms": new_time_ms if new_time_ms != float('inf') else None,
                "ptxas_metrics": new_ptxas_metrics,
                "all_ncu_metrics": new_ncu_metrics,         # <--- 保存完整的 NCU 指标
                "selected_ncu_metrics": relevant_metric_names, # <--- 保存选定的指标名称
                "details": details,
                "code": new_kernel_code if new_kernel_code else "" 
            }
            optimization_history.append(history_entry)

    # 4. 最终报告
    print("\n--- Optimization Finished ---")
    if optimization_history:
        print(f"Baseline performance (Round 0): {optimization_history[0].get('time_ms', 0.0):.3f} ms")
    print(f"Best kernel performance: {best_time_ms:.3f} ms")
    
    final_kernel_path = "best_gemm_kernel.cu"
    with open(final_kernel_path, "w") as f:
        f.write(best_kernel_code_cuda)
    print(f"Best kernel C++/CUDA source saved to {final_kernel_path}")
    
    with open(config.HISTORY_FILE, 'w') as f:
        json.dump(optimization_history, f, indent=2)
    print(f"Optimization history saved to {config.HISTORY_FILE}")
    
    # 5. 运行最终基准测试
    print("\n--- Running Final Benchmark ---")
    pytorch_time_ms = cuda_utils.get_pytorch_performance(A_torch, B_torch)
    print(f"PyTorch (torch.matmul) performance: {pytorch_time_ms:.3f} ms")
    print(f"Our best LLM-optimized kernel: {best_time_ms:.3f} ms")
    
    speedup = pytorch_time_ms / best_time_ms
    if best_time_ms < pytorch_time_ms:
        print(f"SUCCESS: Optimized kernel is {speedup:.2f}x faster than PyTorch!")
    else:
        print(f"Result: PyTorch is {1/speedup:.2f}x faster.")

if __name__ == "__main__":
    main()