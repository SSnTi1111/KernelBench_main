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
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDING_LIB = True
except ImportError:
    print("[Warning] 'sentence_transformers' or 'sklearn' not found. Embedding similarity check will be disabled.")
    HAS_EMBEDDING_LIB = False

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "test_mini_version"))

from test_script import validate_extracted_code, correct_cuda_kernel



# [!!! 新增 !!!] 全局 Embedding 模型实例 (单例模式)
_EMBEDDING_MODEL = None
EMBEDDING_MODEL_PATH = "/home/lxt/models/all-MiniLM-L6-v2" # 用户指定的路径

def get_embedding_model():
    """
    懒加载 Embedding 模型，确保只加载一次。
    """
    global _EMBEDDING_MODEL
    if not HAS_EMBEDDING_LIB:
        return None

    if _EMBEDDING_MODEL is None:
        try:
            if os.path.exists(EMBEDDING_MODEL_PATH):
                # print(f"[System] Loading embedding model from {EMBEDDING_MODEL_PATH} ...")
                _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_PATH)
            else:
                print(f"[Warning] Embedding model path not found: {EMBEDDING_MODEL_PATH}. Falling back to exact match.")
                # 也可以尝试自动下载: _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"[Warning] Failed to load embedding model: {e}")
            
    return _EMBEDDING_MODEL

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

# def get_diverse_champions(history: list, current_best_code: str, num_kernels=2) -> str: # 针对TODO2 做的修改，完整的修改在下面👇
#     """
#     (此函数保持不变)
#     """
    
#     # 1. 查找所有成功的条目 (不包括 Round 0)
#     success_entries = [
#         h for h in history 
#         if "Success" in h['status'] and h['round'] > 0 and h.get('code')
#     ]
    
#     # 2. 按性能排序
#     success_entries.sort(key=lambda x: x['time_ms'])
    
#     diverse_str = "--- Diverse Successful Kernel Examples (Best first) ---\n"
#     count = 0
    
#     # 3. 提取代码 (确保它与当前最佳代码 *不同*)
#     for entry in success_entries:
#         if entry['code'] == current_best_code:
#             continue # 跳过与当前最佳完全相同的代码
            
#         diverse_str += f"\n\n--- Example {count+1} (From Round {entry['round']}) ---\n"
#         diverse_str += f"// Goal: {entry['goal']}\n"#这个目标是父节点代码的优化目标，优化后的代码是当前的代码
#         diverse_str += f"// Performance: {entry['time_ms']:.3f} ms\n"#是当前代码的执行时间
        
#         # 添加 PTXAS 指标
#         ptxas = entry.get('ptxas_metrics', {})# 是当前代码执行过程中的PTXAS信息
#         for k, v in sorted(ptxas.items()):
#             diverse_str += f"// {k}: {v}\n"
        
#         # [!!! 新增 !!!] 仅添加该轮选择的 NCU 指标
#         selected_metrics = entry.get('selected_ncu_metrics')
#         all_ncu = entry.get('all_ncu_metrics')
        
#         if isinstance(selected_metrics, list) and isinstance(all_ncu, dict) and selected_metrics:
#             diverse_str += f"// Selected NCU Metrics (for Goal):\n"
#             for metric_name in selected_metrics:
#                 value = all_ncu.get(metric_name, 'N/A')
#                 diverse_str += f"//  - {metric_name}: {value}\n"
#         # [!!! 结束新增 !!!]

#         diverse_str += entry['code']
#         count += 1

#         if count >= num_kernels:
#             break
            
#     if count == 0:
#         return "No other diverse successful examples available in history."
#     return diverse_str

# def summarize_history(history: list) -> str: # 由于TODO 1 做的修改，完整的实现见👇
#     """
#     (此函数保持不变)
#     """
#     if not history:
#         return "No previous attempts."
    
#     summary = "Previous Optimization Attempts:\n"
#     for i, entry in enumerate(history):
#         summary += f"  Round {entry['round']}:\n"
#         summary += f"    Goal: {entry['goal']}\n"
#         summary += f"    Status: {entry['status']}\n"
        
#         perf_str = "N/A"
#         if entry['time_ms'] is not None:
#             perf_str = f"{entry['time_ms']:.3f} ms"
#         summary += f"    Performance: {perf_str}\n"

#         # 添加 PTXAS 指标
#         if entry.get('ptxas_metrics'):
#             # 使用 sorted 保证输出顺序稳定，方便阅读
#             for k, v in sorted(entry['ptxas_metrics'].items()):
#                 summary += f"    {k}: {v}\n"

#         # [!!! 新增 !!!] 仅添加该轮选择的 NCU 指标
#         selected_metrics = entry.get('selected_ncu_metrics')
#         all_ncu = entry.get('all_ncu_metrics')
        
#         # 检查 'selected_metrics' 是否是列表，'all_ncu' 是否是字典，并且 'selected_metrics' 不为空
#         if isinstance(selected_metrics, list) and isinstance(all_ncu, dict) and selected_metrics:
#             summary += f"    Selected NCU Metrics (for Goal):\n"
#             for metric_name in selected_metrics:
#                 value = all_ncu.get(metric_name, 'N/A')
#                 summary += f"      - {metric_name}: {value}\n"
#         # [!!! 结束新增 !!!]

#         elif "Error" in entry['status'] or "Failed" in entry['status']:
#             details = entry.get('details', 'No details')
#             if len(details) > 200:
#                 details = details[:200] + "..."
#             summary += f"    Error Details: {details}\n"
#     return summary

def get_diverse_champions(history: list, current_best_code: str, num_kernels=2) -> str:
    """
    (优化版) 获取多样化的成功案例，构建清晰的 [问题 -> 方案 -> 代码 -> 结果] 因果链。
    """
    
    # 1. 查找所有成功的条目 (不包括 Round 0，因为 Round 0 没有优化动机)
    success_entries = [
        h for h in history 
        if ("Success" in h['status'] or "Failed (Performance Regression)" == h['status']) and h['round'] > 0 and h.get('code')
    ]
    
    # 2. 按性能排序 (越快越前)
    success_entries.sort(key=lambda x: x['time_ms'])
    
    diverse_str = "--- Diverse Successful Kernel Examples (Best first) ---\n"
    count = 0

    model = get_embedding_model()
    current_best_emb = None
    similarity_threshold = 0.95 # 相似度阈值，大于此值视为“相同/太相似”

    if model:
        try:
            # 预先计算 current_best_code 的 embedding
            current_best_emb = model.encode(current_best_code).reshape(1, -1)
        except Exception as e:
            print(f"[Warning] Embedding calculation failed for current best code: {e}")
    
    # 3. 提取代码
    for entry in success_entries:
        # DONE: 这里应该改成用embedding做计算，直接判断相等太绝对了
        # if entry['code'] == current_best_code:
        #     continue # 跳过与当前最佳完全相同的代码
        is_too_similar = False
        
        # --- Embedding 相似度计算逻辑 ---
        if model and current_best_emb is not None:
            try:
                entry_code = entry.get('code', '')
                if not entry_code: continue
                
                entry_emb = model.encode(entry_code).reshape(1, -1)
                sim = cosine_similarity(current_best_emb, entry_emb)[0][0]
                
                # 如果相似度过高，认为缺乏多样性，跳过
                if sim > similarity_threshold:
                    print(f"[Info] Skipping similar kernel (Sim: {sim:.4f})")# 因为历史信息中包含自己，因此肯定有一个用例的相似度是1会跳过
                    is_too_similar = True
            except Exception as e:
                # 发生异常降级为字符串比较
                if entry['code'] == current_best_code:
                    is_too_similar = True
        else:
            # 降级：如果没有模型，使用原始字符串完全相等判断
            if entry['code'] == current_best_code:
                is_too_similar = True

        if is_too_similar:
            continue

        print("找到一个正确的且不相似的样例!!!!")
            
        diverse_str += f"\n\n/* ==================================================================================\n"
        diverse_str += f" * Example {count+1} (From Round {entry['round']})\n"
        diverse_str += f" * ================================================================================== */\n"
        
        # --- 1. [Before] 修改前的缺陷 (Motivation) ---
        # 解释：这是对“上一版代码”的诊断，是产生当前这段代码的根本原因
        diag = entry.get('bottleneck_analysis', 'N/A')
        diverse_str += f"// [1. Motivation] Bottleneck of the Previous Kernel (Before Modification):\n"
        diverse_str += f"//    Diagnosis: {diag}\n"
        
        # --- 2. [Action] 优化目标与策略 (Strategy) ---
        # 解释：为了解决上述瓶颈，我们设定了这个目标，并生成了下面的代码
        goal = entry.get('goal', 'N/A')
        plan = entry.get('detailed_plan', 'N/A').replace('\n', ' ')[:120] + "..." # 截取部分计划
        diverse_str += f"// [2. Strategy] Optimization Goal & Plan (Target of this Code):\n"
        diverse_str += f"//    Goal: {goal}\n"
        diverse_str += f"//    Plan Snippet: {plan}\n"
        
        # --- 3. [Code] 优化后的代码 (Implementation) ---
        diverse_str += f"\n/* [3. Implementation] Optimized CUDA Kernel (Addressing the Goal above) */\n"
        diverse_str += entry['code'] + "\n"
        
        # --- 4. [After] 优化后的指标 (Outcome) ---
        diverse_str += f"\n/* [4. Outcome] Performance Metrics of THIS Code (After Modification) */\n"
        diverse_str += f"// Execution Time: {entry['time_ms']:.3f} ms\n"
        
        # PTXAS 指标 (编译器反馈)
        # ptxas = entry.get('ptxas_metrics', {})# 针对TODO3做的修改，完整的修改内容在👇
        # if ptxas:
        #     diverse_str += "// PTXAS Compiler Stats:\n"
        #     for k, v in sorted(ptxas.items()):
        #         diverse_str += f"//   - {k}: {v}\n"
        ptxas = entry.get('ptxas_metrics', {})
        if ptxas:
            diverse_str += "// PTXAS Compiler Stats:\n"
            for kernel_name, stats in sorted(ptxas.items()):
                if isinstance(stats, dict):
                    # 新格式：提取关键指标并紧凑展示
                    info_parts = []
                    # 1. 寄存器
                    if 'registers' in stats: 
                        info_parts.append(f"Regs={stats['registers']}")
                    
                    # 2. 溢出 (重点关注)
                    spill = stats.get('spill_bytes', 0)
                    if spill > 0: 
                        info_parts.append(f"SPILL={spill}B") # 溢出时显式显示
                    else:
                        info_parts.append("Spill=0B")
                        
                    # 3. 向量化宽度 (如果有)
                    width = stats.get('width')
                    if width and width not in ["Scalar", "?", ""]: 
                        info_parts.append(f"Vec={width}")
                    
                    # 4. 常量/共享内存 (可选，看你需要)
                    if stats.get('smem_bytes', 0) > 0:
                        info_parts.append(f"SMem={stats['smem_bytes']}B")
                    
                    details = ", ".join(info_parts)
                    # 简化 Kernel 名字显示，去除过长的模板参数
                    short_name = kernel_name.split('<')[0] if '<' in kernel_name else kernel_name
                    if width and str(width).isdigit(): short_name += f"(vec{width})"
                    
                    diverse_str += f"//   - [{short_name}]: {details}\n"
                else:
                    # 旧格式兼容 (Fallback)
                    diverse_str += f"//   - {kernel_name}: {stats}\n"
        
        # NCU 指标 (运行时反馈 - 仅展示为了验证目标而选择的指标)
        selected_metrics = entry.get('selected_ncu_metrics')
        all_ncu = entry.get('all_ncu_metrics')
        
        if isinstance(selected_metrics, list) and isinstance(all_ncu, dict) and selected_metrics:
            diverse_str += "// Key NCU Hardware Metrics (Verified against Goal):\n"
            for metric_name in selected_metrics:
                value = all_ncu.get(metric_name, 'N/A')
                diverse_str += f"//   - {metric_name}: {value}\n"
        
        count += 1
        if count >= num_kernels:
            break
            
    if count == 0:
        return "No other diverse successful examples available in history."
    return diverse_str

def summarize_history(history: list) -> str:
    """
    (优化版) 生成具有明确因果关系链的历史摘要。
    结构：[Before: 动机] -> [Action: 动作] -> [After: 结果]
    """
    if not history:
        return "No previous attempts."
    
    summary = "=== Optimization Experiments Knowledge Base ===\n"
    
    # 1. Baseline 单独展示
    baseline = history[0]
    summary += f"[Round 0 - Baseline] Performance: {baseline.get('time_ms', 'N/A'):.3f} ms\n"
    # 如果 Baseline 有 NCU 指标，也稍微展示一下关键的
    if baseline.get('all_ncu_metrics'):
        base_ncu = baseline['all_ncu_metrics']
        
        # [!!! 修复 !!!] 使用实际存在的键名
        # 优先尝试简化的键名，如果不存在则尝试原始键名，最后回退到 N/A
        dram_val = base_ncu.get('DRAMThroughput', base_ncu.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A'))
        sm_val = base_ncu.get('ComputeSMThroughput', base_ncu.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A'))
        
        # 如果取到了数值，格式化一下
        dram_str = f"{dram_val}%" if isinstance(dram_val, (int, float)) else "N/A"
        sm_str = f"{sm_val}%" if isinstance(sm_val, (int, float)) else "N/A"
        
        summary += f"  > Baseline Context: DRAM={dram_str}, SM={sm_str}\n"
        
    summary += "-" * 50 + "\n"

    # 2. 展示后续迭代
    for entry in history[1:]:
        r = entry['round']
        status = entry['status']
        time_ms = entry.get('time_ms')
        perf_str = f"{time_ms:.3f} ms" if time_ms else "N/A"
        
        summary += f"[Round {r}] Status: {status} | Time: {perf_str}\n"
        
        # --- [Before: 动机] ---
        # 解释：这是针对“上一轮代码”的诊断，是这一轮优化的起因
        diag = entry.get('bottleneck_analysis', 'N/A')
        summary += f"  > [Motivation] Bottlenecks in the kernel before optimization: {diag}\n"
        
        # --- [Action: 动作] ---
        # 解释：这是这一轮具体做了什么
        goal = entry.get('goal', 'N/A')
        summary += f"  > [Action] Optimization Goal: {goal}\n"
        
        # Plan 摘要
        plan_text = entry.get('detailed_plan', '')
        if plan_text:
            plan_snippet = plan_text.replace('\n', ' ')[:150] + "..."
            summary += f"  > [Action] Plan Details: {plan_snippet}\n"

        # --- [After: 结果] ---
        # 解释：这是这一轮代码编译和运行后的客观数据
        summary += "  > [Result] According to the optimization objective, the information of each index of the optimized CUDA kernel is as follows:(PTXAS & NCU):\n"
        
        # 1. PTXAS (编译器结果)
        # ptxas = entry.get('ptxas_metrics', {})# 针对与TODO3的修改,修改内容如下👇
        # metrics_shown = False
        # for k, v in sorted(ptxas.items()):
        #     # 只显示非零的溢出或关键寄存器信息，减少噪音
        #     if 'spill' in k and v > 0:
        #         summary += f"    ! PTXAS {k}: {v} (SPILL DETECTED)\n"
        #         metrics_shown = True
        #     elif 'registers' in k:
        #         summary += f"    - PTXAS {k}: {v}\n"
        #         metrics_shown = True

        ptxas = entry.get('ptxas_metrics', {})
        if ptxas:
            for kernel_name, stats in sorted(ptxas.items()):
                # 防御性检查：确保 stats 是字典（兼容旧日志格式）
                if not isinstance(stats, dict):
                    # 旧格式回退处理
                    if 'spill' in kernel_name and stats > 0:
                        summary += f"    ! PTXAS {kernel_name}: {stats} (SPILL!)\n"
                    continue

                # 新格式解析
                regs = stats.get('registers', 'N/A')
                spill = stats.get('spill_bytes', 0)
                width = stats.get('width', '')
                
                # 构建显示的字符串
                info_parts = [f"Regs={regs}"]
                if spill > 0:
                    info_parts.append(f"SPILL={spill} bytes")
                if width and width not in ["Scalar", "?", ""]:
                    info_parts.append(f"Vec={width}") # 显示向量化宽度对优化很有用
                
                info_str = ", ".join(info_parts)
                
                # 如果有溢出，用感叹号开头
                prefix = "!" if spill > 0 else "-"
                warning = " (SPILL DETECTED!)" if spill > 0 else ""
                
                # 简化 Kernel 名字显示 (去除过长的模板参数，保留核心)
                # 例如: sigmoid_kernel_vec<float, 4> -> sigmoid_kernel_vec
                short_name = kernel_name.split('<')[0] if '<' in kernel_name else kernel_name
                if width and width.isdigit(): short_name += f"(vec{width})"
                
                summary += f"    {prefix} PTXAS [{short_name}]: {info_str}{warning}\n"
                metrics_shown = True
        
        # 2. NCU (运行时硬件结果 - 闭环验证)
        # 这里展示 Tool Agent 专门挑选来验证 Goal 的指标
        selected_ncu = entry.get('selected_ncu_metrics') # List[str]
        all_ncu = entry.get('all_ncu_metrics') # Dict
        
        if isinstance(selected_ncu, list) and isinstance(all_ncu, dict) and selected_ncu:
            for metric_name in selected_ncu:
                val = all_ncu.get(metric_name, 'N/A')
                # 简化指标名称，去掉过长的前缀以便 LLM 阅读
                short_name = metric_name.split('.')[-1] if '.' in metric_name else metric_name
                summary += f"    - NCU {short_name}: {val}\n"
            metrics_shown = True
            
        if not metrics_shown:
            summary += "    (No significant metrics available)\n"

        # 3. 失败原因 (如果有)
        if "Error" in status or "Failed" in status:
            err = entry.get('details', '')
            summary += f"  > [Result] The reason why the optimized kernel failed: {err[:250]}...\n"
        
        summary += "\n"
        
    return summary

def format_metrics_for_llm(ptxas_metrics: dict, ncu_metrics: dict) -> str:
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
    baseline_time_ms, # [!!! 已修改 !!!] 接收Pytorch基准
    full_pytorch_source_code,
    pytorch_kernel_module
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
    
    best_kernel_code_cuda = initial_cuda_code  
    best_time_ms = float('inf')
    best_ptxas_metrics = {}
    best_ncu_metrics = {}
    current_ncu_metrics = {}
    
    optimization_history = []
    
    if os.path.exists(history_file_path): 
        print(f"Loading existing history from {history_file_path}")
        try:
            with open(history_file_path, 'r', encoding='utf-8') as f:  
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
        current_module_name = f"{problem_name}_0"  
        try:
            module, stdout_log, stderr_log = cuda_utils.load_module(
                # cpp_source, 
                best_kernel_code_cuda, 
                current_module_name,
                init_inputs,
                pytorch_kernel_module
                # wrapper_function_name=wrapper_function_name  
            )
            print("Baseline kernel compiled successfully.")
            best_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log)
            
            # [!!! 已更新 !!!]
            is_correct = cuda_utils.check_correctness(inputs, ref_outputs, module)
            if not is_correct:
                print("❌ Baseline kernel is INCORRECT. Exiting.")
                return {"error": "Baseline kernel incorrect."}  
                
            print("Baseline kernel is correct. Benchmarking...")
            # [!!! 已更新 !!!]
            best_time_ms = cuda_utils.benchmark_kernel(inputs, module)
            
            print("Analyzing baseline kernel with NCU (this may take a while)...")
            # [!!! 已更新 !!!]
            best_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, 
                current_module_name, 
                # kernel_name,            
                # wrapper_function_name,  
                inputs                  
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
                    f.write(best_kernel_code_cuda)
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
            return {"error": f"Baseline failed: {e}"}  
        
        finally:
            if module is not None:
                del module
    
    if not current_ncu_metrics: 
        current_ncu_metrics = best_ncu_metrics if best_ncu_metrics else {}
    # if module is not None:
    #     del module
    torch.cuda.empty_cache()


    # 3. 开始优化循环
    # [!!! 已更新 !!!]
    for i in tqdm(range(len(optimization_history), iteration_rounds + 1), desc="Optimization Rounds"):
        if i == 0: continue # Round 0 已经完成
        
        print(f"\n--- Round {i}/{iteration_rounds} ---")
        
        history_summary = summarize_history(optimization_history)# 如果是优化之前的首轮，这里面只有ptxas_metrics，没有ncu信息，全部的ncu信息也没有(是所有历史信息)
        metrics_summary = format_metrics_for_llm(best_ptxas_metrics, best_ncu_metrics)# 是当前最好的ptxax信息和ncu信息，是全部
        
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
            parent_kernel_code = best_kernel_code_cuda
            
            planner_response = agents.call_llm(
                "planner", 
                prompts.PLANNER_SYSTEM_PROMPT,
                f"Optimization History:\n{history_summary}\n\n"#DONE 1 这个提示词信息需要重新设计一下，history_summary光有这些信息没有用啊，不知道每个指标的对应的代码是什么啊，每个记录的代码是在哪个版本上做的修改啊，这些都不知道
                f"=== Hardware Metrics for Current Best Kernel(Need to be optimized) ===\n{metrics_summary}\n\n"
                f"Current Best C++/CUDA Source (Time: {best_time_ms:.3f} ms):\n{parent_kernel_code}"  
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
                
            # tool_response = agents.call_llm(# DONE👇:这个提示词也要重新设计一下，只有27个指标和优化目标，让LLM从中选5个，可是不知道现在要优化的任务代码是什么啊？
            #     "tool", 
            #     prompts.TOOL_SYSTEM_PROMPT,
            #     f"All Available NCU Metric Names ({len(all_metric_names)}): {all_metric_names}\n\nOptimization Goal: {opt_goal}"
            # )
            tool_response = agents.call_llm(
                "tool", 
                prompts.TOOL_SYSTEM_PROMPT,
                f"Optimization Goal: {opt_goal}\n\n"
                f"Planner's Bottleneck Analysis: {bottleneck_analysis}\n\n"
                f"Current C++/CUDA Source:\n{parent_kernel_code}\n\n" 
                f"All Available NCU Metric Names ({len(all_metric_names)}): {all_metric_names}"
            )

            print("-----------------------LXT:tool_response----------------------")
            print(tool_response)
            print("-----------------------LXT:tool_response----------------------")
            
            relevant_metric_names = extract_metrics(tool_response)# 将五个指标的名称从回复中提取出来。
            
            if not relevant_metric_names:
                status, details = "Failed (Tool)", "Tool Agent did not return a valid metric list."
                print(f"❌ {status} {details}")
                continue 
            print(f"[Tool Agent] Selected {len(relevant_metric_names)} metrics: {relevant_metric_names}")
            
            relevant_metrics_dict = {
                metric: current_ncu_metrics.get(metric, 0.0) 
                for metric in relevant_metric_names
            }# 将选择的五个指标提取出来
            
            diverse_kernels_str = get_diverse_champions(optimization_history, best_kernel_code_cuda)# 从历史信息中找到和当前的最好版本best_kernel_code_cuda不是很相似的最好的两个代码和相关指标。
            
            # 3. Analysis Agent [!!! 已更新 !!!]
            print("[Analysis Agent] Formulating plan...")
            analysis_response = agents.call_llm(
                "analysis", 
                prompts.ANALYSIS_SYSTEM_PROMPT,
                f"Planner's Bottleneck Analysis: {bottleneck_analysis}\n\n"
                f"Optimization Goal: {opt_goal}\n\n"
                f"Optimization History:\n{history_summary}\n\n"# DONE：和planner的同理，这个history_summary的信息是否足够有用？
                f"Diverse Successful Kernel Examples:\n{diverse_kernels_str}\n\n"#DONE 2 这里的信息是不是应该好好组织一下，不然LLM分不清这些指标和优化目标是当前代码的还是当前代码的父节点的
                f"Current C++/CUDA Source need to be optimized:\n{parent_kernel_code}\n\n" # parent_kernel_code就是当前最好的kernel,也就是正在改的版本，是当前这个！每次改的都是最好的那个
                f"Current Hardware Metrics (Full Set): {metrics_summary}\n\n"# 是当前kernel的全部PTXAS信息和NCU信息
                f"Tool-Selected Metrics from *Previous* Run (Values): {relevant_metrics_dict}" # 是当前kernel的选择出来的五个相关NCU指标。
            )
            print("-----------------------LXT:analysis_response----------------------")
            print(analysis_response)#DONE:当前的输出部分没有think的过程
            print("-----------------------LXT:analysis_response----------------------")
            if not analysis_response or "DETAILED_PLAN:" not in analysis_response:
                status, details = "Failed (Analysis)", "Analysis Agent did not return a valid plan."
                print(f"❌ {status} {details}")
                continue 
            detailed_plan = analysis_response.split("DETAILED_PLAN:")[1].strip()

            # 4. Coder Agent
            print("[Coder Agent] Generating new kernel...")
            coder_response = agents.call_llm(# DONE:coder agent没有思考过程
                "coder", 
                prompts.CODER_SYSTEM_PROMPT,
                f"Original C++/CUDA Source:\n{parent_kernel_code}\n\nDetailed Plan:\n{detailed_plan}" 
            )
            print("-----------------------LXT:coder_response----------------------")
            print(coder_response)
            print("-----------------------LXT:coder_response----------------------")
            
            new_kernel_code_full = final_extract(coder_response) 
            #好
            # new_kernel_code_full = '''import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\n# ---------------------------------------------------------------------------\n# CUDA source (kernels + C++/ATen host wrappers)\n# ---------------------------------------------------------------------------\nsource = r\'\'\'\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n\ntemplate <typename scalar_t>\n__device__ __forceinline__ scalar_t sigmoid_func(scalar_t x) {\n    return scalar_t(1) / (scalar_t(1) + exp(-x));\n}\n\n/* ---------------------------------------------------------\n * Scalar fallback kernel : one-element per thread\n * ------------------------------------------------------- */\ntemplate <typename scalar_t>\n__global__ void sigmoid_kernel_scalar(const scalar_t* __restrict__ input,\n                                      scalar_t* __restrict__ output,\n                                      const int64_t numel) {\n    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < numel) {\n        output[idx] = sigmoid_func(input[idx]);\n    }\n}\n\n/* ---------------------------------------------------------\n * Vectorised kernel : VEC elements per thread\n * VEC = 4 for float (float4, 16-byte transaction)\n *     = 2 for double (double2, 16-byte transaction)\n * The last (numel % VEC) elements are processed by a\n * single thread (vec_idx == 0) inside the same kernel.\n * ------------------------------------------------------- */\ntemplate <typename scalar_t , int VEC>\n__global__ void sigmoid_kernel_vec(const scalar_t* __restrict__ input,\n                                   scalar_t*       __restrict__ output,\n                                   const int64_t   vec_elems,\n                                   const int64_t   tail_start,\n                                   const int64_t   tail_size) {\n    using VecT = typename std::conditional< (sizeof(scalar_t)==4),\n                                            float4,              // 4 x fp32 = 16 B\n                                            double2               // 2 x fp64 = 16 B\n                                          >::type;\n\n    const int64_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n    /* ---------------- Aligned, vectorised path ---------------- */\n    if (vec_idx < vec_elems) {\n        VecT v = reinterpret_cast<const VecT*>(input)[vec_idx];\n\n        scalar_t* v_elem = reinterpret_cast<scalar_t*>(&v);\n        #pragma unroll\n        for (int i = 0; i < VEC; ++i) {\n            v_elem[i] = sigmoid_func(v_elem[i]);\n        }\n\n        reinterpret_cast<VecT*>(output)[vec_idx] = v;\n    }\n\n    /* ---------------- Tail handling by one thread ------------- */\n    if (tail_size && vec_idx == 0) {\n        for (int64_t j = 0; j < tail_size; ++j) {\n            const int64_t idx = tail_start + j;\n            output[idx] = sigmoid_func(input[idx]);\n        }\n    }\n}\n\n/* ---------------------------------------------------------\n * Host launcher\n * ------------------------------------------------------- */\ntorch::Tensor sigmoid_forward(torch::Tensor input) {\n    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");\n    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");\n\n    auto output = torch::empty_like(input);\n    const int64_t numel = input.numel();\n    const int threads = 256;\n    auto stream = at::cuda::getCurrentCUDAStream();\n\n    // Fast path : fp32 / fp64 with vectorised kernel\n    if (input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble) {\n\n        if (input.scalar_type() == at::kFloat) {\n            using scalar_t = float;\n            constexpr int  VEC = 4;\n            const int64_t  vec_elems  = numel / VEC;\n            const int64_t  tail_start = vec_elems * VEC;\n            const int64_t  tail_sz    = numel - tail_start;\n            const int64_t  blocks     = (vec_elems + threads - 1) / threads;\n\n            if (blocks > 0) {\n                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>(),\n                    output.data_ptr<scalar_t>(),\n                    vec_elems,\n                    tail_start,\n                    tail_sz);\n            } else if (tail_sz) {\n                // Fallback to scalar kernel if vector part is empty\n                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;\n                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>() + tail_start,\n                    output.data_ptr<scalar_t>() + tail_start,\n                    tail_sz);\n            }\n        } else { // double\n            using scalar_t = double;\n            constexpr int  VEC = 2;\n            const int64_t  vec_elems  = numel / VEC;\n            const int64_t  tail_start = vec_elems * VEC;\n            const int64_t  tail_sz    = numel - tail_start;\n            const int64_t  blocks     = (vec_elems + threads - 1) / threads;\n\n            if (blocks > 0) {\n                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>(),\n                    output.data_ptr<scalar_t>(),\n                    vec_elems,\n                    tail_start,\n                    tail_sz);\n            } else if (tail_sz) {\n                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;\n                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>() + tail_start,\n                    output.data_ptr<scalar_t>() + tail_start,\n                    tail_sz);\n            }\n        }\n\n    } else {\n        /* Generic scalar kernel for remaining dtypes (half, bfloat16, etc.) */\n        const int64_t blocks = (numel + threads - 1) / threads;\n        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),\n                                            "sigmoid_forward_cuda_scalar", ([&] {\n            sigmoid_kernel_scalar<scalar_t><<<blocks, threads, 0, stream>>>(\n                input.data_ptr<scalar_t>(),\n                output.data_ptr<scalar_t>(),\n                numel);\n        }));\n    }\n\n    cudaError_t err = cudaGetLastError();\n    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed with error code ", err);\n\n    return output;\n}\n\'\'\'\n\n# ---------------------------------------------------------------------------\n# C++ function prototypes\n# ---------------------------------------------------------------------------\ncpp_src = r\'\'\'\ntorch::Tensor sigmoid_forward(torch::Tensor input);\n\'\'\'\n\n# ---------------------------------------------------------------------------\n# Build & load extension\n# ---------------------------------------------------------------------------\nsigmoid_module = load_inline(\n    name         = \'sigmoid_cuda_opt\',\n    cpp_sources  = cpp_src,\n    cuda_sources = source,\n    functions    = [\'sigmoid_forward\'],\n    with_cuda    = True,\n    verbose      = True,\n    extra_cuda_cflags=[\'-O3\', \'--ptxas-options=-v\']\n)\n\n# ---------------------------------------------------------------------------\n# PyTorch Module wrapper\n# ---------------------------------------------------------------------------\nclass ModelNew(nn.Module):\n    """\n    CUDA-accelerated model that applies element-wise Sigmoid.\n    Mirrors the original Model interface.\n    """\n    def __init__(self):\n        super(ModelNew, self).__init__()\n        self.sigmoid = sigmoid_module\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.sigmoid.sigmoid_forward(x)'''
            # 这个是有结果问题的好用例
            # new_kernel_code_full = '''import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\n# ---------------------------------------------------------------------------\n# CUDA source (kernels + C++/ATen host wrappers)\n# ---------------------------------------------------------------------------\nsource = r\'\'\'\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n\ntemplate <typename scalar_t>\n__device__ __forceinline__ scalar_t sigmoid_func(scalar_t x) {\n    return scalar_t(1) / (scalar_t(1) + exp(x));\n}\n\n/* ---------------------------------------------------------\n * Scalar fallback kernel : one-element per thread\n * ------------------------------------------------------- */\ntemplate <typename scalar_t>\n__global__ void sigmoid_kernel_scalar(const scalar_t* __restrict__ input,\n                                      scalar_t* __restrict__ output,\n                                      const int64_t numel) {\n    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < numel) {\n        output[idx] = sigmoid_func(input[idx]);\n    }\n}\n\n/* ---------------------------------------------------------\n * Vectorised kernel : VEC elements per thread\n * VEC = 4 for float (float4, 16-byte transaction)\n *     = 2 for double (double2, 16-byte transaction)\n * The last (numel % VEC) elements are processed by a\n * single thread (vec_idx == 0) inside the same kernel.\n * ------------------------------------------------------- */\ntemplate <typename scalar_t , int VEC>\n__global__ void sigmoid_kernel_vec(const scalar_t* __restrict__ input,\n                                   scalar_t*       __restrict__ output,\n                                   const int64_t   vec_elems,\n                                   const int64_t   tail_start,\n                                   const int64_t   tail_size) {\n    using VecT = typename std::conditional< (sizeof(scalar_t)==4),\n                                            float4,              // 4 x fp32 = 16 B\n                                            double2               // 2 x fp64 = 16 B\n                                          >::type;\n\n    const int64_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n    /* ---------------- Aligned, vectorised path ---------------- */\n    if (vec_idx < vec_elems) {\n        VecT v = reinterpret_cast<const VecT*>(input)[vec_idx];\n\n        scalar_t* v_elem = reinterpret_cast<scalar_t*>(&v);\n        #pragma unroll\n        for (int i = 0; i < VEC; ++i) {\n            v_elem[i] = sigmoid_func(v_elem[i]);\n        }\n\n        reinterpret_cast<VecT*>(output)[vec_idx] = v;\n    }\n\n    /* ---------------- Tail handling by one thread ------------- */\n    if (tail_size && vec_idx == 0) {\n        for (int64_t j = 0; j < tail_size; ++j) {\n            const int64_t idx = tail_start + j;\n            output[idx] = sigmoid_func(input[idx]);\n        }\n    }\n}\n\n/* ---------------------------------------------------------\n * Host launcher\n * ------------------------------------------------------- */\ntorch::Tensor sigmoid_forward(torch::Tensor input) {\n    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");\n    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");\n\n    auto output = torch::empty_like(input);\n    const int64_t numel = input.numel();\n    const int threads = 256;\n    auto stream = at::cuda::getCurrentCUDAStream();\n\n    // Fast path : fp32 / fp64 with vectorised kernel\n    if (input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble) {\n\n        if (input.scalar_type() == at::kFloat) {\n            using scalar_t = float;\n            constexpr int  VEC = 4;\n            const int64_t  vec_elems  = numel / VEC;\n            const int64_t  tail_start = vec_elems * VEC;\n            const int64_t  tail_sz    = numel - tail_start;\n            const int64_t  blocks     = (vec_elems + threads - 1) / threads;\n\n            if (blocks > 0) {\n                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>(),\n                    output.data_ptr<scalar_t>(),\n                    vec_elems,\n                    tail_start,\n                    tail_sz);\n            } else if (tail_sz) {\n                // Fallback to scalar kernel if vector part is empty\n                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;\n                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>() + tail_start,\n                    output.data_ptr<scalar_t>() + tail_start,\n                    tail_sz);\n            }\n        } else { // double\n            using scalar_t = double;\n            constexpr int  VEC = 2;\n            const int64_t  vec_elems  = numel / VEC;\n            const int64_t  tail_start = vec_elems * VEC;\n            const int64_t  tail_sz    = numel - tail_start;\n            const int64_t  blocks     = (vec_elems + threads - 1) / threads;\n\n            if (blocks > 0) {\n                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>(),\n                    output.data_ptr<scalar_t>(),\n                    vec_elems,\n                    tail_start,\n                    tail_sz);\n            } else if (tail_sz) {\n                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;\n                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(\n                    input.data_ptr<scalar_t>() + tail_start,\n                    output.data_ptr<scalar_t>() + tail_start,\n                    tail_sz);\n            }\n        }\n\n    } else {\n        /* Generic scalar kernel for remaining dtypes (half, bfloat16, etc.) */\n        const int64_t blocks = (numel + threads - 1) / threads;\n        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),\n                                            "sigmoid_forward_cuda_scalar", ([&] {\n            sigmoid_kernel_scalar<scalar_t><<<blocks, threads, 0, stream>>>(\n                input.data_ptr<scalar_t>(),\n                output.data_ptr<scalar_t>(),\n                numel);\n        }));\n    }\n\n    cudaError_t err = cudaGetLastError();\n    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed with error code ", err);\n\n    return output;\n}\n\'\'\'\n\n# ---------------------------------------------------------------------------\n# C++ function prototypes\n# ---------------------------------------------------------------------------\ncpp_src = r\'\'\'\ntorch::Tensor sigmoid_forward(torch::Tensor input);\n\'\'\'\n\n# ---------------------------------------------------------------------------\n# Build & load extension\n# ---------------------------------------------------------------------------\nsigmoid_module = load_inline(\n    name         = \'sigmoid_cuda_opt\',\n    cpp_sources  = cpp_src,\n    cuda_sources = source,\n    functions    = [\'sigmoid_forward\'],\n    with_cuda    = True,\n    verbose      = True,\n    extra_cuda_cflags=[\'-O3\', \'--ptxas-options=-v\']\n)\n\n# ---------------------------------------------------------------------------\n# PyTorch Module wrapper\n# ---------------------------------------------------------------------------\nclass ModelNew(nn.Module):\n    """\n    CUDA-accelerated model that applies element-wise Sigmoid.\n    Mirrors the original Model interface.\n    """\n    def __init__(self):\n        super(ModelNew, self).__init__()\n        self.sigmoid = sigmoid_module\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.sigmoid.sigmoid_forward(x)'''
            # 这个是会对输入数据进行修改的用例
            # new_kernel_code_full = """import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nsource = r'''\n#include <torch/extension.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <stdint.h>\n\n// ---------------------------------------------------------------------\n// Fast sigmoid for FP32 (uses CUDA fast-math intrinsic __expf)\n// ---------------------------------------------------------------------\n__device__ __forceinline__ float sigmoidf(float v)\n{\n    return 1.f / (1.f + __expf(-v));\n}\n\n// ---------------------------------------------------------------------\n// Scalar in-place kernel (generic, used for FP64 fallback)\n// ---------------------------------------------------------------------\ntemplate<typename scalar_t>\n__global__ __launch_bounds__(256, 2)\nvoid sigmoid_kernel_scalar_ip(scalar_t* __restrict__ data,\n                              const int64_t          numel)\n{\n    const int idx    = blockDim.x * blockIdx.x + threadIdx.x;\n    const int stride = blockDim.x * gridDim.x;\n\n    for (int64_t i = idx; i < numel; i += stride)\n    {\n        scalar_t v = data[i];\n        data[i] = scalar_t(1.0) / (scalar_t(1.0) + exp(-v));\n    }\n}\n\n// ---------------------------------------------------------------------\n// Vectorised FP32 kernel \u2013 processes 4 elements at a time, in-place\n// ---------------------------------------------------------------------\n__global__ __launch_bounds__(256, 2)\nvoid sigmoid_kernel_vec4_ip(float*       __restrict__ data,\n                            const int64_t            numel)\n{\n    const int idx    = blockDim.x * blockIdx.x + threadIdx.x;\n    const int stride = blockDim.x * gridDim.x;\n\n    // Number of float4 elements\n    const int64_t vec_elems = numel >> 2;  // numel / 4\n\n    // Reinterpret data pointer as float4*\n    float4* __restrict__ d4 = reinterpret_cast<float4*>(data);\n\n    // ---- main vectorised loop ----\n    for (int64_t i = idx; i < vec_elems; i += stride)\n    {\n        float4 v = d4[i];   // 128-bit load\n        v.x = sigmoidf(v.x);\n        v.y = sigmoidf(v.y);\n        v.z = sigmoidf(v.z);\n        v.w = sigmoidf(v.w);\n        d4[i] = v;          // 128-bit store (same location)\n    }\n\n    // ---- scalar tail (handles numel % 4) ----\n    for (int64_t i = (vec_elems << 2) + idx; i < numel; i += stride)\n    {\n        data[i] = sigmoidf(data[i]);\n    }\n}\n\n// ---------------------------------------------------------------------\n// Host wrapper \u2013 launches in-place kernels\n// ---------------------------------------------------------------------\ntorch::Tensor sigmoid_cuda_inplace(torch::Tensor x)\n{\n    TORCH_CHECK(x.is_cuda(),  \"Input must reside on CUDA device\");\n    TORCH_CHECK(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat64,\n                \"Supported dtypes are float32 and float64\");\n\n    // Ensure contiguous layout; otherwise make a contiguous copy\n    auto x_contig = x.contiguous();\n    const int64_t numel = x_contig.numel();\n\n    if (x_contig.scalar_type() == torch::kFloat32)\n    {\n        const int threads = 256;\n        // Each thread handles 4 elements via float4\n        const int blocks  = (((numel + 3) >> 2) + threads - 1) / threads;\n\n        sigmoid_kernel_vec4_ip<<<blocks, threads>>>(\n            x_contig.data_ptr<float>(),\n            numel);\n    }\n    else    // FP64 path\n    {\n        const int threads = 256;\n        const int blocks  = (numel + threads - 1) / threads;\n\n        sigmoid_kernel_scalar_ip<double><<<blocks, threads>>>(\n            x_contig.data_ptr<double>(),\n            numel);\n    }\n\n    // Check for launch / runtime errors\n    cudaError_t err = cudaGetLastError();\n    if (err != cudaSuccess)\n        throw std::runtime_error(std::string(\"sigmoid_cuda_inplace failed: \")\n                                 + cudaGetErrorString(err));\n\n    return x_contig;\n}\n'''\n\ncpp_src = r'''\ntorch::Tensor sigmoid_cuda_inplace(torch::Tensor x);\n'''\n\nsigmoid_mod = load_inline(\n    name         = 'sigmoid_mod_ip',\n    cpp_sources  = cpp_src,\n    cuda_sources = source,\n    functions    = ['sigmoid_cuda_inplace'],\n    with_cuda    = True,\n    verbose      = True,\n    extra_cuda_cflags=['-O3', '--use_fast_math', '--ptxas-options=-v'],\n)\n\nclass ModelNew(nn.Module):\n    \"\"\"\n    Drop-in replacement using an in-place, vectorised CUDA sigmoid.\n    \"\"\"\n    def __init__(self):\n        super().__init__()\n        self.sigmoid_cuda = sigmoid_mod.sigmoid_cuda_inplace\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.sigmoid_cuda(x)"""

            #坏
            # new_kernel_code_full = '''import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\nsource = r\'\'\'\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n\ntemplate <typename scalar_t>\n__device__ __forceinline__ scalar_t sigmoid_func(scalar_t x) {\n    return scalar_t(1) / (scalar_t(1) + exp(-x));\n}\n\n// Kernel: element-wise Sigmoid\ntemplate <typename scalar_t>\n__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,\n                               scalar_t* __restrict__ output,\n                               const int64_t numel) {\n    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < numel) {\n        scalar_t val = input[idx];\n        output[idx] = sigmoid_func(val);\n    }\n}\n\ntorch::Tensor sigmoid_forward(torch::Tensor input) {\n    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");\n    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");\n    auto output = torch::empty_like(input);\n\n    const int64_t numel = input.numel();\n    const int threads = 256;\n    const int64_t blocks = (numel + threads - 1) / threads;\n\n    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_forward_cuda", ([&] {\n        sigmoid_kernel<scalar_t><<<blocks, threads, 0,\n                                   at::cuda::getCurrentCUDAStream()>>>(\n            input.data_ptr<scalar_t>(),\n            output.data_ptr<scalar_t>(),\n            numel);\n    }));\n\n    cudaError_t err = cudaGetLastError();\n    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed with error code ", err);\n    return output;\n}\n\'\'\'\n\ncpp_src = r\'\'\'\ntorch::Tensor sigmoid_forward(torch::Tensor input);\n\'\'\'\n\nsigmoid_module = load_inline(\n    name=\'sigmoid_cuda\',\n    cpp_sources=cpp_src,\n    cuda_sources=source,\n    functions=[\'sigmoid_forward\'],\n    with_cuda=True,\n verbose=True,\n    extra_cuda_cflags=[\'-O2\',\'--ptxas-options=-v\']\n)\n\n\nclass ModelNew(nn.Module):\n    """\n    CUDA-accelerated model that applies element-wise Sigmoid.\n    Mirrors the original Model interface.\n    """\n    def __init__(self):\n        super(ModelNew, self).__init__()\n        self.sigmoid = sigmoid_module\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.sigmoid.sigmoid_forward(x)'''

            if not new_kernel_code_full: 
                status, details = "Failed (Coder)", "Coder Agent did not produce valid code."
                print(f"❌ {status} {details}")
                continue 
            print("[Coder Agent] New kernel source generated.")
                
            # 5. 验证和分析
            current_module_name = f"{problem_name}_{i}" # 当前的new_kernel_code_full的新名字
            print(f"Compiling new kernel (module: {current_module_name})...")
            
            try:
                # current_code_is_correct = False
                for j in range(5):
                    # gc.collect()
                    verResult, errMessage = validate_extracted_code(new_kernel_code_full, init_inputs, inputs, ref_outputs, pytorch_kernel_module)# 这个errMessage中对于结果错误的信息没有做前五个错误数据提取，是全部的错误数据
                    if not verResult:
                        print(f"尝试修正当前错误，第{j}次尝试")
                        err_str = str(errMessage)
                        print(f"--- Error Snippet ---\n{err_str[:500]}...\n---------------------")
                        if len(err_str) > 4000:
                            err_str = err_str[:2000] + "\n...[TRUNCATED]...\n" + err_str[-2000:]
                        new_kernel_code_full = correct_cuda_kernel(
                            full_pytorch_source_code,
                            new_kernel_code_full,
                            errMessage
                        )
                        if new_kernel_code_full:
                            print("Code corrected by LLM. Retrying verification...")
                        else:
                            print("LLM correction failed (did not return valid code). Aborting.")
                            break
                    else:
                        # current_code_is_correct = True
                        break
                # gc.collect()
                module, stdout_log, err_msg = cuda_utils.load_module(
                    new_kernel_code_full,
                    current_module_name,
                    init_inputs, 
                    pytorch_kernel_module
                )
                # print("Compilation successful.")
                
                new_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log)# DONE3 针对21用例这里提取的PTXAS信息不太对劲
                if not module:
                    status, details = "Failed (Compilation)", f"New kernel is COMPILATION INCORRECT.{err_msg}"
                    print(f"❌ {status}")
                    continue 
                is_correct, err_str = cuda_utils.check_correctness(inputs, ref_outputs, module)
                if not is_correct:
                    status, details = "Failed (Correctness)", f"New kernel is OUTPUT RESULT INCORRECT.{err_str}"
                    print(f"❌ {status}")
                    continue 



                # module, stdout_log, err_msg = cuda_utils.load_module(
                #     new_kernel_code_full,
                #     current_module_name,
                #     init_inputs, 
                # )
                # # print("Compilation successful.")
                
                # new_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log)# DONE3 针对21用例这里提取的PTXAS信息不太对劲
                # current_code_is_correct = True
                # if not module:
                #     status, details = "Failed (Compilation)", f"New kernel is COMPILATION INCORRECT.{err_msg}"
                #     print(f"❌ {status}")
                #     current_code_is_correct = False
                #     # continue 
                # else: 
                #     is_correct, err_str = cuda_utils.check_correctness(inputs, ref_outputs, module)
                #     if not is_correct:
                #         status, details = "Failed (Correctness)", f"New kernel is OUTPUT RESULT INCORRECT.{err_str}"
                #         print(f"❌ {status}")
                #         current_code_is_correct = False
                #         # continue 
                # if not current_code_is_correct:
                #     for i in range(5):
                #         verResult, errMessage = validate_extracted_code(new_kernel_code_full, init_inputs, inputs, ref_outputs)# 这个errMessage中对于结果错误的信息没有做前五个错误数据提取，是全部的错误数据
                #         if not verResult:
                #             print(f"尝试修正当前错误，第{i}次尝试")
                #             err_str = str(errMessage)
                #             print(f"--- Error Snippet ---\n{err_str[:500]}...\n---------------------")
                #             if len(err_str) > 4000:
                #                 err_str = err_str[:2000] + "\n...[TRUNCATED]...\n" + err_str[-2000:]
                #             new_kernel_code_full = correct_cuda_kernel(
                #                 full_pytorch_source_code,
                #                 new_kernel_code_full,
                #                 errMessage
                #             )
                #             if new_kernel_code_full:
                #                 print("Code corrected by LLM. Retrying verification...")
                #             else:
                #                 print("LLM correction failed (did not return valid code). Aborting.")
                #                 break
                #         else:
                #             current_code_is_correct = True
                #             break
                # current_module_name = current_module_name + "_verify"
                # # if not current_code_is_correct:
                # module, stdout_log, err_msg = cuda_utils.load_module(
                #     new_kernel_code_full,
                #     current_module_name,
                #     init_inputs, 
                # )
                # # print("Compilation successful.")
                
                # new_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log)# DONE3 针对21用例这里提取的PTXAS信息不太对劲
                # if not module:
                #     status, details = "Failed (Compilation)", f"New kernel is COMPILATION INCORRECT.{err_msg}"
                #     print(f"❌ {status}")
                #     continue 
                # is_correct, err_str = cuda_utils.check_correctness(inputs, ref_outputs, module)
                # if not is_correct:
                #     status, details = "Failed (Correctness)", f"New kernel is OUTPUT RESULT INCORRECT.{err_str}"
                #     print(f"❌ {status}")
                #     continue 

            except Exception as e:
                status, details = "An exception occurred during compilation or validation!", str(e)
                print(f"❌ {status}")
                continue 
                
            print("New kernel is CORRECT. Benchmarking...")
            
            new_time_ms = cuda_utils.benchmark_kernel(inputs, module)
            print("Analyzing new kernel with NCU...")
            
            new_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, 
                current_module_name, 
                inputs    
            )
            
            if new_time_ms < best_time_ms:
                status = "Success (New Best)"
                details = f"Performance improved from {best_time_ms:.3f} ms to {new_time_ms:.3f} ms."
                print(f"✅ {status} {details}")
                
                best_time_ms = new_time_ms
                
                # [!!! 已更新 !!!] 提取 CUDA-only 代码以供下次迭代
                # if new_kernel_code_full.startswith(cpp_source):
                #     new_kernel_code_cuda_only = new_kernel_code_full[len(cpp_source):].strip()
                # else:
                #     match = re.search(r'__global__\s+void\s+' + kernel_name + r'\(.*?\)\s*\{.*\}', new_kernel_code_full, re.DOTALL)
                #     if match:
                #         new_kernel_code_cuda_only = match.group(0)
                #     else:
                #         new_kernel_code_cuda_only = new_kernel_code_full # 无法分离
                #         print(f"[Warning] Round {i}: Could not auto-extract kernel. Saving full code.")

                best_kernel_code_cuda = new_kernel_code_full  
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
                        f.write(best_kernel_code_cuda)
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
            
            current_ncu_metrics = best_ncu_metrics# DONE：这里应该是best_ncu_metrics吧，之前是new_ncu_metrics

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
            # code_to_save = ""
            # if new_kernel_code_full: # 仅当 Coder 成功时
            #     if new_kernel_code_cuda_only: # 如果在 'Success' 块中已提取
            #         code_to_save = new_kernel_code_cuda_only
            #     # 否则，再次尝试提取（以防失败）
            #     elif new_kernel_code_full.startswith(cpp_source):
            #         code_to_save = new_kernel_code_full[len(cpp_source):].strip()
            #     else:
            #         match = re.search(r'__global__\s+void\s+' + kernel_name + r'\(.*?\)\s*\{.*\}', new_kernel_code_full, re.DOTALL)
            #         if match: 
            #             code_to_save = match.group(0)
            #         else: 
            #             code_to_save = new_kernel_code_full # 无法分离，保存全部
            
            # history_entry = { # 由于TODO 1 做的修改，改成下面的历史存储方式👇
            #     "round": i,# 当前的轮数
            #     "goal": opt_goal,# 生成该项中的代码的优化目标（本项中的code就是基于这个goal生成的）
            #     "status": status,# 当前code的状态
            #     "time_ms": new_time_ms if new_time_ms != float('inf') else None,# 当前code的执行时间
            #     "ptxas_metrics": new_ptxas_metrics,# 当前code的ptxas指标
            #     "all_ncu_metrics": new_ncu_metrics,# 当前code的ncu指标
            #     "selected_ncu_metrics": relevant_metric_names,# 在生成当前code的时候选择的ncu指标
            #     "details": details,
            #     "code": new_kernel_code_full# 当前code
            # }
            history_entry = {
                "round": i,
                "goal": opt_goal,
                # [!!! 新增 !!!] 保存诊断和具体的实施计划，这就是“代码改动”的语义替身
                "bottleneck_analysis": bottleneck_analysis, 
                "detailed_plan": detailed_plan,
                
                "status": status,
                "time_ms": new_time_ms if new_time_ms != float('inf') else None,
                "ptxas_metrics": new_ptxas_metrics,
                "all_ncu_metrics": new_ncu_metrics,
                "selected_ncu_metrics": relevant_metric_names,
                "details": details,
                "code": new_kernel_code_full
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
        f.write(best_kernel_code_cuda)  
    print(f"Best kernel C++/CUDA source saved to {final_kernel_path}")
    
    with open(history_file_path, 'w') as f:  
        json.dump(optimization_history, f, indent=2)
    print(f"Optimization history saved to {history_file_path}")
    
    # 5. [!!! 已更新 !!!] 返回最佳节点
    best_entry = None
    if best_time_ms != float('inf'):
         best_entry = next((h for h in reversed(optimization_history) if h.get('time_ms') == best_time_ms), None)
    
    if not best_entry: # 如果没有成功的，返回 Baseline
         best_entry = optimization_history[0] if optimization_history else {"error": "No history found."}
         
    return best_entry #返回的是最好的那个历史优化项，注意是整个项不仅仅是cuda kernel

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