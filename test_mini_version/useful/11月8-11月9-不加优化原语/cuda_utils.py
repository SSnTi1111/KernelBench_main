import torch
from torch.utils.cpp_extension import load_inline
import os
import re
import config
import time
import random
import subprocess 
import csv        
import io         
import json       
import sys        
import importlib.util 
import traceback  
import numpy as np  
from typing import Dict, List # <--- [修复] 添加 List

# 编译后的模块的全局缓存
_gemm_module = None

# vvv --- [!!! 已更新 !!!] NCU 模板现在是通用的 --- vvv
NCU_TARGET_SCRIPT_TEMPLATE = """
import torch
import importlib.util
import os
import sys
import traceback

# 从命令行参数获取路径、模块名和 wrapper 名
MODULE_PATH = sys.argv[1]
MODULE_NAME = sys.argv[2]
WRAPPER_FUNCTION_NAME = sys.argv[3]
# [!!! 已移除 !!!] N = int(sys.argv[3])

try:
    # 加载由评估器编译好的 .so 模块
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None:
        print(f"Error: 无法从 {MODULE_PATH} 加载 spec", file=sys.stderr)
        sys.exit(1)
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # [!!! 已更新 !!!] 从文件加载真实的输入数据
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    
    try:
        # 从保存的文件中加载输入
        inputs = torch.load("_ncu_inputs.pt")
        gpu_inputs = [t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs]
    except Exception as e:
        print(f"Failed to load _ncu_inputs.pt: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    torch.cuda.synchronize(device)
    
    # --- 这是 NCU 将重点分析的目标 ---
    # 仅运行一次，不进行预热
    
    # [!!! 已更新 !!!] 使用 getattr 动态调用 wrapper
    wrapper_func = getattr(module, WRAPPER_FUNCTION_NAME)
    wrapper_func(*gpu_inputs)
    # --- 结束分析 ---
    
    torch.cuda.synchronize(device)
    # print("NCU target run complete.")

except Exception as e:
    print(f"NCU target script failed: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"""
# ^^^ --- 模板结束 --- ^^^


# [!!! 已更新 !!!] 接受 wrapper_function_name
def load_gemm_module(cpp_source, cuda_source, module_name="gemm_evolved_default", wrapper_function_name="gemm_cuda"):
    """
    (此函数已更新)
    使用PyTorch的JIT编译C++/CUDA源码。
    返回 (module, stdout_log, stderr_log)
    """
    global _gemm_module
    
    block_size = 16 
    try:
        match = re.search(r'#define\s+BLOCK_SIZE\s+(\d+)', cuda_source)
        if match:
            block_size = int(match.group(1))
    except:
        pass 
        
    cuda_flags = [
        '-O3',
        '-allow-unsupported-compiler',
        f'-DBLOCK_SIZE={block_size}',
        '--ptxas-options=-v', # <--- 关键：请求 ptxas 详细输出
        '-gencode=arch=compute_80,code=sm_80' 
    ]

    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    r_out, w_out = os.pipe()
    r_err, w_err = os.pipe()
    os.dup2(w_out, 1)
    os.dup2(w_err, 2)
    os.close(w_out)
    os.close(w_err)

    stdout_log = ""
    stderr_log = ""
    _module = None

    try:
        _module = load_inline(
            name=module_name, 
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=[wrapper_function_name], # <--- [!!! 已更新 !!!] 使用参数
            verbose=True, # <--- 关键：必须为 True 才能捕获日志
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags
        )
        
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        stdout_bytes = os.read(r_out, 100000)
        stderr_bytes = os.read(r_err, 100000)
        stdout_log = stdout_bytes.decode('utf-8', errors='ignore')
        stderr_log = stderr_bytes.decode('utf-8', errors='ignore')
        
    except Exception as e:
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        stdout_bytes = os.read(r_out, 100000)
        stderr_bytes = os.read(r_err, 100000)
        stdout_log = stdout_bytes.decode('utf-8', errors='ignore')
        stderr_log = stderr_bytes.decode('utf-8', errors='ignore')
        
        detailed_error_msg = f"""CUDA C++ 扩展编译失败: {e}
--- [ NVCC/Ninja STDOUT ] ---
{stdout_log}
--- [ NVCC/Ninja STDERR ] ---
{stderr_log}
-----------------------------
"""
        raise RuntimeError(detailed_error_msg)

    finally:
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)
        os.close(r_out)
        os.close(r_err)

    _gemm_module = _module
    return _gemm_module, stdout_log, stderr_log

# [!!! 已更新 !!!] 接受通用输入
def run_gemm(inputs: List[torch.Tensor], wrapper_function_name: str):
    """
    (此函数已更新)
    运行当前加载的模块。
    """
    if _gemm_module is None:
        raise RuntimeError("模块未编译。请先调用 load_gemm_module()")
    
    # 使用 getattr 动态调用 wrapper
    wrapper_func = getattr(_gemm_module, wrapper_function_name)
    return wrapper_func(*inputs)

# [!!! 已更新 !!!] 接受通用输入和引用
def check_correctness(inputs: List[torch.Tensor], ref_outputs: List[torch.Tensor], wrapper_function_name: str):
    """
    (此函数已更新)
    检查通用内核的正确性。
    """
    print("Running evolved kernel for correctness check...")
    try:
        # 确保输入在 GPU 上
        gpu_inputs = [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in inputs]
        gpu_ref_outputs = [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in ref_outputs]

        C_evolved_outputs = run_gemm(gpu_inputs, wrapper_function_name)
        
        # 确保 C_evolved_outputs 是一个列表，以便进行 zip
        if not isinstance(C_evolved_outputs, (list, tuple)):
            C_evolved_outputs = [C_evolved_outputs]

        if len(C_evolved_outputs) != len(gpu_ref_outputs):
            print(f"--- KERNEL IS INCORRECT ---")
            print(f"Error: Output count mismatch. Expected {len(gpu_ref_outputs)}, got {len(C_evolved_outputs)}.")
            print("---------------------------")
            return False

        is_correct = True
        for i, (evolved_t, ref_t) in enumerate(zip(C_evolved_outputs, gpu_ref_outputs)):
            if not torch.allclose(evolved_t, ref_t, atol=1e-3, rtol=1e-3):
                is_correct = False
                print(f"--- KERNEL IS INCORRECT (Output {i}) ---")
                print("Ref [0,0]:", ref_t.flatten()[0].item())
                print("Evolved [0,0]:", evolved_t.flatten()[0].item())
                print("---------------------------")
                break # 
        
        return is_correct

    except Exception as e:
        print(f"--- KERNEL RUNTIME FAILED ---")
        print(e)
        traceback.print_exc()
        print("-----------------------------")
        return False

# vvv --- PTXAS 解析器 (保持不变) --- vvv
def parse_ptxas_info(log_str: str) -> Dict[str, float]:
    """
    (此函数保持不变)
    """
    metrics = {
        'registers_used': 0.0,
        'shared_mem_bytes': 0.0,
        'spill_bytes': 0.0, # (加载+存储)
    }
    
    try:
        # 匹配 "Used XX registers"
        reg_match = re.search(r'Used\s+(\d+)\s+registers', log_str)
        if reg_match:
            metrics['registers_used'] = float(reg_match.group(1))

        # 匹配 shared memory (smem)
        smem_match = re.search(r'(\d+)\s+bytes\s+smem', log_str)
        if smem_match:
            metrics['shared_mem_bytes'] = float(smem_match.group(1))

        # 匹配 "Z bytes spill stores/loads"
        spill_stores_match = re.search(r'(\d+)\s+bytes\s+spill\s+stores', log_str)
        spill_loads_match = re.search(r'(\d+)\s+bytes\s+spill\s+loads', log_str)
        
        spill_bytes = 0.0
        if spill_stores_match:
            spill_bytes += float(spill_stores_match.group(1))
        if spill_loads_match:
            spill_bytes += float(spill_loads_match.group(1))
            
        metrics['spill_bytes'] = spill_bytes

    except Exception as e:
        print(f"警告：解析 PTXAS 日志失败: {e}", file=sys.stderr)
    
    print(f"--- [ PTXAS Metrics Parsed ] ---")
    print(json.dumps(metrics, indent=2))
    return metrics
# ^^^ --- PTXAS 解析器结束 --- ^^^


# vvv --- [!!! 已更新 !!!] 真实 NCU 分析器 (现在是通用的) --- vvv
def get_real_ncu_metrics(module_path: str, module_name: str, kernel_name: str, wrapper_function_name: str, inputs: List[torch.Tensor]) -> Dict[str, float]:
    """
    动态创建一个目标脚本，运行 ncu，解析 CSV 输出，并返回指标。
    [!!! 已更新 !!!] 接受通用输入和内核/wrapper 名称。
    """
    ncu_metrics = {}
    target_script_path = f"_ncu_target_{module_name}.py"
    
    try:
        # 1. 写入 ncu 目标脚本
        with open(target_script_path, "w", encoding="utf-8") as f:
            f.write(NCU_TARGET_SCRIPT_TEMPLATE)

        # [!!! 已更新 !!!] 保存输入以供 ncu 脚本加载
        torch.save(inputs, '_ncu_inputs.pt')

        # 2. 构建 ncu 命令 (不带 --metrics 以获取全集)
        ncu_command = [
            'ncu',
            '--csv',
            # '--kernel-name', kernel_name, # <--- [!!! 已删除 !!!]
            '--launch-count', '1',
            '--clock-control', 'none', # 避免 ncu 锁定频率
            'python', 
            target_script_path,
            module_path, 
            module_name, 
            wrapper_function_name # <--- [!!! 已更新 !!!]
            # [!!! 已移除 !!!] str(matrix_n)
        ]
        
        print(f"--- [ 正在运行 NCU (全集)... ] ---")
        # print(f"命令: {' '.join(ncu_command)}") # 调试时取消注释

        # 3. 运行 ncu
        proc = subprocess.run(
            ncu_command, 
            capture_output=True, 
            text=True, 
            encoding="utf-8", 
            errors="ignore",
            timeout=300 # NCU (全集) 可能非常慢
        )

        if proc.returncode != 0:
            print(f"警告：NCU 运行失败。返回码: {proc.returncode}", file=sys.stderr)
            print(f"NCU Stderr: {proc.stderr}", file=sys.stderr)
            return ncu_metrics

        # 4. 解析 CSV 输出
        csv_reader = csv.reader(io.StringIO(proc.stdout))
        metric_name_idx = -1
        metric_value_idx = -1

        for row in csv_reader:
            if "Metric Name" in row and "Metric Value" in row:
                header = [h.strip().strip('"') for h in row]
                try:
                    metric_name_idx = header.index("Metric Name")
                    metric_value_idx = header.index("Metric Value")
                except ValueError:
                    print(f"警告：在 NCU CSV 表头中找不到 'Metric Name' 或 'Metric Value'。", file=sys.stderr)
                    return ncu_metrics
                continue 

            if metric_name_idx != -1 and len(row) > max(metric_name_idx, metric_value_idx):
                
                # [!!! 已删除 !!!] 
                # if kernel_name not in str(row):
                #     continue

                metric_name = row[metric_name_idx].strip().strip('"')
                val_str = row[metric_value_idx].strip().strip('"')
                
                if not metric_name or not val_str:
                    continue

                try:
                    # 清理指标名称
                    cleaned_name = re.sub(r'[^a-zA-Z0-9_.]', '', metric_name)
                    
                    val_str_cleaned = val_str.replace(',', '')
                    if val_str_cleaned == "N/A":
                        val = 0.0
                    else:
                        val = float(val_str_cleaned)

                    ncu_metrics[cleaned_name] = val
                
                except (ValueError, IndexError):
                    pass
        
        if not ncu_metrics:
            print(f"警告：无法从 NCU CSV 输出中解析任何 {kernel_name} 指标数据。", file=sys.stderr)
            # print(f"NCU STDOUT: {proc.stdout}") # 调试时取消注释
            # print(f"NCU STDERR: {proc.stderr}") # 调试时取消注释
            return ncu_metrics

    except FileNotFoundError:
        print("="*50, file=sys.stderr)
        print("评估器错误：找不到 'ncu' (Nsight Compute)。", file=sys.stderr)
        print("请确保 NVIDIA Nsight Compute 已安装并在您的系统 PATH 中。", file=sys.stderr)
        print("="*50, file=sys.stderr)
        sys.exit(1) # 这是一个关键错误，终止程序
    except Exception as e:
        print(f"警告：NCU 分析期间发生意外错误: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    
    finally:
        if os.path.exists(target_script_path):
            os.remove(target_script_path)
        # [!!! 新增 !!!] 清理 ncu 输入文件
        if os.path.exists("_ncu_inputs.pt"):
            os.remove("_ncu_inputs.pt")
            
    print(f"--- [ NCU 指标已解析 (共 {len(ncu_metrics)} 个) ] ---")
    if ncu_metrics:
        sample_keys = random.sample(list(ncu_metrics.keys()), min(5, len(ncu_metrics)))
        sample_metrics = {k: ncu_metrics[k] for k in sample_keys}
        print(json.dumps(sample_metrics, indent=2))
        
    return ncu_metrics
# ^^^ --- NCU 函数结束 --- ^^^


# vvv --- [!!! 已更新 !!!] 真实性能评测函数 (现在是通用的) --- vvv
def benchmark_kernel(inputs: List[torch.Tensor], wrapper_function_name: str, warmup_runs=5, benchmark_runs=10):
    """
    对当前加载的 _gemm_module 执行预热和基准测试。
    [!!! 已更新 !!!] 接受通用输入。
    """
    if _gemm_module is None:
        raise RuntimeError("模块未编译。")
    
    gpu_inputs = [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in inputs]

    print(f"Warming up evolved kernel ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        _ = run_gemm(gpu_inputs, wrapper_function_name)
    torch.cuda.synchronize()

    # 测量
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(benchmark_runs):
        _ = run_gemm(gpu_inputs, wrapper_function_name)
    end.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start.elapsed_time(end) / benchmark_runs
    print(f"Evolved kernel benchmark: {avg_time_ms:.3f} ms")
    return avg_time_ms
# ^^^ --- 性能评测函数结束 --- ^^^


def get_pytorch_performance(A_torch, B_torch):
    """(此函数保持不变, 仅用于原始 main() 的后向兼容)"""
    print("Warming up PyTorch...")
    for _ in range(10):
        _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(50):
        _ = torch.matmul(A_torch, B_torch)
    end.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start.elapsed_time(end) / 50
    return avg_time_ms