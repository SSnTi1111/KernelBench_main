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
from typing import Dict # <--- [修复] 添加此导入

# 编译后的模块的全局缓存
_gemm_module = None

# vvv --- 新增：NCU 分析的目标脚本模板 (来自您的示例) --- vvv
NCU_TARGET_SCRIPT_TEMPLATE = """
import torch
import importlib.util
import os
import sys
import traceback

# 从命令行参数获取路径、模块名和矩阵大小
MODULE_PATH = sys.argv[1]
MODULE_NAME = sys.argv[2]
N = int(sys.argv[3])

try:
    # 加载由评估器编译好的 .so 模块
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None:
        print(f"Error: 无法从 {MODULE_PATH} 加载 spec", file=sys.stderr)
        sys.exit(1)
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 准备数据
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    # 使用固定的种子以确保 ncu 每次分析的数据相同
    torch.manual_seed(42) 
    A = torch.randn(N, N, device=device, dtype=torch.float32)
    B = torch.randn(N, N, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize(device)
    
    # --- 这是 NCU 将重点分析的目标 ---
    # 仅运行一次，不进行预热
    module.gemm_cuda(A, B)
    # --- 结束分析 ---
    
    torch.cuda.synchronize(device)
    # print("NCU target run complete.") # 保持安静，避免污染stdout

except Exception as e:
    print(f"NCU target script failed: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"""
# ^^^ --- 模板结束 --- ^^^


def load_gemm_module(cpp_source, cuda_source, module_name="gemm_evolved_default"):
    """
    (此函数保持不变)
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
            functions=["gemm_cuda"],
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


def run_gemm(A_tensor, B_tensor):
    """(此函数保持不变)"""
    if _gemm_module is None:
        raise RuntimeError("模块未编译。请先调用 load_gemm_module()")
    return _gemm_module.gemm_cuda(A_tensor, B_tensor)


def check_correctness(A_torch, B_torch, C_ref_torch):
    """(此函数保持不变)"""
    print("Running evolved kernel for correctness check...")
    try:
        C_evolved = run_gemm(A_torch, B_torch)
        
        is_correct = torch.allclose(C_evolved, C_ref_torch, atol=1e-3, rtol=1e-3)
        if not is_correct:
            print("--- KERNEL IS INCORRECT ---")
            print("Baseline [0,0]:", C_ref_torch[0,0].item())
            print("Evolved [0,0]:", C_evolved[0,0].item())
            print("---------------------------")
        return is_correct

    except Exception as e:
        print(f"--- KERNEL RUNTIME FAILED ---")
        print(e)
        print("-----------------------------")
        return False

# vvv --- 新增：PTXAS 解析器 (来自您的示例) --- vvv
def parse_ptxas_info(log_str: str) -> Dict[str, float]:
    """
    使用正则表达式解析 nvcc (--ptxas-options=-v) 的日志输出，
    提取寄存器、共享内存和溢出(spill)信息。
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


# vvv --- 新增：真实 NCU 分析器 (来自您的示例) --- vvv
def get_real_ncu_metrics(module_path: str, module_name: str, matrix_n: int) -> Dict[str, float]:
    """
    动态创建一个目标脚本，运行 ncu，解析 CSV 输出，并返回指标。
    """
    ncu_metrics = {}
    target_script_path = f"_ncu_target_{module_name}.py"
    
    try:
        # 1. 写入 ncu 目标脚本
        with open(target_script_path, "w", encoding="utf-8") as f:
            f.write(NCU_TARGET_SCRIPT_TEMPLATE)

        # 2. 构建 ncu 命令 (不带 --metrics 以获取全集)
        ncu_command = [
            'ncu',
            '--csv',
            '--kernel-name', 'gemm_kernel',
            '--launch-count', '1',
            '--clock-control', 'none', # 避免 ncu 锁定频率
            'python', 
            target_script_path,
            module_path, 
            module_name, 
            str(matrix_n)
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

        # 4. 解析 CSV 输出 (来自您的示例)
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
                if "gemm_kernel" not in str(row):
                    continue

                metric_name = row[metric_name_idx].strip().strip('"')
                val_str = row[metric_value_idx].strip().strip('"')
                
                if not metric_name or not val_str:
                    continue

                try:
                    # 清理指标名称 (例如：sm__warps_active.avg -> sm__warps_active.avg)
                    # 我们只保留点和下划线
                    cleaned_name = re.sub(r'[^a-zA-Z0-9_.]', '', metric_name)
                    
                    val_str_cleaned = val_str.replace(',', '')
                    if val_str_cleaned == "N/A":
                        val = 0.0
                    else:
                        val = float(val_str_cleaned)

                    ncu_metrics[cleaned_name] = val
                
                except (ValueError, IndexError):
                    # print(f"警告：解析 NCU 指标 '{metric_name}' (值: {val_str}) 失败。", file=sys.stderr)
                    pass
        
        if not ncu_metrics:
            print("警告：无法从 NCU CSV 输出中解析任何 gemm_kernel 指标数据。", file=sys.stderr)
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
            
    print(f"--- [ NCU 指标已解析 (共 {len(ncu_metrics)} 个) ] ---")
    # 随机打印5个指标作为示例
    if ncu_metrics:
        sample_keys = random.sample(list(ncu_metrics.keys()), min(5, len(ncu_metrics)))
        sample_metrics = {k: ncu_metrics[k] for k in sample_keys}
        print(json.dumps(sample_metrics, indent=2))
        
    return ncu_metrics
# ^^^ --- NCU 函数结束 --- ^^^


# vvv --- 新增：真实性能评测函数 --- vvv
def benchmark_kernel(A_tensor, B_tensor, warmup_runs=5, benchmark_runs=10):
    """
    对当前加载的 _gemm_module 执行预热和基准测试。
    """
    if _gemm_module is None:
        raise RuntimeError("模块未编译。")
    
    print(f"Warming up evolved kernel ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        _ = run_gemm(A_tensor, B_tensor)
    torch.cuda.synchronize()

    # 测量
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(benchmark_runs):
        _ = run_gemm(A_tensor, B_tensor)
    end.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start.elapsed_time(end) / benchmark_runs
    print(f"Evolved kernel benchmark: {avg_time_ms:.3f} ms")
    return avg_time_ms
# ^^^ --- 性能评测函数结束 --- ^^^


def get_pytorch_performance(A_torch, B_torch):
    """(此函数保持不变)"""
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