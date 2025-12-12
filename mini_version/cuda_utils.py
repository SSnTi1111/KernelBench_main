import torch
from torch.utils.cpp_extension import load_inline
import os
import io
import re
import config
import time
import random
import subprocess 
import csv        
import io         
import json       
import sys        
import shutil
import importlib.util 
import tempfile 
import contextlib
import warnings
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
# # 返回cuda_code 模块
# def load_gemm_module(cuda_code):
#     global _gemm_module
#     _gemm_module = 

# class FDCapturer:
#     def __init__(self):
#         self._stdout_fd = sys.stdout.fileno()
#         self._stderr_fd = sys.stderr.fileno()
#         self._saved_stdout_fd = os.dup(self._stdout_fd)
#         self._saved_stderr_fd = os.dup(self._stderr_fd)
#         self._temp_file = tempfile.TemporaryFile(mode='w+b') # 使用二进制模式避免编码问题

#     def __enter__(self):
#         # 刷新 Python 缓冲区，防止之前的输出混入
#         sys.stdout.flush()
#         sys.stderr.flush()
#         # 将 stdout (1) and stderr (2) 重定向到临时文件
#         os.dup2(self._temp_file.fileno(), self._stdout_fd)
#         os.dup2(self._temp_file.fileno(), self._stderr_fd)
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # 再次刷新
#         sys.stdout.flush()
#         sys.stderr.flush()
#         # 恢复标准输出/错误
#         os.dup2(self._saved_stdout_fd, self._stdout_fd)
#         os.dup2(self._saved_stderr_fd, self._stderr_fd)
#         os.close(self._saved_stdout_fd)
#         os.close(self._saved_stderr_fd)
    
#     def get_output(self):
#         self._temp_file.seek(0)
#         # 读取并解码
#         return self._temp_file.read().decode('utf-8', errors='replace')

# # --- 修改后的 load_module ---
# def load_module(cuda_code, module_name, init_inputs):
#     shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)# IMPORTANT：调用load_module之前强制清空缓存，因为pytorch会根据cuda_code中load_inline中的name选项是否一致判断这个是否之前编译过，如果编译过就不会编译导致获取不到PTSAX信息（但是实际上为了获取PTXAS信息重新编译会影响整个流程的时间）
#     TEST_NN_MODEL_NAME = 'ModelNew'
#     model_instance = None
#     captured_log = ""
    
#     try:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # 技巧：为了每次强制触发编译（获取PTXAS信息），
#             # 可以在这里动态修改 cuda_code 中的 name 参数，或者让 load_inline 的 name 随时间变化。
#             # 这里假设外部传入的 cuda_code 已经是唯一的，或者我们依赖清理缓存。
            
#             temp_file = os.path.join(temp_dir, "cuda_code_gen.py")
#             with open(temp_file, "w") as f:
#                 f.write(cuda_code)

#             spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, temp_file)
#             if spec is None:
#                 print("ERROR in load_module: spec is None")
#                 return None, "", ""

#             module = importlib.util.module_from_spec(spec)
#             sys.modules[TEST_NN_MODEL_NAME] = module

#             # ---------- 核心修改：使用 FD Capturer 替代 redirect_stdout ----------
#             capturer = FDCapturer()
#             try:
#                 with capturer:
#                     # exec_module 会执行 load_inline，从而触发 nvcc
#                     spec.loader.exec_module(module)
#             except Exception as e:
#                 # 即使出错，也要把捕获到的编译器报错拿出来
#                 print(f"Compilation/Execution Error: {e}")
            
#             # 获取所有底层输出 (nvcc output, ninja output, etc.)
#             captured_log = capturer.get_output()

#             model_class = getattr(module, TEST_NN_MODEL_NAME, None)
#             if model_class is None:
#                 print("ERROR: Model class not found")
#                 return None, captured_log, captured_log

#             # 实例化模型
#             try:
#                 if init_inputs is not None:
#                     if isinstance(init_inputs, (list, tuple)):
#                         model_instance = model_class(*init_inputs)
#                     elif isinstance(init_inputs, dict):
#                         model_instance = model_class(**init_inputs)
#                     else:
#                         model_instance = model_class(init_inputs)
#                 else:
#                     model_instance = model_class()
#             except Exception as e:
#                 print(f"Instantiation Error: {e}")

#     except Exception as e:
#         print(f"General Error: {e}")
    
#     # 为了兼容你原来的返回格式，这里把 log 同时赋给 stdout 和 stderr
#     # 因为 nvcc 的输出通常混合在一起，FD 捕获时也是混合的
#     return model_instance, captured_log, captured_log

# [!!! 已更新 !!!] 接受 wrapper_function_name
def load_module(cuda_code, module_name,init_inputs):
    shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)# IMPORTANT：调用load_module之前强制清空缓存，因为pytorch会根据cuda_code中load_inline中的name选项是否一致判断这个是否之前编译过，如果编译过就不会编译导致获取不到PTSAX信息（但是实际上为了获取PTXAS信息重新编译会影响整个流程的时间）
    TEST_NN_MODEL_NAME = 'ModelNew'
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "cuda_code1.py")
            with open(temp_file, "w") as f:
                f.write(cuda_code)

            spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, temp_file)
            if spec is None:
                print("ERROR in load_module 1")

            module = importlib.util.module_from_spec(spec)
            sys.modules[TEST_NN_MODEL_NAME] = module
            # ---------- 执行模块 & 捕获所有输出 ----------
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()  
            try:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    spec.loader.exec_module(module)
            except Exception as e:
                print(e)
                print("ERROR in load_module 2")

            model_class = getattr(module, TEST_NN_MODEL_NAME, None)
            if model_class is None:
                print("ERROR in load_module 3")

            # 实例化模型
            try:
                if init_inputs is not None:
                    if isinstance(init_inputs, (list, tuple)):
                        model_instance = model_class(*init_inputs)
                    elif isinstance(init_inputs, dict):
                        model_instance = model_class(**init_inputs)
                    else:
                        # 单值初始化
                        model_instance = model_class(init_inputs)
                else:
                    # 无初始化参数
                    model_instance = model_class()
            except Exception as e:
                print(e)
                print("ERROR in load_module 4")
    except Exception as e:
        print(e)
        print("ERROR in load_module 5")
    return model_instance,stdout_capture,stderr_capture

#     """
#     (此函数已更新)
#     使用PyTorch的JIT编译C++/CUDA源码。
#     返回 (module, stdout_log, stderr_log)
#     """
#     global _gemm_module
    
#     block_size = 16 
#     try:
#         match = re.search(r'#define\s+BLOCK_SIZE\s+(\d+)', cuda_source)
#         if match:
#             block_size = int(match.group(1))
#     except:
#         pass 
        
#     cuda_flags = [
#         '-O3',
#         '-allow-unsupported-compiler',
#         f'-DBLOCK_SIZE={block_size}',
#         '--ptxas-options=-v', # <--- 关键：请求 ptxas 详细输出
#         '-gencode=arch=compute_80,code=sm_80' 
#     ]

#     original_stdout_fd = os.dup(1)
#     original_stderr_fd = os.dup(2)
#     r_out, w_out = os.pipe()
#     r_err, w_err = os.pipe()
#     os.dup2(w_out, 1)
#     os.dup2(w_err, 2)
#     os.close(w_out)
#     os.close(w_err)

#     stdout_log = ""
#     stderr_log = ""
#     _module = None

#     try:
#         _module = load_inline(
#             name=module_name, 
#             cpp_sources=cpp_source,
#             cuda_sources=cuda_source,
#             functions=[wrapper_function_name], # <--- [!!! 已更新 !!!] 使用参数
#             verbose=True, # <--- 关键：必须为 True 才能捕获日志
#             extra_cflags=["-O3"],
#             extra_cuda_cflags=cuda_flags
#         )
        
#         os.dup2(original_stdout_fd, 1)
#         os.dup2(original_stderr_fd, 2)
#         stdout_bytes = os.read(r_out, 100000)
#         stderr_bytes = os.read(r_err, 100000)
#         stdout_log = stdout_bytes.decode('utf-8', errors='ignore')
#         stderr_log = stderr_bytes.decode('utf-8', errors='ignore')
        
#     except Exception as e:
#         os.dup2(original_stdout_fd, 1)
#         os.dup2(original_stderr_fd, 2)
#         stdout_bytes = os.read(r_out, 100000)
#         stderr_bytes = os.read(r_err, 100000)
#         stdout_log = stdout_bytes.decode('utf-8', errors='ignore')
#         stderr_log = stderr_bytes.decode('utf-8', errors='ignore')
        
#         detailed_error_msg = f"""CUDA C++ 扩展编译失败: {e}
# --- [ NVCC/Ninja STDOUT ] ---
# {stdout_log}
# --- [ NVCC/Ninja STDERR ] ---
# {stderr_log}
# -----------------------------
# """
#         raise RuntimeError(detailed_error_msg)

#     finally:
#         os.close(original_stdout_fd)
#         os.close(original_stderr_fd)
#         os.close(r_out)
#         os.close(r_err)

#     _gemm_module = _module
#     return _gemm_module, stdout_log, stderr_log

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
# [!!! 请替换 cuda_utils.py 中的 check_correctness 函数 !!!]

def check_correctness(inputs: List[torch.Tensor], ref_outputs: List[torch.Tensor], wrapper_function_name: str):
    """
    (此函数已更新)
    检查通用内核的正确性。
    返回: (is_correct: bool, error_msg: str)
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

        # 1. 检查输出数量
        if len(C_evolved_outputs) != len(gpu_ref_outputs):
            msg = (f"Failed (Correctness): Output count mismatch. "
                   f"Expected {len(gpu_ref_outputs)}, got {len(C_evolved_outputs)}.")
            print(f"--- KERNEL IS INCORRECT ---")
            print(msg)
            print("---------------------------")
            return False, msg

        is_correct = True
        error_msgs = []

        # 2. 逐个检查输出张量
        for i, (evolved_t, ref_t) in enumerate(zip(C_evolved_outputs, gpu_ref_outputs)):
            # 检查形状
            if evolved_t.shape != ref_t.shape:
                is_correct = False
                msg = (f"Failed (Correctness): Shape mismatch at Output {i}. "
                       f"Expected {ref_t.shape}, got {evolved_t.shape}.")
                error_msgs.append(msg)
                print(msg)
                continue # 继续检查下一个输出，或者直接返回也可以

            # 检查数值 (atol=1e-2, rtol=1e-2)
            if not torch.allclose(evolved_t, ref_t, atol=1e-2, rtol=1e-2):
                is_correct = False
                
                # --- [核心修改] 捕获前 5 个错误值 ---
                diff = torch.abs(evolved_t - ref_t)
                # 计算允许的误差范围
                tol = 1e-2 + 1e-2 * torch.abs(ref_t)
                # 找出超出误差的掩码
                error_mask = diff > tol
                # 获取错误索引
                error_indices = torch.nonzero(error_mask, as_tuple=False)
                num_errors = error_indices.size(0)
                
                msg_header = f"Failed (Correctness): Output {i} has {num_errors} mismatches (total elements: {ref_t.numel()})."
                error_details = [msg_header]
                error_details.append("Top 5 Mismatches (Index | Reference Value | Actual Value):")
                
                # 取前 5 个
                for j in range(min(5, num_errors)):
                    idx = error_indices[j]
                    idx_tuple = tuple(idx.tolist())
                    ref_val = ref_t[idx_tuple].item()
                    act_val = evolved_t[idx_tuple].item()
                    error_details.append(f"  [{j}] Index: {idx_tuple} | Ref: {ref_val:.6f} | Act: {act_val:.6f}")
                
                full_msg = "\n".join(error_details)
                error_msgs.append(full_msg)
                
                print(f"--- KERNEL IS INCORRECT (Output {i}) ---")
                print(full_msg)
                print("---------------------------")
                # 只要发现一个输出不对，通常就可以返回了，或者收集所有错误
                # 这里我们收集第一个主要错误后直接返回，避免 Prompt 过长
                return False, full_msg

        if is_correct:
            return True, ""
        else:
            # 只有形状错误会走到这里
            return False, "\n".join(error_msgs)

    except Exception as e:
        err_str = f"Runtime Error during check_correctness: {e}\n{traceback.format_exc()}"
        print(f"--- KERNEL RUNTIME FAILED ---")
        print(err_str)
        print("-----------------------------")
        return False, err_str
        
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