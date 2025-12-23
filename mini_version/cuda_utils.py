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
import weakref 
import numpy as np  
from typing import Dict, List, Any # <--- [修复] 添加 List
import gc
import copy
import torch.nn as nn
from collections import defaultdict

# 编译后的模块的全局缓存
_gemm_module = None
# data_type_info = ""
# vvv --- [!!! 已更新 !!!] NCU 模板现在是通用的 --- vvv
# NCU_TARGET_SCRIPT_TEMPLATE = """
# import torch
# import importlib.util
# import os
# import sys
# import traceback

# # 1. 获取命令行参数 (只期望 2 个参数: 路径和模块名)
# MODULE_PATH = sys.argv[1]
# MODULE_NAME = sys.argv[2]
# # WRAPPER_FUNCTION_NAME = sys.argv[3] # <--- [已移除] 不再需要

# try:
#     # 2. 加载模块
#     spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
#     if spec is None:
#         print(f"Error: 无法从 {MODULE_PATH} 加载 spec", file=sys.stderr)
#         sys.exit(1)
        
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)

#     # 3. 准备设备和数据
#     torch.cuda.set_device(0)
#     device = torch.device("cuda")
    
#     try:
#         # 从保存的文件中加载输入
#         inputs = torch.load("_ncu_inputs.pt")
#         # 确保输入移动到 GPU
#         gpu_inputs = [t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs]
#     except Exception as e:
#         print(f"Failed to load _ncu_inputs.pt: {e}", file=sys.stderr)
#         traceback.print_exc()
#         sys.exit(1)

#     # 4. 实例化模型 (ModelNew)
#     # 注意：这里假设 ModelNew 的 __init__ 不需要参数，或者参数已硬编码。
#     # 对于 Level 1 的问题，生成的代码通常遵循这一模式。
#     if not hasattr(module, 'ModelNew'):
#         print(f"Error: 模块 {MODULE_NAME} 中未找到 'ModelNew' 类", file=sys.stderr)
#         sys.exit(1)
        
#     try:
#         model = module.ModelNew()
#         model.to(device)
#         model.eval() # 切换到评估模式 (影响某些 layers 如 Dropout/BatchNorm)
#     except Exception as e:
#         print(f"Error: 实例化 ModelNew 失败: {e}", file=sys.stderr)
#         traceback.print_exc()
#         sys.exit(1)

#     torch.cuda.synchronize(device)
    
#     # --- 5. 运行目标 (NCU 分析区域) ---
#     # 仅运行一次，不进行预热 (NCU 不需要预热，且 launch-count=1)
    
#     try:
#         model(*gpu_inputs)
#     except Exception as e:
#         print(f"Error: 模型执行失败: {e}", file=sys.stderr)
#         traceback.print_exc()
#         sys.exit(1)
        
#     # --- 结束分析 ---
    
#     torch.cuda.synchronize(device)

# except Exception as e:
#     print(f"NCU target script failed: {e}", file=sys.stderr)
#     traceback.print_exc()
#     sys.exit(1)
# """

NCU_TARGET_SCRIPT_TEMPLATE = """
import torch
import importlib.util
import os
import sys
import traceback

# 1. 获取命令行参数
MODULE_PATH = sys.argv[1]
MODULE_NAME = sys.argv[2]

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

try:
    # 2. 加载模块
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None:
        sys.exit(1)
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 3. 准备设备和数据
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    
    try:
        # 加载推理输入
        inputs = torch.load("_ncu_inputs.pt")
        # [核心修改] 加载模型初始化参数
        init_inputs = []
        if os.path.exists("_ncu_init_inputs.pt"):
            init_inputs = torch.load("_ncu_init_inputs.pt")
            
        gpu_inputs = [move_to_cuda(t) for t in inputs]
    except Exception as e:
        print(f"Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. 实例化模型 (ModelNew)
    if not hasattr(module, 'ModelNew'):
        sys.exit(1)
        
    try:
        # [核心修改] 使用 init_inputs 实例化模型，解决参数缺失问题
        if isinstance(init_inputs, (list, tuple)):
            model = module.ModelNew(*init_inputs)
        elif isinstance(init_inputs, dict):
            model = module.ModelNew(**init_inputs)
        else:
            model = module.ModelNew(init_inputs)
            
        model.to(device)
        model.eval() 
    except Exception as e:
        print(f"Error: 实例化 ModelNew 失败: {e}", file=sys.stderr)
        sys.exit(1)

    torch.cuda.synchronize(device)

    for _ in range(5):
        model(*gpu_inputs)
    torch.cuda.synchronize()
    
    # 5. 运行目标 (NCU 分析区域)
    print("Start Profiling...")
    try:
        torch.cuda.cudart().cudaProfilerStart()
        model(*gpu_inputs)
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    except Exception as e:
        print(f"Error: 模型执行失败: {e}", file=sys.stderr)
        sys.exit(1)
    print("Stop Profiling.")

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
"""

def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """获取模型中所有参数和缓冲区的扁平化字典"""
    named: dict[str, torch.Tensor] = {}
    for k, p in model.named_parameters(recurse=True):
        named[f"param::{k}"] = p
    for k, b in model.named_buffers(recurse=True):
        named[f"buffer::{k}"] = b
    return named

@torch.no_grad()
def _safe_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """尝试直接拷贝（形状必须完全一致）"""
    if dst.shape != src.shape:
        return False
    dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    return True

@torch.no_grad()
def _try_map_shape_and_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """
    尝试处理形状不匹配的情况（例如生成的 Kernel 使用了不同的内存布局）。
    支持：转置、压缩维度等常见操作。
    """
    s = tuple(src.shape)
    d = tuple(dst.shape)

    # 1. 完全相同，直接拷
    if s == d:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
        return True

    # 2. 5D 权重首两维交换 (常见于 Conv3d: Out,In,... <-> In,Out,...)
    if len(s) == 5 and len(d) == 5 and s[0] == d[1] and s[1] == d[0] and s[2:] == d[2:]:
        dst.copy_(src.permute(1, 0, 2, 3, 4).contiguous().to(dtype=dst.dtype, device=dst.device))
        return True

    # 3. 压缩/解压维度 (例如 Linear 的 weight 是 2D，但某些 Conv 实现可能是 4D (Out, In, 1, 1))
    if src.numel() == dst.numel():
        # 尝试 reshape 后拷贝
        try:
            dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
            return True
        except:
            pass
            
    return False

@torch.no_grad()
def align_params_smart(ref_model: nn.Module, test_model: nn.Module):
    """
    智能对齐参数：
    1. 优先尝试同名拷贝。
    2. 如果名字对不上，尝试通过“唯一形状匹配”来拷贝。
    """
    if ref_model is None:
        return

    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)
    aligned_test_keys = set()

    print("--- Syncing Weights (Smart Alignment) ---")

    # 1. 策略 A：同名同形状 (Name Match)
    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None:
            if _try_map_shape_and_copy_(t_dst, t_src):
                aligned_test_keys.add(name)
                # print(f"  [Sync] Matched by name: {name}")

    # 2. 策略 B：唯一形状匹配 (Unique Shape Match)
    # 如果生成的代码改了层名字（比如 self.conv 改成了 self.conv1），load_state_dict 会失败。
    # 这里通过形状来“猜”对应关系。
    shape2ref = defaultdict(list)
    shape2test = defaultdict(list)
    
    for n, t in ref_named.items():
        shape2ref[tuple(t.shape)].append((n, t))
    
    for n, t in test_named.items():
        if n not in aligned_test_keys: # 只处理还没对齐的
            shape2test[tuple(t.shape)].append((n, t))

    for shp, items in shape2test.items():
        # 如果这个形状在 ref 和 test 中都只出现了一次，那它们肯定是一对！
        if len(items) == 1 and len(shape2ref.get(shp, [])) == 1:
            tname_dst, t_dst = items[0]
            _, t_src = shape2ref[shp][0]
            if _safe_copy_(t_dst, t_src):
                aligned_test_keys.add(tname_dst)
                print(f"  [Sync] Matched by unique shape: {shp}")

    # 统计
    print(f"  Synced {len(aligned_test_keys)} / {len(test_named)} tensors.")

# --- 1. 添加 FDCapturer 类 ---
class FDCapturer:
    def __init__(self):
        self._stdout_fd = sys.stdout.fileno()
        self._stderr_fd = sys.stderr.fileno()
        # 保存原始的文件描述符
        self._saved_stdout_fd = os.dup(self._stdout_fd)
        self._saved_stderr_fd = os.dup(self._stderr_fd)
        # 创建一个临时文件来接收输出
        self._temp_file = tempfile.TemporaryFile(mode='w+b')

    def __enter__(self):
        # 刷新 Python 缓冲区，防止之前的 Python 输出混入
        sys.stdout.flush()
        sys.stderr.flush()
        # 将 stdout (1) 和 stderr (2) 重定向到临时文件
        os.dup2(self._temp_file.fileno(), self._stdout_fd)
        os.dup2(self._temp_file.fileno(), self._stderr_fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 再次刷新，确保所有底层输出都写进文件
        sys.stdout.flush()
        sys.stderr.flush()
        # 恢复标准输出/错误
        os.dup2(self._saved_stdout_fd, self._stdout_fd)
        os.dup2(self._saved_stderr_fd, self._stderr_fd)
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
    
    def get_output(self):
        # 回到文件开头读取所有内容
        self._temp_file.seek(0)
        return self._temp_file.read().decode('utf-8', errors='replace')

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

# --- 2. 修改 load_module ---
def load_module(cuda_code, module_name, init_inputs, ref_model):
    # 1. 强制清理缓存
    shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)
    
    TEST_NN_MODEL_NAME = 'ModelNew'
    model_instance = None
    captured_log = ""
    
    # [持久化路径] 必须使用绝对路径，因为 module.__file__ 需要它
    file_path = os.path.abspath(f"{module_name}.py")
    
    try:
        # --- 动态重命名逻辑 ---
        timestamp = int(time.time() * 1000)
        pattern = r"(name\s*=\s*['\"])([\w_]+)(['\"])"
        
        def replace_func(match):
            prefix = match.group(1)
            old_name = match.group(2)
            suffix = match.group(3)
            new_name = f"{old_name}_{timestamp}"
            # print(f"[DEBUG] Renaming: {old_name} -> {new_name}")
            return f"{prefix}{new_name}{suffix}"
            
        cuda_code_modified = re.sub(pattern, replace_func, cuda_code, count=1)
        
        # 2. 写入文件
        with open(file_path, "w") as f:
            f.write(cuda_code_modified)

        # 3. 加载模块
        spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, file_path)
        if spec is None:
            print("ERROR in load_module: spec is None")
            # 如果加载失败，立即清理文件
            if os.path.exists(file_path):
                os.remove(file_path)
            return None, "", ""

        module = importlib.util.module_from_spec(spec)
        sys.modules[TEST_NN_MODEL_NAME] = module

        # 4. 编译 & 捕获输出
        capturer = FDCapturer()
        try:
            with capturer:
                spec.loader.exec_module(module)
        except Exception as e:
            print(f"Compilation Error: {e}")
            # 编译失败也清理文件
            if os.path.exists(file_path):
                os.remove(file_path)
            err_msg = extract_error_and_next_line(capturer.get_output())
            # 依然返回日志供分析
            return None, capturer.get_output(), err_msg
        
        captured_log = capturer.get_output()
        
        # 5. 实例化模型
        model_class = getattr(module, TEST_NN_MODEL_NAME, None)
        if model_class is None:
            print("ERROR: Model class not found")
            if os.path.exists(file_path):
                os.remove(file_path)
            return None, captured_log, captured_log

        try:
            if init_inputs is not None:
                if isinstance(init_inputs, (list, tuple)):
                    model_instance = model_class(*init_inputs)
                elif isinstance(init_inputs, dict):
                    model_instance = model_class(**init_inputs)
                else:
                    model_instance = model_class(init_inputs)
            else:
                model_instance = model_class()
                
            if torch.cuda.is_available():
                    model_instance = model_instance.cuda()

            if ref_model is not None:
                try:
                    align_params_smart(ref_model, model_instance)
                except Exception as e:
                    print(f"[Warning] Smart weight sync failed: {e}")
                    # 即使同步失败，也让它继续跑，说不定 LLM 运气好
            
            # [!!! 关键修复 1 !!!] 绑定 __file__ 属性，供 main.py 中的 NCU 使用
            model_instance.__file__ = file_path
            
            # [!!! 关键修复 2 !!!] 注册自动清理钩子
            # 当 model_instance 被 del 或 垃圾回收时，自动执行 lambda 删除文件
            weakref.finalize(
                model_instance, 
                lambda p=file_path: os.remove(p) if os.path.exists(p) else None
            )
            
        except Exception as e:
            print(f"Instantiation Error: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return None, captured_log, captured_log

    except Exception as e:
        print(f"General Error inside load_module: {e}")
        traceback.print_exc()
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return model_instance, captured_log, captured_log

# def load_module(cuda_code, module_name, init_inputs):
#     # 1. 强制清理 Torch 扩展缓存 (保持原有逻辑)
#     shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)
    
#     TEST_NN_MODEL_NAME = 'ModelNew'
#     model_instance = None
#     captured_log = ""
#     err_msg = ""
    
#     # 声明变量以便在 finally 中清理
#     module = None
#     spec = None
#     capturer = None
    
#     # [持久化路径]
#     file_path = os.path.abspath(f"{module_name}.py")
    
#     try:
#         # --- 动态重命名逻辑 ---
#         timestamp = int(time.time() * 1000)
#         pattern = r"(name\s*=\s*['\"])([\w_]+)(['\"])"
        
#         def replace_func(match):
#             prefix = match.group(1)
#             old_name = match.group(2)
#             suffix = match.group(3)
#             new_name = f"{old_name}_{timestamp}"
#             return f"{prefix}{new_name}{suffix}"
            
#         cuda_code_modified = re.sub(pattern, replace_func, cuda_code, count=1)
        
#         # 2. 写入文件
#         with open(file_path, "w") as f:
#             f.write(cuda_code_modified)

#         # 3. 加载模块
#         spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, file_path)
#         if spec is None:
#             print("ERROR in load_module: spec is None")
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#             return None, "", ""

#         module = importlib.util.module_from_spec(spec)
        
#         # [注意] 这里注册到全局，如果不清理，模块永远存活
#         sys.modules[TEST_NN_MODEL_NAME] = module

#         # 4. 编译 & 捕获输出
#         capturer = FDCapturer()
#         try:
#             with capturer:
#                 spec.loader.exec_module(module)
#         except Exception as e:
#             print(f"Compilation Error: {e}")
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#             err_msg = extract_error_and_next_line(capturer.get_output())
#             return None, capturer.get_output(), err_msg
        
#         captured_log = capturer.get_output()
        
#         # 5. 实例化模型
#         model_class = getattr(module, TEST_NN_MODEL_NAME, None)
#         if model_class is None:
#             print("ERROR: Model class not found")
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#             return None, captured_log, captured_log

#         try:
#             if init_inputs is not None:
#                 if isinstance(init_inputs, (list, tuple)):
#                     model_instance = model_class(*init_inputs)
#                 elif isinstance(init_inputs, dict):
#                     model_instance = model_class(**init_inputs)
#                 else:
#                     model_instance = model_class(init_inputs)
#             else:
#                 model_instance = model_class()
            
#             # 绑定文件路径
#             model_instance.__file__ = file_path
            
#             # 注册自动清理钩子 (当 model_instance 销毁时删除文件)
#             weakref.finalize(
#                 model_instance, 
#                 lambda p=file_path: os.remove(p) if os.path.exists(p) else None
#             )
            
#         except Exception as e:
#             print(f"Instantiation Error: {e}")
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#             return None, captured_log, captured_log

#     except Exception as e:
#         print(f"General Error inside load_module: {e}")
#         traceback.print_exc()
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         return None, "", str(e)

#     finally:
#         # ==========================================================
#         # [核心优化] 函数退出前的强力垃圾回收
#         # ==========================================================
        
#         # 1. 从全局 sys.modules 中移除模块引用
#         # model_instance 已经实例化，它内部的 __class__ 会持有 module 的引用，
#         # 所以只要 model_instance 活着，代码就能跑。
#         # 但从 sys.modules 移除后，当 model_instance 死亡时，module 也会随之死亡。
#         if TEST_NN_MODEL_NAME in sys.modules:
#             del sys.modules[TEST_NN_MODEL_NAME]

#         # 2. 删除局部大对象引用
#         if 'cuda_code_modified' in locals(): del cuda_code_modified
#         if 'capturer' in locals() and capturer is not None: del capturer
#         if 'spec' in locals(): del spec
        
#         # 注意：不要 del model_instance，这是我们要返回的！
#         # 也不要 del captured_log，这是要返回的日志 (str类型占内存不大，可以接受)

#         # 3. 显式断开 module 引用
#         if module is not None:
#             del module
            
#         # 4. 强制触发垃圾回收
#         # 这会清理掉刚才产生的编译图、AST对象等循环引用
#         gc.collect()

#     return model_instance, captured_log, captured_log

# [!!! 已更新 !!!] 接受 wrapper_function_name
# def load_module(cuda_code, module_name,init_inputs):
#     shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)# IMPORTANT：调用load_module之前强制清空缓存，因为pytorch会根据cuda_code中load_inline中的name选项是否一致判断这个是否之前编译过，如果编译过就不会编译导致获取不到PTSAX信息（但是实际上为了获取PTXAS信息重新编译会影响整个流程的时间）
#     TEST_NN_MODEL_NAME = 'ModelNew'
#     try:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             temp_file = os.path.join(temp_dir, "cuda_code1.py")
#             with open(temp_file, "w") as f:
#                 f.write(cuda_code)

#             spec = importlib.util.spec_from_file_location(TEST_NN_MODEL_NAME, temp_file)
#             if spec is None:
#                 print("ERROR in load_module 1")

#             module = importlib.util.module_from_spec(spec)
#             sys.modules[TEST_NN_MODEL_NAME] = module
#             # ---------- 执行模块 & 捕获所有输出 ----------
#             stdout_capture = io.StringIO()
#             stderr_capture = io.StringIO()  
#             try:
#                 with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
#                     spec.loader.exec_module(module)
#             except Exception as e:
#                 print(e)
#                 print("ERROR in load_module 2")

#             model_class = getattr(module, TEST_NN_MODEL_NAME, None)
#             if model_class is None:
#                 print("ERROR in load_module 3")

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
#                 print(e)
#                 print("ERROR in load_module 4")
#     except Exception as e:
#         print(e)
#         print("ERROR in load_module 5")
#     return model_instance,stdout_capture,stderr_capture

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
def run_gemm(inputs, module):
    """
    (此函数已更新)
    运行当前加载的模块。
    """
    # if module is None:
    #     raise RuntimeError("模块未编译。请先调用 load_module()")
    
    # 使用 getattr 动态调用 wrapper
    # wrapper_func = getattr(_gemm_module, wrapper_function_name)
    
    return module(*inputs)



# def check_correctness(inputs, ref_outputs, module):
#     """
#     (此函数已更新)
#     检查通用内核的正确性。
#     返回: (is_correct: bool, error_msg: str)
#     """
#     print("Running evolved kernel for correctness check...")
#     data_type_info = ""
#     try:
#         # 确保输入在 GPU 上
#         # gpu_inputs = [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in inputs]
#         # gpu_ref_outputs = [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in ref_outputs]

#         C_evolved_outputs = run_gemm(inputs, module)
        
#         # 确保 C_evolved_outputs 是一个列表，以便进行 zip
#         # if not isinstance(C_evolved_outputs, (list, tuple)):
#         #     C_evolved_outputs = [C_evolved_outputs]

#         # 1. 检查输出数量
#         if len(C_evolved_outputs) != len(ref_outputs):
#             msg = (f"Failed (Correctness): Output count mismatch. "
#                    f"Expected {len(ref_outputs)}, got {len(C_evolved_outputs)}.")
#             print(f"--- KERNEL IS INCORRECT ---")
#             print(msg)
#             print("---------------------------")
#             return False, msg

#         is_correct = True
#         error_msgs = []

#         def compare_outputs(a, b, atol=1e-2, rtol=1e-2):
#             # global data_type_info
#             # tuple 情况
#             if isinstance(a, tuple) and isinstance(b, tuple):
#                 if len(a) != len(b):
#                     return False
#                 return all(compare_outputs(x, y, atol, rtol) for x, y in zip(a, b))

#             # tensor 对 tensor
#             if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
#                 return torch.allclose(a, b, atol=atol, rtol=rtol)

#             # # 标量对标量
#             if isinstance(a, (int, float)) and isinstance(b, (int, float)):
#                 return abs(a - b) <= (atol + rtol * abs(b))

#             print("输出类型不匹配：", type(a), type(b))
#             data_type_info = f"The value type of some values in the return value is incorrect. The current value type is {type(b)} and the correct value type is f{type(a)}"
#             return False
        
#         if C_evolved_outputs.shape != ref_outputs.shape:
#             is_correct = False
#             msg = (f"Failed (Correctness): Shape mismatch at Output. "
#                     f"Expected {ref_outputs.shape}, got {C_evolved_outputs.shape}.")
#             error_msgs.append(msg)
#             print(msg)
#             return False,msg
#         if not compare_outputs(C_evolved_outputs,ref_outputs): 
#             is_correct = False
#             if not data_type_info:
#                return False, data_type_info
#             # --- [核心修改] 捕获前 5 个错误值 ---
#             diff = torch.abs(C_evolved_outputs - ref_outputs)
#             # 计算允许的误差范围
#             tol = 1e-2 + 1e-2 * torch.abs(ref_outputs)
#             # 找出超出误差的掩码
#             error_mask = diff > tol
#             # 获取错误索引
#             error_indices = torch.nonzero(error_mask, as_tuple=False)
#             num_errors = error_indices.size(0)
            
#             msg_header = f"Failed (Correctness): Output has {num_errors} mismatches (total elements: {ref_outputs.numel()})."
#             error_details = [msg_header]
#             error_details.append("Top 5 Mismatches (Index | Reference Value | Actual Value):")
            
#             # 取前 5 个
#             for j in range(min(5, num_errors)):
#                 idx = error_indices[j]
#                 idx_tuple = tuple(idx.tolist())
#                 ref_val = ref_outputs[idx_tuple].item()
#                 act_val = C_evolved_outputs[idx_tuple].item()
#                 error_details.append(f"  [{j}] Index: {idx_tuple} | Ref: {ref_val:.6f} | Act: {act_val:.6f}")
            
#             full_msg = "\n".join(error_details)
#             error_msgs.append(full_msg)
            
#             print(f"--- KERNEL IS INCORRECT (Output) ---")
#             print(full_msg)
#             print("---------------------------")
#             # 只要发现一个输出不对，通常就可以返回了，或者收集所有错误
#             # 这里我们收集第一个主要错误后直接返回，避免 Prompt 过长
#             return False, full_msg
        
#         if is_correct:
#             return True, ""
#         else:
#             # 只有形状错误会走到这里
#             return False, "\n".join(error_msgs)

#         # # 2. 逐个检查输出张量
#         # for i, (evolved_t, ref_t) in enumerate(zip(C_evolved_outputs, gpu_ref_outputs)):
#         #     # 检查形状
#         #     if evolved_t.shape != ref_t.shape:
#         #         is_correct = False
#         #         msg = (f"Failed (Correctness): Shape mismatch at Output {i}. "
#         #                f"Expected {ref_t.shape}, got {evolved_t.shape}.")
#         #         error_msgs.append(msg)
#         #         print(msg)
#         #         continue # 继续检查下一个输出，或者直接返回也可以

#         #     # 检查数值 (atol=1e-2, rtol=1e-2)
#         #     if not torch.allclose(evolved_t, ref_t, atol=1e-2, rtol=1e-2):
#         #         is_correct = False
                
#         #         # --- [核心修改] 捕获前 5 个错误值 ---
#         #         diff = torch.abs(evolved_t - ref_t)
#         #         # 计算允许的误差范围
#         #         tol = 1e-2 + 1e-2 * torch.abs(ref_t)
#         #         # 找出超出误差的掩码
#         #         error_mask = diff > tol
#         #         # 获取错误索引
#         #         error_indices = torch.nonzero(error_mask, as_tuple=False)
#         #         num_errors = error_indices.size(0)
                
#         #         msg_header = f"Failed (Correctness): Output {i} has {num_errors} mismatches (total elements: {ref_t.numel()})."
#         #         error_details = [msg_header]
#         #         error_details.append("Top 5 Mismatches (Index | Reference Value | Actual Value):")
                
#         #         # 取前 5 个
#         #         for j in range(min(5, num_errors)):
#         #             idx = error_indices[j]
#         #             idx_tuple = tuple(idx.tolist())
#         #             ref_val = ref_t[idx_tuple].item()
#         #             act_val = evolved_t[idx_tuple].item()
#         #             error_details.append(f"  [{j}] Index: {idx_tuple} | Ref: {ref_val:.6f} | Act: {act_val:.6f}")
                
#         #         full_msg = "\n".join(error_details)
#         #         error_msgs.append(full_msg)
                
#         #         print(f"--- KERNEL IS INCORRECT (Output {i}) ---")
#         #         print(full_msg)
#         #         print("---------------------------")
#         #         # 只要发现一个输出不对，通常就可以返回了，或者收集所有错误
#         #         # 这里我们收集第一个主要错误后直接返回，避免 Prompt 过长
#         #         return False, full_msg

#         # if is_correct:
#         #     return True, ""
#         # else:
#         #     # 只有形状错误会走到这里
#         #     return False, "\n".join(error_msgs)

def check_correctness(inputs, ref_outputs, module):
    """
    (此函数已更新 - 内存优化版)
    检查通用内核的正确性。
    函数退出时会强制清理所有中间变量占用的显存。
    """
    print("Running evolved kernel for correctness check...")
    
    # [内存管理] 初始化所有可能产生的大对象变量为 None
    # 这样 finally 块可以安全地检查和删除它们
    C_evolved_outputs = None
    diff = None
    tol = None
    error_mask = None
    error_indices = None
    
    # 内部辅助函数 (保持不变)
    def compare_outputs(a, b, atol=1e-2, rtol=1e-2):
        nonlocal data_type_info # 使用 nonlocal 修改外部变量
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b): return False
            return all(compare_outputs(x, y, atol, rtol) for x, y in zip(a, b))
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.allclose(a, b, atol=atol, rtol=rtol)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= (atol + rtol * abs(b))
        
        # print("输出类型不匹配：", type(a), type(b))
        data_type_info = f"Type mismatch: expected {type(b)}, got {type(a)}"
        return False

    data_type_info = ""
    cloned_inputs = None
    try:
        cloned_inputs = copy.deepcopy(inputs)
        # --- 1. 执行 Kernel ---
        C_evolved_outputs = run_gemm(cloned_inputs, module)
        
        # --- 2. 检查输出数量 ---
        # 如果是单个 Tensor，统一转为 list/tuple 处理可能会方便些，
        # 但既然下面用了 shape 对比，这里假设 run_gemm 返回的和 ref_outputs 结构一致
        current_len = len(C_evolved_outputs) if isinstance(C_evolved_outputs, (list, tuple)) else 1
        ref_len = len(ref_outputs) if isinstance(ref_outputs, (list, tuple)) else 1

        if current_len != ref_len:
            msg = (f"Failed (Correctness): Output count mismatch. "
                   f"Expected {ref_len}, got {current_len}.")
            print(f"--- KERNEL IS INCORRECT ---")
            print(msg)
            print("---------------------------")
            return False, msg

        # --- 3. 检查 Shape ---
        # 注意：这里假设两者都是 Tensor 直接比较 shape
        # 如果是 list/tuple，这里可能需要调整逻辑，但照搬你原代码的逻辑：
        if hasattr(C_evolved_outputs, 'shape') and hasattr(ref_outputs, 'shape'):
            if C_evolved_outputs.shape != ref_outputs.shape:
                msg = (f"Failed (Correctness): Shape mismatch at Output. "
                       f"Expected {ref_outputs.shape}, got {C_evolved_outputs.shape}.")
                print(msg)
                return False, msg

        # --- 4. 检查数值 ---
        if not compare_outputs(C_evolved_outputs, ref_outputs): 
            # 类型不匹配
            if data_type_info:
               return False, data_type_info
            
            # --- [核心修改] 错误分析 (产生大量临时 Tensor) ---
            try:
                # 计算差值
                diff = torch.abs(C_evolved_outputs - ref_outputs)
                tol = 1e-2 + 1e-2 * torch.abs(ref_outputs)
                error_mask = diff > tol
                
                # 获取错误索引 (GPU -> CPU 转换可能在这里隐式发生，产生同步)
                error_indices = torch.nonzero(error_mask, as_tuple=False)
                num_errors = error_indices.size(0)
                
                msg_header = f"Failed (Correctness): Output has {num_errors} mismatches (total elements: {ref_outputs.numel()})."
                error_details = [msg_header, "Top 5 Mismatches (Index | Reference Value | Actual Value):"]
                
                # 取前 5 个 (只提取数值，不保留 Tensor 引用)
                for j in range(min(5, num_errors)):
                    idx = error_indices[j]
                    idx_tuple = tuple(idx.tolist())
                    
                    # 使用 .item() 将 GPU 标量转为 Python float，断开计算图引用
                    ref_val = ref_outputs[idx_tuple].item()
                    act_val = C_evolved_outputs[idx_tuple].item()
                    
                    error_details.append(f"  [{j}] Index: {idx_tuple} | Ref: {ref_val:.6f} | Act: {act_val:.6f}")
                
                full_msg = "\n".join(error_details)
                print(f"--- KERNEL IS INCORRECT (Output) ---")
                print(full_msg)
                print("---------------------------")
                
                return False, full_msg
            
            finally:
                # [内部清理] 这里的临时变量用完即弃
                # 虽然外层 finally 也会清理，但如果 error calculation 耗尽了显存，
                # 尽早释放有助于防止后续步骤 OOM
                if diff is not None: del diff
                if tol is not None: del tol
                if error_mask is not None: del error_mask
                if error_indices is not None: del error_indices
                diff, tol, error_mask, error_indices = None, None, None, None

        return True, ""

    except Exception as e:
        err_str = f"Runtime Error during check_correctness: {e}\n{traceback.format_exc()}"
        print(f"--- KERNEL RUNTIME FAILED ---")
        print(err_str)
        print("-----------------------------")
        return False, err_str

    finally:
        # ==========================================================
        # [关键修改] 函数退出前的终极清理
        # ==========================================================
        
        # 1. 删除主要的计算结果 Tensor
        if C_evolved_outputs is not None:
            del C_evolved_outputs

        if cloned_inputs is not None:
            # 1. 解除变量引用，使 Tensor 对象的引用计数减 1
            del cloned_inputs

        # 2. 删除错误分析阶段可能遗留的 Tensor (如果内部 try 没跑完)
        if diff is not None: del diff
        if tol is not None: del tol
        if error_mask is not None: del error_mask
        if error_indices is not None: del error_indices
        
        # 3. 强制垃圾回收 Python 对象
        gc.collect()
        
        # 4. 强制清空 CUDA 缓存，将显存归还给操作系统 (给 NCU 腾地)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # print("Memory released in check_correctness.")

#     except Exception as e:
#         err_str = f"Runtime Error during check_correctness: {e}\n{traceback.format_exc()}"
#         print(f"--- KERNEL RUNTIME FAILED ---")
#         print(err_str)
#         print("-----------------------------")
#         return False, err_str
        
# vvv --- PTXAS 解析器 (保持不变) --- vvv
# def parse_ptxas_info(log_str: str) -> Dict[str, float]: #针对TODO3做的修改，详细的修改内容见👇
#     """
#     解析 PTXAS 日志，返回扁平化的指标字典。
#     键名会自动添加数据类型前缀，例如 'float_registers_used', 'double_spill_bytes' 等。
#     """
#     metrics = {}
    
#     try:
#         # 1. 按 "Compiling entry function" 将日志切分为不同的内核块
#         # 这样可以防止不同内核的指标混淆
#         blocks = log_str.split("Compiling entry function")
        
#         for block in blocks:
#             if not block.strip():
#                 continue
                
#             # 2. 识别内核类型 (通过 C++ Name Mangling)
#             # _Z...If... -> float
#             # _Z...Id... -> double
#             # _Z...Ih... -> half (fp16)
#             # _Z...Ib... -> bfloat16 (bf16)
#             kernel_type = "unknown"
            
#             # 提取函数名，例如 '_Z14sigmoid_kernelIfEvPKT_PS0_l'
#             # 这里的正则匹配单引号内的修饰名
#             name_match = re.search(r"\'(_Z\w+)\'", block)
#             if name_match:
#                 mangled_name = name_match.group(1)
#                 if "If" in mangled_name:
#                     kernel_type = "float"
#                 elif "Id" in mangled_name:
#                     kernel_type = "double"
#                 elif "Ih" in mangled_name:
#                     kernel_type = "half"
#                 elif "Ib" in mangled_name:
#                     kernel_type = "bfloat16"
#                 else:
#                     # 如果无法识别具体类型，就使用 "kernel" 或者保留一部分特征
#                     kernel_type = "kernel" 
#             else:
#                 # 如果找不到函数名，可能是全局共有代码或其他部分，跳过
#                 continue

#             # 3. 解析该块内的具体指标，并构建带前缀的键名
            
#             # --- 寄存器 (Registers) ---
#             reg_match = re.search(r'Used\s+(\d+)\s+registers', block)
#             if reg_match:
#                 metrics[f'{kernel_type}_registers_used'] = float(reg_match.group(1))

#             # --- 共享内存 (Shared Memory / smem) ---
#             smem_match = re.search(r'(\d+)\s+bytes\s+smem', block)
#             if smem_match:
#                 metrics[f'{kernel_type}_shared_mem_bytes'] = float(smem_match.group(1))
#             else:
#                 metrics[f'{kernel_type}_shared_mem_bytes'] = 0.0
            
#             # --- 常量内存 (Constant Memory / cmem) [新增] ---
#             # 可能会有多段 cmem (e.g., cmem[0], cmem[2])，我们需要求和
#             cmem_matches = re.findall(r'(\d+)\s+bytes\s+cmem', block)
#             if cmem_matches:
#                 metrics[f'{kernel_type}_constant_mem_bytes'] = sum(float(x) for x in cmem_matches)
#             else:
#                 metrics[f'{kernel_type}_constant_mem_bytes'] = 0.0

#             # --- 溢出 (Spill Stores/Loads) ---
#             spill_stores = re.search(r'(\d+)\s+bytes\s+spill\s+stores', block)
#             spill_loads = re.search(r'(\d+)\s+bytes\s+spill\s+loads', block)
            
#             spill_total = 0.0
#             if spill_stores: spill_total += float(spill_stores.group(1))
#             if spill_loads:  spill_total += float(spill_loads.group(1))
#             metrics[f'{kernel_type}_spill_bytes'] = spill_total

#     except Exception as e:
#         print(f"警告：解析 PTXAS 日志失败: {e}", file=sys.stderr)
    
#     print(f"--- [ PTXAS Metrics Parsed ] ---")
#     print(json.dumps(metrics, indent=2))
    
#     return metrics

def parse_ptxas_info(log_str: str) -> Dict[str, Any]:
    """
    [升级版] 高级解析 PTXAS 日志，生成结构化表格。
    修复：支持从函数名中解析向量化宽度 (如 sigmoid_vec4 -> width=4)
    修复：支持从函数参数中推断数据类型 (如 PKf -> float, PK6__half -> Half)
    """
    metrics = {}
    
    # 辅助函数：增强版 Demangler
    def _demangle_info(mangled: str):
        # 1. 提取函数名 (Itanium ABI: _Z + len + name)
        # 例如: _Z19sigmoid_kernel_vec4... -> len=19, name=sigmoid_kernel_vec4
        name_match = re.match(r'_Z(\d+)(\w+)', mangled)
        func_name = "unknown"
        suffix_part = "" # 包含模板参数 或 函数参数签名
        
        if name_match:
            length = int(name_match.group(1))
            full_string = name_match.group(2)
            func_name = full_string[:length]
            suffix_part = full_string[length:] # 剩下的部分
        else:
            func_name = mangled

        # 2. 解析向量化宽度 (Width)
        width = "Scalar"
        
        # 策略 A: 查找模板参数中的 Li (Literal Int)，例如 <float, 4> -> Li4
        vec_match_template = re.search(r'Li(\d+)', suffix_part)
        
        # 策略 B: [新增] 查找函数名中的 vecX，例如 sigmoid_vec4
        vec_match_name = re.search(r'vec(\d+)', func_name, re.IGNORECASE)
        
        if vec_match_template:
            width = vec_match_template.group(1)
        elif vec_match_name:
            width = vec_match_name.group(1)
        elif "vec" in func_name.lower():
            width = "?" # 即使是向量化函数，也没找到具体数字
            
        # 3. 解析数据类型 (Data Type)
        dtype = "Unknown"
        name_lower = func_name.lower()
        
        # 策略 A: [新增] 优先检查函数名中的显式标记 (如 fp16_vec4)
        if "fp16" in name_lower or "half" in name_lower:
            dtype = "Half(FP16)"
        elif "bf16" in name_lower or "bfloat16" in name_lower:
            dtype = "BFloat16"
        elif "fp64" in name_lower or "double" in name_lower:
            dtype = "double(FP64)"
        elif "fp32" in name_lower or "float" in name_lower:
            dtype = "float(FP32)"
        
        # 策略 B: 检查 Mangled Suffix (模板参数 或 函数参数类型)
        if dtype == "Unknown":
            # Half 检测: PyTorch ATen Half (N3c104HalfE) 或 CUDA __half (6__half)
            if 'Half' in suffix_part or '__half' in suffix_part:
                dtype = "Half(FP16)"
            elif 'BFloat16' in suffix_part or '__nv_bfloat16' in suffix_part:
                dtype = "BFloat16"
            
            # Double 检测: 模板 Id, 指针 Pd/PKd
            elif 'Id' in suffix_part or 'Pd' in suffix_part or 'PKd' in suffix_part:
                 dtype = "double(FP64)"
            
            # Float 检测: 模板 If, 指针 Pf/PKf
            elif 'If' in suffix_part or 'Pf' in suffix_part or 'PKf' in suffix_part:
                 dtype = "float(FP32)"
            
            # 兜底: 简单字符匹配 (慎用，防止匹配到函数名的一部分)
            elif 'd' in suffix_part and 'f' not in suffix_part: # 只有d没有f
                 dtype = "double(FP64)"
            elif 'f' in suffix_part and 'd' not in suffix_part: # 只有f没有d
                 dtype = "float(FP32)"

        # 构建可读函数名 (Pretty Name)
        clean_func_name = func_name
        # 如果类型已知，生成类似 sigmoid<float, 4> 的名字
        type_str = dtype.split('(')[0]
        if width != "Scalar" and width != "?":
            pretty_name = f"{clean_func_name}<{type_str}, {width}>"
        else:
            pretty_name = f"{clean_func_name}<{type_str}>"
            
        return func_name, pretty_name, dtype, width

    try:
        # 1. 按 Entry Function 分块
        blocks = log_str.split("Compiling entry function")
        
        # 打印表头
        print(f"\n{'='*100}")
        print(f"{'内核函数 (Mangled Name)':<45} | {'数据类型':<12} | {'宽度':<5} | {'寄存器':<6} | {'备注 (Local/Const/Shared)'}")
        print(f"{'-'*100}")

        for block in blocks:
            if not block.strip(): continue
            
            # 提取 Mangled Name
            name_match = re.search(r"\'(_Z\w+)\'", block)
            if not name_match: continue
            
            mangled_name = name_match.group(1)
            func_base, pretty_name, dtype, width = _demangle_info(mangled_name)
            
            # 提取指标
            regs = 0
            reg_match = re.search(r'Used\s+(\d+)\s+registers', block)
            if reg_match: regs = int(reg_match.group(1))
            
            # 内存指标
            smem = 0
            smem_match = re.search(r'(\d+)\s+bytes\s+smem', block)
            if smem_match: smem = int(smem_match.group(1))
            
            cmem_matches = re.findall(r'(\d+)\s+bytes\s+cmem', block)
            cmem_str = "+".join(cmem_matches) if cmem_matches else "0"
            cmem_total = sum(int(x) for x in cmem_matches)
            
            spill_store = 0
            spill_load = 0
            spill_s = re.search(r'(\d+)\s+bytes\s+spill\s+stores', block)
            spill_l = re.search(r'(\d+)\s+bytes\s+spill\s+loads', block)
            if spill_s: spill_store = int(spill_s.group(1))
            if spill_l: spill_load = int(spill_l.group(1))
            spill_total = spill_store + spill_load
            
            # 构建备注
            remarks = []
            if cmem_total > 0: remarks.append(f"Cmem: {cmem_str}")
            if smem > 0: remarks.append(f"Smem: {smem}")
            if spill_total > 0: remarks.append(f"SPILL: {spill_total}B")
            remark_str = ", ".join(remarks)
            
            # 打印表格行
            display_mangled = (mangled_name[:42] + '..') if len(mangled_name) > 44 else mangled_name
            print(f"{display_mangled:<45} | {dtype:<12} | {width:<5} | {regs:<6} | {remark_str}")

            # 存入 metrics 字典
            metrics[pretty_name] = {
                "registers": regs,
                "spill_bytes": spill_total,
                "cmem_bytes": cmem_total,
                "smem_bytes": smem,
                "type": dtype,
                "width": width
            }
            
        print(f"{'='*100}\n")

    except Exception as e:
        print(f"警告：解析 PTXAS 日志失败: {e}", file=sys.stderr)
    
    return metrics


def get_kernel_name(cuda_code_string):
    """
    从 CUDA 源代码字符串中解析出第一个 __global__ void 内核函数的名称。
    
    参数:
        cuda_code_string (str): 包含 CUDA 源代码的字符串。
    
    返回:
        str: 找到的内核函数名称。如果未找到，返回 None。
    """
    # 正则表达式解释：
    # __global__  : 匹配 __global__ 关键字
    # \s+         : 匹配一个或多个空格
    # void        : 匹配 void 关键字
    # \s+         : 匹配一个或多个空格
    # ([a-zA-Z0-9_]+) : 捕获组，匹配内核名称（字母、数字、下划线）
    # \s*\(       : 匹配零个或多个空格后跟左括号 (，标志着函数参数的开始
    pattern = r"__global__\s+void\s+([a-zA-Z0-9_]+)\s*\("
    
    match = re.search(pattern, cuda_code_string)
    
    if match:
        return match.group(1)
    else:
        return None

# vvv --- [!!! 已更新 !!!] 真实 NCU 分析器 (现在是通用的) --- vvv
def get_real_ncu_metrics(module_path, module_name, inputs, init_inputs=None, cuda_code=None) -> Dict[str, float]:
    """
    动态创建一个目标脚本，运行 ncu，解析 CSV 输出，并返回指标。
    [!!! 已更新 !!!] 接受通用输入和内核/wrapper 名称。
    """
    kernel_name = get_kernel_name(cuda_code)
    ncu_metrics = {}
    target_script_path = f"_ncu_target_{module_name}.py"
    temp_csv_path = f"_ncu_output_{module_name}.csv"
    
    try:
        # 1. 写入 ncu 目标脚本
        with open(target_script_path, "w", encoding="utf-8") as f:
            f.write(NCU_TARGET_SCRIPT_TEMPLATE)

        # [!!! 已更新 !!!] 保存输入以供 ncu 脚本加载
        torch.save(inputs, '_ncu_inputs.pt')
        if init_inputs is not None:
            torch.save(init_inputs, '_ncu_init_inputs.pt')

        # 2. 构建 ncu 命令 (不带 --metrics 以获取全集)
        ncu_command = [
            'ncu',
            '--csv',
            '--profile-from-start', 'off',
            # '--kernel-name', kernel_name, # <--- [!!! 已删除 !!!]
            # '--launch-count', '1',
            '--kernel-name',f'{kernel_name}',
            '--clock-control', 'none', # 避免 ncu 锁定频率
            '--target-processes', 'all',
            'python', 
            target_script_path,
            module_path, 
            module_name
            # wrapper_function_name # <--- [!!! 已更新 !!!]
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

        try:
            with open(temp_csv_path, "w", encoding="utf-8") as debug_f:
                debug_f.write(proc.stdout)
            print(f"--- [DEBUG] NCU CSV 内容已保存至: {temp_csv_path} ---")
        except Exception as e:
            print(f"警告：保存调试 CSV 文件失败: {e}", file=sys.stderr)

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
        if os.path.exists("_ncu_init_inputs.pt"):
            os.remove("_ncu_init_inputs.pt")
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            
    print(f"--- [ NCU 指标已解析 (共 {len(ncu_metrics)} 个) ] ---")
    if ncu_metrics:
        sample_keys = random.sample(list(ncu_metrics.keys()), min(5, len(ncu_metrics)))
        sample_metrics = {k: ncu_metrics[k] for k in sample_keys}
        print(json.dumps(sample_metrics, indent=2))
        
    return ncu_metrics
# ^^^ --- NCU 函数结束 --- ^^^


# vvv --- [!!! 已更新 !!!] 真实性能评测函数 (现在是通用的) --- vvv
def benchmark_kernel(inputs, module, warmup_runs=5, benchmark_runs=10):
    """
    对当前加载的 _gemm_module 执行预热和基准测试。
    [!!! 已更新 !!!] 接受通用输入。
    """
    # if _gemm_module is None:
    #     raise RuntimeError("模块未编译。")
    
    # gpu_inputs = [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in inputs]
    cloned_inputs = None
    try:
        cloned_inputs = copy.deepcopy(inputs)
        print(f"Warming up evolved kernel ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            _ = run_gemm(cloned_inputs, module)
        torch.cuda.synchronize()

        # 测量
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(benchmark_runs):
            _ = run_gemm(cloned_inputs, module)
        end.record()

        torch.cuda.synchronize()
        avg_time_ms = start.elapsed_time(end) / benchmark_runs
        print(f"Evolved kernel benchmark: {avg_time_ms:.3f} ms")
        return avg_time_ms
    finally:
        # ==========================================================
        # [关键修改] 自动清理逻辑
        # ==========================================================
        # 无论 try 块中发生了什么（正常 return output，或者抛出 Exception），
        # finally 块中的代码永远会在函数退出前最后执行。
        
        if cloned_inputs is not None:
            # 1. 解除变量引用，使 Tensor 对象的引用计数减 1
            del cloned_inputs
            
            # 2. (可选) 如果显存极度紧张，可以手动触发 Python GC
            # 这能确保 PyTorch 的 C++ 后端更快收到“显存可释放”的信号
            # gc.collect()
    
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