import os
import sys
import json
import time
import argparse
import torch
import traceback
import re
import signal
import gc

# [!!! 路径配置 - 请根据你的环境修改 !!!]
# 指向你的 KernelBench_main 目录
KERNELBENCH_MAIN_PATH = "/home/lxt/KernelBench/KernelBench_main"
# 指向你的 QiMeng-xpiler-eval 目录
XPILER_EVAL_PATH = "/home/lxt/QiMeng-xpiler-eval/QiMeng-xpiler-eval"

# 添加路径以导入 mini_version 模块
MINI_VERSION_PATH = os.path.join(KERNELBENCH_MAIN_PATH, "mini_version")
if MINI_VERSION_PATH not in sys.path:
    sys.path.append(MINI_VERSION_PATH)

# 导入 mini_version 模块
try:
    import config as mv_config
    import llm_api as mv_llm_api
    import prompts as mv_prompts
    import main as mv_main
    import cuda_utils as mv_cuda_utils
except ImportError as e:
    print(f"Error: 无法从 {MINI_VERSION_PATH} 导入 mini_version 模块。")
    print(e)
    sys.exit(1)

# 导入本地的 Xpiler 加载器
import xpiler_loader

# --- 辅助函数 ---

def extract_all_code_blocks(text):
    """
    提取文本中所有的代码块内容。
    返回一个列表，包含所有 ```lang ... ``` 中的内容。
    """
    # 匹配 ```任意语言 ... ```
    pattern = r'```(?:\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]

def generate_wrapper_code(cuda_source, inputs, ref_outputs, kernel_name, wrapper_name):
    """
    调用 LLM 为现有的 Kernel 生成 PyTorch Wrapper
    """
    # 1. 获取 Prompt
    prompt = mv_prompts.get_wrapper_generation_prompt(
        cuda_source, inputs, ref_outputs, kernel_name, wrapper_name
    )
    
    system_prompt = "你是一位专业的 CUDA/PyTorch 绑定专家。"
    
    # 2. 调用 LLM
    try:
        response_text = mv_llm_api.call_llm(
            agent_name="initial_generator", 
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        # 3. 提取代码 (修正版逻辑)
        # 因为 Prompt 模板让两个块都叫 ```cpp，所以我们需要按顺序提取
        blocks = extract_all_code_blocks(response_text)
        
        cpp_sig = None
        wrapper_impl = None

        if len(blocks) >= 2:
            # 假设第一个是签名，第二个是实现
            cpp_sig = blocks[0]
            wrapper_impl = blocks[1]
        elif len(blocks) == 1:
            # 只有一块，可能混在一起了，尝试当做实现，签名可能缺失或在其中
            print("Warning: Only 1 code block found in wrapper generation.")
            wrapper_impl = blocks[0]
            # 尝试从实现中正则提取签名（备用方案）
            if "torch::Tensor" in wrapper_impl and ";" in wrapper_impl.split("{")[0]:
                 cpp_sig = wrapper_impl.split("{")[0].strip() + ";"
        
        return cpp_sig, wrapper_impl, response_text
    
    except Exception as e:
        print(f"Wrapper Generation Error: {e}")
        return None, None, str(e)

def extract_kernel_body(full_cuda_source):
    """
    从 Xpiler 的 .cu 文件中提取 __global__ 函数部分，
    去掉 extern "C" 的 host 代码，以免与我们生成的 Wrapper 冲突。
    """
    lines = full_cuda_source.split('\n')
    cleaned_lines = []
    skip = False
    for line in lines:
        if 'extern "C"' in line:
            skip = True
        if not skip:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# --- 主逻辑 ---

def main(args):
    print(f"🚀 Starting XpilerBench Optimization Loop")
    print(f"📂 Xpiler Path: {XPILER_EVAL_PATH}")
    # print(f"🤖 LLM Config: {json.dumps(mv_config.AGENT_MODELS, indent=2)}")
    
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "xpiler_summary.json")
    summary_results = {}

    # 1. 初始化加载器
    loader = xpiler_loader.XpilerBenchmarkLoader(XPILER_EVAL_PATH)
    
    # 2. 遍历算子
    problems = loader.get_problems(limit=args.limit_files)
    
    for prob in problems:
        name = prob['name']
        op_name = prob['op']
        args_dims = prob['args']
        raw_cuda_code = prob['code'] # Xpiler 原始代码 (Ground Truth)
        
        print(f"\n\n=== Processing: {name} ({op_name}) ===")
        
        problem_dir = os.path.join(results_dir, name)
        os.makedirs(problem_dir, exist_ok=True)
        history_file = os.path.join(problem_dir, "history.json")
        
        inputs = None
        ref_outputs = None
        
        try:
            # --- 步骤 1: 建立 PyTorch 基线 & 输入 ---
            print("Step 1: Generating PyTorch Baseline...")
            torch_func, inputs = xpiler_loader.get_torch_baseline(op_name, args_dims, device="cuda")
            
            # 运行基线以获得 ref_outputs
            torch.cuda.synchronize()
            # 预热
            for _ in range(5): torch_func(*inputs)
            # 测速
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(20):
                ref_out = torch_func(*inputs)
            end_ev.record()
            torch.cuda.synchronize()
            baseline_ms = start_ev.elapsed_time(end_ev) / 20.0
            
            print(f"PyTorch Baseline: {baseline_ms:.4f} ms")
            
            if isinstance(ref_out, torch.Tensor):
                ref_outputs = [ref_out]
            elif isinstance(ref_out, (list, tuple)):
                ref_outputs = list(ref_out)
            else:
                ref_outputs = [ref_out]

            # --- 步骤 2: 清洗原始代码 & 生成 Wrapper ---
            print("Step 2: Generating PyTorch Wrapper for Ground Truth Kernel...")
            
            clean_kernel_code = extract_kernel_body(raw_cuda_code)
            
            # 尝试从代码中提取 kernel 名字
            kernel_name_match = re.search(r'__global__\s+void\s+(\w+)', clean_kernel_code)
            if not kernel_name_match:
                # 有些特殊的写法可能包含 launch_bounds 等宏，尝试更宽泛的匹配
                kernel_name_match = re.search(r'__global__\s+void\s+.*?\s+(\w+)\s*\(', clean_kernel_code, re.DOTALL)
            
            kernel_name = kernel_name_match.group(1) if kernel_name_match else "unknown_kernel"
            wrapper_name = f"{name}_wrapper".replace("-", "_") # 确保 wrapper 名字合法
            
            cpp_sig, wrapper_impl, _ = generate_wrapper_code(
                clean_kernel_code, inputs, ref_outputs, kernel_name, wrapper_name
            )
            
            if not cpp_sig or not wrapper_impl:
                print("Failed to generate wrapper.")
                summary_results[name] = {"status": "Wrapper Generation Failed"}
                continue
                
            # 组合成初始的可编译代码
            initial_cuda_code = clean_kernel_code + "\n\n" + wrapper_impl
            
            # 验证一下初始代码是否能跑
            print("Verifying Initial Code Correctness...")
            try:
                # 临时加载
                # 先清除可能的旧模块缓存
                mv_cuda_utils._gemm_module = None 
                
                mv_cuda_utils.load_gemm_module(
                    cpp_sig, initial_cuda_code, f"{name}_sanity_check", wrapper_name
                )
                is_valid = mv_cuda_utils.check_correctness(inputs, ref_outputs, wrapper_name)
                if not is_valid:
                    print("Warning: Initial wrapper compiled but correctness check failed.")
                    summary_results[name] = {"status": "Initial Correctness Failed"}
                    # 即使失败也可能进入优化，或者选择跳过。这里选择跳过以保证质量。
                    continue
                print("Initial Code Verified ✅")
            except Exception as e:
                print(f"Initial Compilation Failed: {e}")
                summary_results[name] = {"status": f"Initial Compilation Failed: {e}"}
                continue

            # --- 步骤 3: 进入优化循环 ---
            print("Step 3: Running Optimization Loop...")
            
            best_result = mv_main.run_optimization_on_problem(
                problem_name=name,
                cpp_source=cpp_sig,           
                initial_cuda_code=initial_cuda_code,
                inputs=inputs,
                ref_outputs=ref_outputs,
                kernel_name=kernel_name,      
                wrapper_function_name=wrapper_name,
                iteration_rounds=mv_config.ITERATION_ROUNDS,
                history_file_path=history_file,
                baseline_time_ms=baseline_ms
            )
            
            best_time = best_result.get('time_ms', float('inf'))
            speedup = baseline_ms / best_time if best_time > 0 else 0
            
            print(f"🏁 Finished {name}")
            print(f"Baseline: {baseline_ms:.4f} ms | Best: {best_time:.4f} ms | Speedup: {speedup:.2f}x")
            
            summary_results[name] = {
                "baseline_ms": baseline_ms,
                "best_cuda_ms": best_time,
                "speedup": speedup,
                "status": "Success" if best_time < float('inf') else "Optimization Failed"
            }
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            traceback.print_exc()
            summary_results[name] = {"status": f"Error: {e}"}
            
        finally:
            # 清理显存
            inputs = None
            ref_outputs = None
            gc.collect()
            torch.cuda.empty_cache()
            
        # 实时保存摘要
        with open(summary_path, "w") as f:
            json.dump(summary_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./xpiler_optimization_results")
    parser.add_argument("--limit_files", type=int, default=0, help="0 for all")
    args = parser.parse_args()
    
    main(args)