import os
import json
import torch
import torch.nn.functional as F
import re

# -------------------------------------------------------------------------
# 1. 算子映射与基线实现
#    (根据 TransEval/cuda.json 的常见算子名提供 PyTorch 等价实现)
# -------------------------------------------------------------------------

def get_torch_baseline(op_name, args, device="cuda"):
    """
    返回: (func, inputs)
    func: 一个可调用的 PyTorch 函数
    inputs: 一个 list，包含生成好的 GPU Tensor
    """
    dtype = torch.float32 # 默认 float32
    
    # --- A. 构造输入数据 ---
    inputs = []
    
    # 通用形状解析 (假设 args 是 shape 列表或 [M, N, K] 等)
    # 大多数 Xpiler 用例 args 都是维度列表
    dims = [int(x) for x in args if str(x).isdigit()]
    
    if op_name in ["add", "sub", "mul", "div", "max", "min"]:
        # 双目逐元素操作 (Element-wise binary)
        # 通常 args 是 shape
        shape = dims
        if not shape: shape = [1024] # Fallback
        A = torch.randn(*shape, device=device, dtype=dtype)
        B = torch.randn(*shape, device=device, dtype=dtype)
        inputs = [A, B]
        
        if op_name == "add": func = lambda a, b: a + b
        elif op_name == "sub": func = lambda a, b: a - b
        elif op_name == "mul": func = lambda a, b: a * b
        elif op_name == "div": func = lambda a, b: a / b
        elif op_name == "max": func = torch.maximum
        elif op_name == "min": func = torch.minimum

    elif op_name in ["relu", "sigmoid", "tanh", "gelu", "silu", "abs", "exp", "log", "sqrt", "sin", "cos"]:
        # 单目逐元素操作 (Element-wise unary)
        shape = dims
        if not shape: shape = [1024]
        A = torch.randn(*shape, device=device, dtype=dtype)
        inputs = [A]
        
        if op_name == "relu": func = F.relu
        elif op_name == "sigmoid": func = torch.sigmoid
        elif op_name == "tanh": func = torch.tanh
        elif op_name == "gelu": func = F.gelu
        elif op_name == "silu": func = F.silu
        elif op_name == "sin": func = torch.sin
        elif op_name == "cos": func = torch.cos
        elif op_name == "exp": func = torch.exp
        
    elif op_name == "matmul" or op_name == "gemm":
        # 矩阵乘法 [M, K, N] 或 [M, N, K]
        # Xpiler 通常是 M, N, K (或者 M, K, N，需尝试)
        if len(dims) >= 3:
            M, N, K = dims[0], dims[1], dims[2]
        else:
            M, N, K = 128, 128, 128
            
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
        inputs = [A, B]
        func = torch.matmul

    elif op_name == "softmax":
        shape = dims
        if not shape: shape = [128, 128]
        A = torch.randn(*shape, device=device, dtype=dtype)
        # 最后一个维度做 softmax
        dim = -1
        inputs = [A]
        func = lambda x: F.softmax(x, dim=dim)
        
    elif op_name == "layernorm":
        # args 通常是 [N, H, W] 之类
        shape = dims
        if not shape: shape = [128, 128]
        A = torch.randn(*shape, device=device, dtype=dtype)
        normalized_shape = shape[-1:]
        inputs = [A]
        # LayerNorm 需要 weight 和 bias，这里简化为默认
        layer_norm = torch.nn.LayerNorm(normalized_shape, elementwise_affine=False).to(device)
        func = lambda x: layer_norm(x)

    elif op_name == "rmsnorm":
        # PyTorch 没有直接的 RMSNorm，手动实现一个简单的
        shape = dims
        if not shape: shape = [128, 128]
        A = torch.randn(*shape, device=device, dtype=dtype)
        inputs = [A]
        def rms_norm(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        func = rms_norm

    else:
        # 未知算子，尝试通用的 Unary
        print(f"[XpilerLoader] Warning: Unknown op '{op_name}', using generic Unary inputs.")
        shape = dims if dims else [1024]
        A = torch.randn(*shape, device=device, dtype=dtype)
        inputs = [A]
        # 默认 Identity，仅为了跑通流程
        func = lambda x: x 

    return func, inputs

# -------------------------------------------------------------------------
# 2. 文件加载器
# -------------------------------------------------------------------------

class XpilerBenchmarkLoader:
    def __init__(self, xpiler_root):
        self.xpiler_root = xpiler_root
        self.cuda_dir = os.path.join(xpiler_root, "XpilerBench", "CUDA")
        self.json_path = os.path.join(xpiler_root, "TransEval", "cuda.json")
        
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Missing config: {self.json_path}")
        if not os.path.exists(self.cuda_dir):
            raise FileNotFoundError(f"Missing CUDA dir: {self.cuda_dir}")

        with open(self.json_path, 'r') as f:
            self.configs = json.load(f)
            
    def get_problems(self, limit=0):
        """
        生成器，返回 (problem_name, cuda_file_path, op_name, args)
        """
        count = 0
        for cfg in self.configs:
            op_name = cfg['op_name']
            args = cfg['args'] # list of ints
            
            # 构造期望的文件名
            # 格式通常是: {op_name}_{arg1}_{arg2}... .cu
            # 有时 args 包含非 int，需要处理
            arg_str = "_".join([str(x) for x in args])
            filename = f"{op_name}_{arg_str}.cu"
            filepath = os.path.join(self.cuda_dir, filename)
            
            # 如果找不到，尝试模糊匹配（有些文件命名不规范）
            if not os.path.exists(filepath):
                # 简单尝试：只匹配 op_name
                # (这里为了稳健性，如果找不到精确匹配，暂时跳过)
                continue
                
            problem_name = f"xpiler_{op_name}_{arg_str}"
            
            yield {
                "name": problem_name,
                "path": filepath,
                "op": op_name,
                "args": args,
                "code": open(filepath, 'r').read()
            }
            
            count += 1
            if limit > 0 and count >= limit:
                break