import torch
import os

# 为 PyTorch >= 2.1 设置环境变量 (推荐)
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# --- General Settings ---
# MATRIX_N = 8192 # <--- [!!! 已移除 !!!] 不再需要全局 N
ITERATION_ROUNDS = 10  # 可外部设置的迭代轮数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- [!!! 已更新 !!!] ---
# 解决问题 3: 为不同 Agent 配置不同模型
# AGENT_MODELS = {
#     "initial_generator": "DeepSeek-V3.1-Fast", # 建议使用最强的模型
#     "initial_corrector": "DeepSeek-V3.1-Fast", # [!!! 新增 !!!] (建议使用您最强的模型)

#     "planner": "deepseek-v3.1",  # 规划 Agent，用一个较快、便宜的模型
#     "tool": "deepseek-v3.1",       # 工具 Agent，用一个较快、便宜的模型
#     "analysis": "deepseek-v3.1",    # 分析 Agent，用一个更强的模型
#     "coder": "qwen3-max-2025-10-30",       # 编码 Agent，用一个最强的模型
#     # "coder":"gpt-5-codex-high"
# }
AGENT_MODELS = {
    "initial_generator": "gpt-5", # 建议使用最强的模型
    "initial_corrector": "DeepSeek-V3.1-Fast", # [!!! 新增 !!!] (建议使用您最强的模型)

    "planner": "grok-4-fast",  # 规划 Agent，用一个较快、便宜的模型
    "tool": "grok-4-fast",       # 工具 Agent，用一个较快、便宜的模型
    "analysis": "grok-4-fast",    # 分析 Agent，用一个更强的模型
    "coder": "gpt-5",       # 编码 Agent，用一个最强的模型
    # "coder":"gpt-5-codex-high"
}

# AGENT_MODELS = {
#     "planner": "deepseek-v3.2-exp",  # 规划 Agent，用一个较快、便宜的模型
#     "tool": "deepseek-v3.2-exp",       # 工具 Agent，用一个较快、便宜的模型
#     "analysis": "deepseek-v3.2-exp",    # 分析 Agent，用一个更强的模型
#     "coder": "deepseek-v3.2-exp",       # 编码 Agent，用一个最强的模型
# }
# --- [!!! 已更新 !!!] ---

# --- LLM Settings (dmxapi) ---
DMX_API_KEY = "sk-EurbTIrCe7fqGs8LxhxS77d5wKb0LlmUlPHWUnYa59PTVY9P" # (使用您提供的key)
DMX_API_BASE_URL = "https://www.dmxapi.cn/v1"

# --- 模拟开关 ---
# !! 根据您的要求，设置为 False 以启用真实的 LLM API 调用 !!
MOCK_LLM_CALLS = False

# --- [!!! 新增 !!!] ---
# 解决问题 4: 历史记忆
# 将优化历史保存到 JSON 文件中
HISTORY_FILE = "optimization_history.json"
# --- [!!! 新增 !!!] ---

# --- Profiler Settings ---
# (此列表现在仅作为后备，或用于 Tool Agent 的提示词示例)
BASE_NCU_METRICS_LIST_EXAMPLE = [
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes_read.sum",
    "lts__t_bytes_write.sum", "l1tex__t_bytes_read.sum", "l1tex__t_bytes_write.sum",
    "sm__cycles_elapsed.avg", "smsp__warps_elapsed.avg", "sm__inst_executed.avg",
    "smsp__thread_inst_executed.avg", "achieved_occupancy.avg", 
    "branch_efficiency.pct", "warp_execution_efficiency.pct"
]