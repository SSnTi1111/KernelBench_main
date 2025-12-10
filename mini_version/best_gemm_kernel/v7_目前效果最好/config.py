import torch
import os

# 为 PyTorch >= 2.1 设置环境变量 (推荐)
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# --- General Settings ---
MATRIX_N = 2048
ITERATION_ROUNDS = 30  # 可外部设置的迭代轮数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- [!!! 已更新 !!!] ---
# 解决问题 3: 为不同 Agent 配置不同模型
AGENT_MODELS = {
    "planner": "DeepSeek-V3.1-Fast",  # 规划 Agent，用一个较快、便宜的模型
    "tool": "DeepSeek-V3.1-Fast",       # 工具 Agent，用一个较快、便宜的模型
    "analysis": "DeepSeek-V3.1-Fast",    # 分析 Agent，用一个更强的模型
    "coder": "gpt-5",       # 编码 Agent，用一个最强的模型
}
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