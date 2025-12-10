import openai
import config
import kernels 

# [!!! 已移除 !!!]
# 全局客户端已被移除，因为不同 Agent 可能使用不同模型

def call_llm(agent_name: str, system_prompt: str, user_prompt: str):
    """
    A simple wrapper for the LLM API call.
    (已更新为使用 dmxapi 并兼容 OpenAI 格式)
    
    [!!! 已更新 !!!]
    - 接受 agent_name 作为参数
    - 根据 agent_name 从 config.py 动态选择模型
    """
    if config.MOCK_LLM_CALLS:
        # (模拟逻辑保持不变，以防您想切回测试)
        print("--- [MOCK LLM CALL] ---")
        if "Planner Agent" in system_prompt:
            return "OPTIMIZATION_GOAL: Implement 16x16 tiling using shared memory."
        if "Tool Agent" in system_prompt:
            return "METRICS: ['dram__bytes_read.sum', 'l1tex__t_bytes_read.sum', 'shared_load_transactions.sum', 'achieved_occupancy.avg', 'warp_execution_efficiency.pct']"
        if "Analysis Agent" in system_prompt:
            return "DETAILED_PLAN:\n1. ... (模拟计划) ..."
        if "Coder Agent" in system_prompt:
            return f"```cuda\n{kernels.TILED_CUDA_SOURCE}\n```"
        return "Mocked response."

    # --- 真实的 API 调用 (使用 dmxapi 客户端) ---
    
    # [!!! 已更新 !!!] 解决问题 3
    # 1. 根据 Agent 名称获取模型
    model_name = config.AGENT_MODELS.get(agent_name)
    if not model_name:
        raise ValueError(f"在 config.py 的 AGENT_MODELS 中未找到 '{agent_name}' 的模型配置")

    # 2. 动态创建客户端 (因为 client 绑定了 base_url)
    try:
        client_dmxapi = openai.OpenAI(
            api_key=config.DMX_API_KEY,
            base_url=config.DMX_API_BASE_URL
        )
        
        # 3. 使用动态获取的 model_name 进行调用
        response = client_dmxapi.chat.completions.create(
            model=model_name, # <--- 使用特定于 Agent 的模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
            # top_p=0.1,
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling DMX API for agent '{agent_name}' (model: {model_name}): {e}")
        return None