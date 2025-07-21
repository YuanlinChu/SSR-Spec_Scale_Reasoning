from transformers import AutoTokenizer
import time

def my_format_chat(messages,
                model_name: str,
                add_generation_prompt: bool = True,
                continue_final_message: bool = False) -> str:
    """
    根据 model_name 选择格式化逻辑，将 messages 列表转换为 prompt 字符串。

    对于 Qwen/QwQ-32B:
      - ChatML 样式: <|im_start|>role\ncontent<|im_end|>\n
      - 末尾若 add_generation_prompt，则添加 <|im_start|>assistant\n

    对于 DeepSeek-R1-Distill-Qwen-1.5B:
      - 深搜自定义格式:
        开始: <｜begin▁of▁sentence｜>
        用户: <｜User｜>content
        助手: <｜Assistant｜>content<｜end▁of▁sentence｜>  （仅在 assistant 后加 end）
      - 末尾若 add_generation_prompt，则追加 <｜Assistant｜><think>
      - 若 continue_final_message=True，则最后一条 assistant 不插入 end

    Args:
      messages: List[{"role": "user"/"assistant", "content": str}]
      model_name: 模型名称，用于分支选择
      add_generation_prompt: 是否在末尾加入生成提示
      continue_final_message: 是否让模型在最后一条消息基础上续写
    """
    # Qwen/QwQ-32B 分支（不变）
    if model_name.lower().startswith("qwen/qwq-32b"):
        prompt = ""
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if continue_final_message and idx == len(messages) - 1:
                prompt += f"<|im_start|>{role}\n{content}"
            else:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n<think>"
        return prompt

    # DeepSeek-R1-Distill-Qwen-1.5B 分支（修正后）
    elif model_name.lower().startswith("deepseek-ai/deepseek-r1-distill-qwen-1.5b"):
        prompt = "<｜begin▁of▁sentence｜>"
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if role == "user":
                # 只加用户标签和内容，不加 end
                prompt += f"<｜User｜>{content}"
            else:  # assistant
                prompt += f"<｜Assistant｜>{content}"
                # 仅在 assistant 消息后添加 end，且不在最后一条需续写的情况下
                if not (continue_final_message and idx == len(messages) - 1):
                    prompt += "<｜end▁of▁sentence｜>"
        # 生成提示
        if add_generation_prompt:
            prompt += "<｜Assistant｜><think>"
        return prompt

    else:
        raise ValueError(f"Unsupported model: {model_name}")

def format_chat(messages, model_name, add_generation_prompt=True, continue_final_message=False):
    """
    使用模型的聊天模板构建对话提示。

    参数：
    - messages: 列表，每个元素是一个字典，包含 'role' 和 'content' 键。
    - model_name: 模型的名称或路径。
    - add_generation_prompt: 是否添加生成提示。

    返回：
    - prompt: 构建好的对话提示字符串。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )
    return prompt

messages = [
    {"role": "user", "content": "problem"},
    {"role": "assistant", "content": "steps_so_far_str"},
    {"role": "user", "content": "Evaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9."},
    {"role": "assistant", "content": "I think the quality score is: "},
]
target_model_name = "Qwen/QwQ-32B"
draft_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# start_time = time.time()
prompt1 = format_chat(messages, "Qwen/Qwen3-32B", add_generation_prompt=True, continue_final_message=False)
# prompt2 = format_chat(messages, draft_model_name, add_generation_prompt=False, continue_final_message=True)
print(prompt1)

# time_cost = time.time() - start_time
# print(f"Time cost: {time_cost:.4f}s")  # 输出时间花费，保留四位小数

# start_time = time.time()
prompt1 = my_format_chat(messages, target_model_name, add_generation_prompt=True, continue_final_message=False)
# prompt2 = my_format_chat(messages, draft_model_name, add_generation_prompt=False, continue_final_message=True)
print(prompt1)

# time_cost = time.time() - start_time
# print(f"Time cost: {time_cost:.4f}s")  # 输出时间花费，保留四位小数
