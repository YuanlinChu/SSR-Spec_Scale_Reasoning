# qwen3_vllm_test.py
import argparse
from vllm import LLM, SamplingParams

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="使用VLLM加载本地Qwen3模型进行推理")
    # parser.add_argument("--model_path", type=str, default="/hpc2hdd/home/bwang423/test/weight/Qwen3-8B", 
                        # help="模型路径")
    parser.add_argument("--max_tokens", type=int, default=512, 
                        help="生成的最大token数量")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="生成的温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="top_p采样参数")
    args = parser.parse_args()

    # 初始化VLLM的LLM对象
    # print(f"正在加载模型: {args.model_path}")
    llm = LLM(
        # model=args.model_path,
        model = "Qwen/Qwen3-0.6B",
        trust_remote_code=True,  # Qwen3模型可能需要自定义代码
        tensor_parallel_size=4,  # 根据您的GPU数量调整
        dtype="auto",           # 自动选择精度，也可以指定为"half"或"float16"等
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    # 测试用的提示语
    prompts = [
        "介绍一下中国的人工智能发展现状。",
        "写一首关于春天的诗。",
        "解释一下量子计算的基本原理。"
    ]
    
    # 进行推理
    print("开始生成回答...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 打印结果
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n问题 {i+1}: {prompt}")
        print(f"回答: {generated_text}")
        print("="*50)

if __name__ == "__main__":
    main()