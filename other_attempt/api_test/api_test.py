import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# 模型服务地址
MODEL_A_URL = "http://localhost:30000/v1"  # Qwen/QwQ-32B
MODEL_B_URL = "http://localhost:30001/v1"  # DeepSeek-1.5B

def query_model(model_name, base_url, prompt, temperature=0.7, top_p=0.95, max_tokens=100):
    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url,
        )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def main():
    prompts = [
        ("Qwen/QwQ-32B", MODEL_A_URL, "Explain quantum computing.", 0.7),
        ("Qwen/QwQ-32B", MODEL_A_URL, "Tell a joke about AI.", 0.9),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", MODEL_B_URL, "Write a poem about the sea.", 0.6),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", MODEL_B_URL, "Summarize the theory of relativity.", 0.3),
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(query_model, *args) for args in prompts]
        for i, future in enumerate(futures):
            print(f"\n--- Response {i + 1} ---\n{future.result()}\n")

if __name__ == "__main__":
    main()
