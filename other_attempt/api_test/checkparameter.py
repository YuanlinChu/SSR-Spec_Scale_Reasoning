from transformers import AutoConfig

# 你可以换成任意模型名称
model_names = [
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

for name in model_names:
    print(f"\nModel: {name}")
    config = AutoConfig.from_pretrained(name)
    
    # 打印关键参数
    print(f"  hidden_size           : {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  num_hidden_layers     : {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"  num_attention_heads   : {getattr(config, 'num_attention_heads', 'N/A')}")
    print(f"  intermediate_size     : {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"  vocab_size            : {getattr(config, 'vocab_size', 'N/A')}")
    print(f"  model_type            : {getattr(config, 'model_type', 'N/A')}")
