from datasets import load_dataset

# 加载数据集
dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]

# 查看数据集基本信息
print(dataset)

# 查看数据集的特征（列名）
print(dataset.features)

# 查看数据集的前几个样本
# print(dataset[:5])

print(dataset[0].keys())  # 查看所有字段
print(dataset[0])  # 查看第一个样本的完整内容


# # 查看特定字段
# print(dataset[0]["problem"])  # 假设有 "problem" 字段