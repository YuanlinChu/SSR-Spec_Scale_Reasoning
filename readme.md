# SpecLRM 推理实验项目

本项目实现了一系列基于大型语言模型的数学推理实验，包括基准测试、并行推理和推测性推理（Speculative Reasoning）等方法。

## 项目结构

### reason

- **base_test.py**: 用于基准测试的实现，通过直接使用大型语言模型一次性生成完整的推理过程，评估模型在数学问题上的基础性能。

### scale + reason

- **scale_reason.py**: 实现了基础模型的并行推理实验，通过同时运行多个推理轨道（使用不同角色提示），提高解题效率和准确率。（不使用--use_role_prompts为无提示并行）

### spec + scale + reason

- **spec_scale_reason.py**: 实现了推测性推理（Speculative Reasoning）的核心实验，通过小模型生成推理步骤，大模型评分和修正的方式，结合多角色并行推理提高效率。

- **spec_scale_reason_async.py**: 在推测性推理基础上增加了异步并行处理功能，理论上可以进一步提高推理效率，但目前尚未完全调试通过。

- **spec_scale_reason_2.py**: 在推测性推理的基础上增加了解题方法路径选择模块，通过为不同问题选择合适的解题策略来提高解题效率和准确率。

### 评估模块

- **baseline_eval.py**: 用于评测 base_test.py 和 scale_reason.py 的输出结果，计算准确率和其他性能指标。

- **spec_eval.py**: 用于评测推测性推理相关实验（spec_scale_reason.py、spec_scale_reason_async.py 和 spec_scale_reason_2.py）的输出结果。

## 主要特点

- **多角色推理**：通过不同角色提示引导模型从不同角度思考问题（已替换为方法提示）
- **推测性推理**：使用小模型生成推理步骤，大模型评分和修正，提高效率
- **方法路径选择**：为不同类型的问题选择合适的解题策略，通过不同方法提示引导模型从不同角度思考问题
- **异步并行处理**：尝试通过异步方式进一步提高推理效率（待定）

## 使用方法

### 基准测试

```bash
python base_test.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --model_name Qwen/QwQ-32B --output_dir results/baseline_vllm_test
```

### 并行推理

```bash
python scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --model_name Qwen/QwQ-32B --output_dir results/scale_reason --use_role_prompts
```

### 推测性推理

```bash
python spec_scale_reason.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale_Inf --score_threshold 7.0 --token_budget 8192 --score_method greedy
```

### 方法路径选择推理

```bash
python spec_scale_reason_2.py --dataset_name aime --problem_id 60-89 --repeat_id 3 --output_dir results/spec_scale_m --score_threshold 7.0 --token_budget 8192 --score_method greedy --method_num 3
```

### 评估结果

```bash
python baseline_eval.py --dataset_name aime --model_name QwQ-32B --results_dir results/baseline_vllm_test
python spec_eval.py --dataset_name aime --model_name QwQ-32b --results_dir results/spec_scale_m
```

## 数据集支持

本项目支持多种数学推理数据集：
- AIME（美国数学邀请赛）
- MATH（复杂数学问题集）
- GPQA（多选题问答）
