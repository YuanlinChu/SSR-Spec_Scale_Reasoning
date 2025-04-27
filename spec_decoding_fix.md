# SpecDecode vLLM 0.8.2 Fix

In your downloaded huggingface model folder, update the vocab_size config of draft model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) to `152064`.

Modify the following lines to pad vocab embedding tensor in `vllm/model_executor/layers/vocab_parallel_embedding.py`

Line 384:

```
# original
assert loaded_weight.shape[output_dim] == self.org_vocab_size

# modified
assert loaded_weight.shape[output_dim] <= self.org_vocab_size   # from `==`  to  `<=`
```

Line 387:

```
# original
loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

# modified
if shard_size <= loaded_weight.shape[0]:
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

Add this at the beginning of `weight_loader` function in `vllm/model_executor/layers/vocab_parallel_embedding.py`

```
if loaded_weight.shape[0] == 151936:
    padding = torch.zeros((128, 1536), dtype=loaded_weight.dtype, device=loaded_weight.device)
    loaded_weight = torch.cat([loaded_weight, padding], dim=0)
```

Afterward, [install vLLM from source](https://docs.vllm.ai/en/v0.8.2/getting_started/installation/gpu.html#build-wheel-from-source):

```
git clone https://github.com/vllm-project/vllm.git
cd vllm
# Do the above modifications to the codebase
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

And use this command to launch the vLLM engine:

```
VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --dtype auto -tp 2 --max_model_len 16384 --gpu-memory-utilization 0.9 --port 30000 --speculative_config '{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "num_speculative_tokens": 5}'
```
