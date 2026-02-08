# 测试命令列表

## 已实现的测试（可以运行）

### test_model.py - 模型相关测试
```bash
uv run pytest tests/test_model.py::test_linear -v
uv run pytest tests/test_model.py::test_embedding -v
uv run pytest tests/test_model.py::test_swiglu -v
uv run pytest tests/test_model.py::test_scaled_dot_product_attention -v
uv run pytest tests/test_model.py::test_4d_scaled_dot_product_attention -v
uv run pytest tests/test_model.py::test_multihead_self_attention -v
uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -v
uv run pytest tests/test_model.py::test_transformer_lm -v
uv run pytest tests/test_model.py::test_transformer_lm_truncated_input -v
uv run pytest tests/test_model.py::test_transformer_block -v
uv run pytest tests/test_model.py::test_rmsnorm -v
uv run pytest tests/test_model.py::test_rope -v
uv run pytest tests/test_model.py::test_silu_matches_pytorch -v
```

### test_nn_utils.py - 神经网络工具函数测试
```bash
uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -v
uv run pytest tests/test_nn_utils.py::test_cross_entropy -v
uv run pytest tests/test_nn_utils.py::test_gradient_clipping -v
```

### test_optimizer.py - 优化器测试
```bash
uv run pytest tests/test_optimizer.py::test_adamw -v
uv run pytest tests/test_optimizer.py::test_get_lr_cosine_schedule -v
```

### test_serialization.py - 序列化测试
```bash
uv run pytest tests/test_serialization.py::test_checkpointing -v
```

### test_data.py - 数据加载测试
```bash
uv run pytest tests/test_data.py::test_get_batch -v
```

### test_tokenizer.py - Tokenizer 测试
```bash
uv run pytest tests/test_tokenizer.py::test_roundtrip_empty -v
uv run pytest tests/test_tokenizer.py::test_empty_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_roundtrip_single_character -v
uv run pytest tests/test_tokenizer.py::test_single_character_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_roundtrip_single_unicode_character -v
uv run pytest tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_roundtrip_ascii_string -v
uv run pytest tests/test_tokenizer.py::test_ascii_string_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_roundtrip_unicode_string -v
uv run pytest tests/test_tokenizer.py::test_unicode_string_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens -v
uv run pytest tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_overlapping_special_tokens -v
uv run pytest tests/test_tokenizer.py::test_address_roundtrip -v
uv run pytest tests/test_tokenizer.py::test_address_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_german_roundtrip -v
uv run pytest tests/test_tokenizer.py::test_german_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_tinystories_sample_roundtrip -v
uv run pytest tests/test_tokenizer.py::test_tinystories_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_encode_special_token_trailing_newlines -v
uv run pytest tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace -v
uv run pytest tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip -v
uv run pytest tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken -v
uv run pytest tests/test_tokenizer.py::test_encode_iterable_memory_usage -v
uv run pytest tests/test_tokenizer.py::test_encode_memory_usage -v
```

### test_train_bpe.py - BPE 训练测试
```bash
uv run pytest tests/test_train_bpe.py::test_train_bpe_speed -v
uv run pytest tests/test_train_bpe.py::test_train_bpe -v
uv run pytest tests/test_train_bpe.py::test_train_bpe_special_tokens -v
```

## ⚠️ 需要注意的测试（adapters.py 中有 NotImplementedError）

以下测试对应的 adapter 函数还未实现，运行这些测试会失败：

1. **test_scaled_dot_product_attention** - `run_scaled_dot_product_attention` 未实现
2. **test_4d_scaled_dot_product_attention** - `run_scaled_dot_product_attention` 未实现
3. **test_transformer_block** - `run_transformer_block` 未实现
4. **test_gradient_clipping** - `run_gradient_clipping` 函数末尾有 `raise NotImplementedError`（可能是误留的）

## 快速运行所有测试

```bash
# 运行所有测试
uv run pytest -v

# 运行特定文件的所有测试
uv run pytest tests/test_model.py -v
uv run pytest tests/test_nn_utils.py -v
uv run pytest tests/test_optimizer.py -v
uv run pytest tests/test_serialization.py -v
uv run pytest tests/test_data.py -v
uv run pytest tests/test_tokenizer.py -v
uv run pytest tests/test_train_bpe.py -v
```
