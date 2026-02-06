

测试 Linear
```
(cs336) (cs336-basics) PS D:\WorkSpace\cs336\assignment1-basics> 
uv run pytest tests/test_model.py::test_linear -v 
```
测试 Embedding 
```python
uv run pytest tests/test_model.py::test_embedding -v 
```
测试 RoPE
```
(base) PS E:\Workspace\cs336-scratch\cs336> uv run pytest tests/test_model.py::test_rope -v 
```

测试多头注意力
```
uv run pytest -k tests/test_model.py::test_multihead_self_attention  -v 
```

测试Transformer 

```
uv run pytest -k tests/test_model.py::test_transformer_lm -v 
```