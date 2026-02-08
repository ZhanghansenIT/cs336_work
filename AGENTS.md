# Agent Guidelines for CS336 Assignment 1

This document provides guidelines for AI agents working on this codebase.

## Build, Lint, and Test Commands

### Environment Setup
```bash
uv sync                    # Install dependencies from pyproject.toml
uv run <command>          # Run any command in the managed environment
```

### Running Tests
```bash
uv run pytest             # Run all tests
uv run pytest -v          # Run all tests with verbose output
uv run pytest -k "test_name"  # Run specific test by name
uv run pytest tests/test_model.py::test_linear -v  # Run single test
uv run pytest -s          # Show print statements (configured in pyproject.toml)
```

### Linting
```bash
uv run ruff check .       # Run ruff linter
uv run ruff check --fix . # Auto-fix linting issues
```

## Code Style Guidelines

### Imports
- Use absolute imports for package modules: `from cs336_basics.Module import Class`
- Group imports: standard library → third-party → local
- Enable `from __future__ import annotations` for all files (PEP 563)

### Formatting
- Line length: 120 characters
- Use ruff for automatic formatting and linting
- Follow PEP 8 conventions

### Type Hints
- **Required**: Use `jaxtyping` for all tensor types with dimension annotations:
  ```python
  from jaxtyping import Float, Int
  from torch import Tensor
  Float[Tensor, " batch seq d_model"]
  Int[Tensor, " batch seq"]
  ```
- Use Python 3.11+ union syntax: `X | None` instead of `Optional[X]`
- Add type hints to all function parameters and return values
- Use `device: torch.device | None = None` pattern for device/dtype parameters

### Naming Conventions
- Classes: `PascalCase` (e.g., `Linear`, `RoPE`, `MultiheadSelfAttention`)
- Functions/methods: `snake_case` (e.g., `init_weights`, `forward`)
- Variables: `snake_case` (e.g., `d_model`, `d_ff`, `num_heads`)
- Private methods: `_snake_case` prefix
- Module names: `snake_case` (e.g., `BPETokenizer.py`)

### Error Handling
- Raise `NotImplementedError` for stub functions that need implementation
- Use descriptive error messages when appropriate
- Handle device/dtype placement explicitly in PyTorch modules

### PyTorch Module Patterns
- Initialize in `__init__` with proper device/dtype handling using factory kwargs
- Store `device` and `dtype` as instance attributes
- Use `nn.Parameter` for trainable weights
- Implement `forward()` method returning `torch.Tensor`
- Use `copy_()` for weight assignment from external weights

### File Structure
- Implementation files in `cs336_basics/`
- Test files in `tests/` with `test_*.py` naming
- Complete adapter functions in `tests/adapters.py` to connect implementations to tests

### Configuration
- `pyproject.toml` defines all dependencies and tools
- Linting configured with ruff (extends "UP" for pyupgrade)
- Pytest configured with `-s` and WARNING level logging
- Ignore `E402`, `F401`, `F403`, `E501` in `__init__.py` files
