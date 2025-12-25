# Attention Mechanism Test Suite

This repository contains a comprehensive test suite for implementing the scaled dot-product attention mechanism, designed to support your learning path from basic attention (Phase 1) to FlashAttention-style fused kernels in Triton (Phases 2-6).

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Triton is only available on Linux with CUDA-capable GPUs. If you're on macOS or don't have CUDA, you can still run Phase 1 tests. Triton tests will automatically skip if Triton is not available.

For Phase 2+ (Triton kernels), install Triton separately on a Linux system with CUDA:

```bash
pip install triton>=2.0.0
# Or use the optional requirements file:
pip install -r requirements.txt -r requirements-triton.txt
```

## Test Files Overview

### `test_attention.py` - Basic Attention Tests

This file contains tests for the fundamental attention mechanism, suitable for Phase 1 learning.

**Test Classes:**

1. **`TestAttention`** (Phase 1)
   - Uses custom `pytorch_attention` reference implementation
   - Tests basic attention with various matrix sizes
   - Good for understanding the core algorithm
   - Tests: small, medium, large matrices, masking, different dimensions, etc.

2. **`TestAttentionPyTorchAPI`** (Phase 2+)
   - Uses `torch.nn.functional.scaled_dot_product_attention` as reference
   - Standard PyTorch API for validation
   - Includes causal mask tests (Phase 2.1 requirement)
   - Includes FP16/BF16 precision tests (Phase 2.1 requirement)
   - Tests: basic attention, causal masking, half-precision

3. **`TestAttentionGradients`** (Phase 3)
   - Tests gradient/backward pass correctness
   - Validates gradients with respect to Q, K, V
   - Tests numerical stability
   - Critical for Phase 3 validation

### `test_triton_attention.py` - Triton Kernel Tests

This file contains tests specifically for Triton-based FlashAttention kernels (Phases 2-6).

**Test Classes:**

1. **`TestTritonAttentionForward`** (Phase 2)
   - Tests forward pass of Triton kernels
   - Validates against PyTorch reference
   - Includes long sequence tests (512, 1024) - FlashAttention's strength
   - Tests causal mask support (Phase 2.1)
   - Tests head_dim ∈ {64, 128} (Phase 2.1)
   - Tests FP16/BF16 precision

2. **`TestTritonAttentionBackward`** (Phase 3)
   - Tests backward pass of Triton kernels
   - Validates gradients dQ, dK, dV
   - Critical for Phase 3 validation
   - Tests with causal masks and long sequences

3. **`TestTritonAttentionAutograd`** (Phase 3)
   - Tests autograd integration
   - Validates custom `torch.autograd.Function` wrapper
   - Ensures gradients flow correctly through the kernel

## Usage

### Running All Tests

Run the complete test suite:

```bash
pytest test_attention.py test_triton_attention.py -v
```

### Running Phase-Specific Tests

**Phase 1 (Basic Attention):**
```bash
pytest test_attention.py::TestAttention -v
```

**Phase 2 (PyTorch API + Triton Forward):**
```bash
pytest test_attention.py::TestAttentionPyTorchAPI -v
pytest test_triton_attention.py::TestTritonAttentionForward -v
```

**Phase 3 (Gradients + Triton Backward):**
```bash
pytest test_attention.py::TestAttentionGradients -v
pytest test_triton_attention.py::TestTritonAttentionBackward -v
pytest test_triton_attention.py::TestTritonAttentionAutograd -v
```

### Running Specific Test Cases

```bash
# Test causal mask
pytest test_attention.py::TestAttentionPyTorchAPI::test_attention_causal_mask -v

# Test FP16
pytest test_attention.py::TestAttentionPyTorchAPI::test_attention_fp16 -v

# Test long sequences
pytest test_triton_attention.py::TestTritonAttentionForward::test_forward_long_sequence -v
```

## Implementing Attention

### Phase 1: Basic Attention

Implement the `attention` function in `attention.py`. The function should:

1. Compute attention scores by matrix multiplying query and key (transposed)
2. Scale the scores by `1/sqrt(d_k)` (or use the provided `scale` parameter)
3. Apply a mask if provided (set masked positions to negative infinity)
4. Apply softmax to get attention weights
5. Multiply attention weights with values to get the output

The function signature is:

```python
def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

### Phase 2+: Triton Kernels

Implement Triton kernels in `triton_attention.py`:

```python
def flash_attn_forward(q, k, v, causal=False):
    """
    FlashAttention forward pass.
    
    Args:
        q: Query tensor (B, H, N, D) on CUDA, FP16/BF16
        k: Key tensor (B, H, N, D) on CUDA, FP16/BF16
        v: Value tensor (B, H, N, D) on CUDA, FP16/BF16
        causal: Whether to apply causal mask
    
    Returns:
        Output tensor (B, H, N, D)
    """
    # Your Triton kernel implementation
    pass
```

## Test Cases by Learning Phase

### Phase 1 - Mental Model
- ✅ Basic attention with small/medium/large matrices
- ✅ Different sequence lengths
- ✅ Masking support
- ✅ Shape validation
- ✅ Determinism checks

### Phase 2 - Forward Pass (Triton)
- ✅ Causal mask support
- ✅ FP16/BF16 precision
- ✅ head_dim ∈ {64, 128}
- ✅ Long sequences (512, 1024+)
- ✅ Comparison against `torch.nn.functional.scaled_dot_product_attention`

### Phase 3 - Backward Pass
- ✅ Gradient correctness (dQ, dK, dV)
- ✅ Gradient numerical stability
- ✅ Gradients with causal mask
- ✅ Autograd integration

### Phase 4+ - Extensions
- Tests can be extended for dropout, padding masks, etc.
- Benchmark tests (not included, but can be added)

## Reference Implementation

See `attention_reference.py` for a working NumPy implementation of basic attention. This is for reference only - implement your own version in `attention.py`.

## Notes

- All tests compare your implementation against PyTorch's reference implementation
- Tolerance levels are adjusted for different precisions (FP32: 1e-5, FP16/BF16: 1e-2)
- Triton tests require CUDA-capable GPU and will skip gracefully if unavailable
- Tests are designed to be run incrementally as you progress through the learning phases

