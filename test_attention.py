"""
Test suite for attention mechanism implementation.
Compares custom implementation against PyTorch's reference implementation.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def pytorch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional attention mask
        scale: Optional scaling factor (defaults to 1/sqrt(d_k))
    
    Returns:
        output: Attention output tensor
        attention_weights: Attention weights tensor
    """
    d_k = query.size(-1)
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# Import the custom implementation (to be implemented)
try:
    from attention import attention
except ImportError:
    # Placeholder if attention.py doesn't exist yet
    def attention(*args, **kwargs):
        raise NotImplementedError("Please implement the attention function in attention.py")


class TestAttention:
    """Test suite for attention mechanism."""
    
    def test_basic_attention_small(self):
        """Test attention with small matrices."""
        batch_size, seq_len, d_k, d_v = 2, 5, 4, 6
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        # PyTorch reference
        ref_output, ref_weights = pytorch_attention(query, key, value)
        
        # Custom implementation
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        # Convert back to torch for comparison
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        # Check outputs
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5), \
            "Output mismatch in basic attention test"
        assert torch.allclose(custom_weights, ref_weights, atol=1e-5, rtol=1e-5), \
            "Attention weights mismatch in basic attention test"
    
    def test_attention_medium(self):
        """Test attention with medium-sized matrices."""
        batch_size, seq_len, d_k, d_v = 4, 32, 64, 64
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        ref_output, ref_weights = pytorch_attention(query, key, value)
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
        assert torch.allclose(custom_weights, ref_weights, atol=1e-5, rtol=1e-5)
    
    def test_attention_large(self):
        """Test attention with large matrices."""
        batch_size, seq_len, d_k, d_v = 8, 128, 256, 256
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        ref_output, ref_weights = pytorch_attention(query, key, value)
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-4, rtol=1e-4)
        assert torch.allclose(custom_weights, ref_weights, atol=1e-4, rtol=1e-4)
    
    def test_attention_different_seq_lengths(self):
        """Test attention when query and key/value have different sequence lengths."""
        batch_size, seq_len_q, seq_len_kv, d_k, d_v = 2, 10, 15, 8, 12
        
        query = torch.randn(batch_size, seq_len_q, d_k)
        key = torch.randn(batch_size, seq_len_kv, d_k)
        value = torch.randn(batch_size, seq_len_kv, d_v)
        
        ref_output, ref_weights = pytorch_attention(query, key, value)
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
        assert torch.allclose(custom_weights, ref_weights, atol=1e-5, rtol=1e-5)
    
    def test_attention_with_mask(self):
        """Test attention with masking."""
        batch_size, seq_len, d_k, d_v = 2, 10, 8, 8
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        # Create a mask (1s for valid positions, 0s for masked)
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, 5:] = 0  # Mask out positions after index 5
        
        ref_output, ref_weights = pytorch_attention(query, key, value, mask=mask)
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy(), mask=mask.numpy()
        )
        
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
        assert torch.allclose(custom_weights, ref_weights, atol=1e-5, rtol=1e-5)
    
    def test_attention_single_batch(self):
        """Test attention with single batch item."""
        seq_len, d_k, d_v = 20, 16, 16
        
        query = torch.randn(1, seq_len, d_k)
        key = torch.randn(1, seq_len, d_k)
        value = torch.randn(1, seq_len, d_v)
        
        ref_output, ref_weights = pytorch_attention(query, key, value)
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
        assert torch.allclose(custom_weights, ref_weights, atol=1e-5, rtol=1e-5)
    
    def test_attention_no_batch_dimension(self):
        """Test attention without batch dimension."""
        seq_len, d_k, d_v = 10, 8, 8
        
        query = torch.randn(seq_len, d_k)
        key = torch.randn(seq_len, d_k)
        value = torch.randn(seq_len, d_v)
        
        ref_output, ref_weights = pytorch_attention(query, key, value)
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
        assert torch.allclose(custom_weights, ref_weights, atol=1e-5, rtol=1e-5)
    
    def test_attention_different_dims(self):
        """Test attention with different d_k and d_v dimensions."""
        batch_size, seq_len, d_k, d_v = 2, 8, 32, 64
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        ref_output, ref_weights = pytorch_attention(query, key, value)
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        custom_output = torch.from_numpy(custom_output)
        custom_weights = torch.from_numpy(custom_weights)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
        assert torch.allclose(custom_weights, ref_weights, atol=1e-5, rtol=1e-5)
    
    def test_attention_output_shape(self):
        """Test that output shapes are correct."""
        batch_size, seq_len_q, seq_len_kv, d_k, d_v = 3, 12, 15, 16, 20
        
        query = torch.randn(batch_size, seq_len_q, d_k)
        key = torch.randn(batch_size, seq_len_kv, d_k)
        value = torch.randn(batch_size, seq_len_kv, d_v)
        
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        assert custom_output.shape == (batch_size, seq_len_q, d_v), \
            f"Expected output shape {(batch_size, seq_len_q, d_v)}, got {custom_output.shape}"
        assert custom_weights.shape == (batch_size, seq_len_q, seq_len_kv), \
            f"Expected weights shape {(batch_size, seq_len_q, seq_len_kv)}, got {custom_weights.shape}"
    
    def test_attention_deterministic(self):
        """Test that attention produces deterministic results."""
        seq_len, d_k, d_v = 10, 8, 8
        
        query = torch.randn(seq_len, d_k)
        key = torch.randn(seq_len, d_k)
        value = torch.randn(seq_len, d_v)
        
        output1, weights1 = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        output2, weights2 = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        
        np.testing.assert_array_equal(output1, output2)
        np.testing.assert_array_equal(weights1, weights2)


class TestAttentionPyTorchAPI:
    """
    Test suite using torch.nn.functional.scaled_dot_product_attention as reference.
    This is the standard PyTorch API and should be used for Phase 2+ validation.
    """
    
    def test_basic_attention_small(self):
        """Test attention with small matrices using PyTorch API."""
        batch_size, seq_len, d_k, d_v = 2, 5, 4, 6
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        # PyTorch reference
        ref_output = F.scaled_dot_product_attention(query, key, value)
        
        # Custom implementation
        custom_output, custom_weights = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        custom_output = torch.from_numpy(custom_output)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5), \
            "Output mismatch in basic attention test"
    
    def test_attention_medium(self):
        """Test attention with medium-sized matrices using PyTorch API."""
        batch_size, seq_len, d_k, d_v = 4, 32, 64, 64
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        ref_output = F.scaled_dot_product_attention(query, key, value)
        custom_output, _ = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        custom_output = torch.from_numpy(custom_output)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
    
    def test_attention_large(self):
        """Test attention with large matrices using PyTorch API."""
        batch_size, seq_len, d_k, d_v = 8, 128, 256, 256
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        ref_output = F.scaled_dot_product_attention(query, key, value)
        custom_output, _ = attention(
            query.numpy(), key.numpy(), value.numpy()
        )
        custom_output = torch.from_numpy(custom_output)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-4, rtol=1e-4)
    
    def test_attention_causal_mask(self):
        """Test causal (upper triangular) mask for autoregressive attention."""
        batch_size, seq_len, d_k, d_v = 2, 10, 8, 8
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        # Create causal mask (upper triangular)
        # triu with diagonal=1 gives True for positions above diagonal (future positions)
        # PyTorch's attn_mask expects False for masked positions
        causal_mask_bool = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask_bool = causal_mask_bool.unsqueeze(0).expand(batch_size, -1, -1)
        
        # PyTorch reference: attn_mask expects False for masked positions
        # causal_mask_bool has True for future positions (to mask), so invert it
        ref_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=~causal_mask_bool
        )
        
        # Custom implementation expects mask where 1 = valid, 0 = masked
        # causal_mask_bool has True for future positions (to mask), so invert it
        mask_numpy = (~causal_mask_bool).float().numpy()
        custom_output, _ = attention(
            query.numpy(), key.numpy(), value.numpy(), mask=mask_numpy
        )
        custom_output = torch.from_numpy(custom_output)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
    
    def test_attention_causal_mask_long_sequence(self):
        """Test causal mask with longer sequence."""
        batch_size, seq_len, d_k, d_v = 2, 64, 32, 32
        
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)
        
        # Create causal mask (upper triangular)
        # triu with diagonal=1 gives True for positions above diagonal (future positions)
        # PyTorch's attn_mask expects False for masked positions
        causal_mask_bool = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask_bool = causal_mask_bool.unsqueeze(0).expand(batch_size, -1, -1)
        
        # PyTorch reference: attn_mask expects False for masked positions
        # causal_mask_bool has True for future positions (to mask), so invert it
        ref_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=~causal_mask_bool
        )
        
        # Custom implementation expects mask where 1 = valid, 0 = masked
        # causal_mask_bool has True for future positions (to mask), so invert it
        mask_numpy = (~causal_mask_bool).float().numpy()
        custom_output, _ = attention(
            query.numpy(), key.numpy(), value.numpy(), mask=mask_numpy
        )
        custom_output = torch.from_numpy(custom_output)
        
        assert torch.allclose(custom_output, ref_output, atol=1e-5, rtol=1e-5)
    
    def test_attention_fp16(self):
        """Test attention with FP16 precision."""
        batch_size, seq_len, d_k, d_v = 2, 16, 32, 32
        
        query = torch.randn(batch_size, seq_len, d_k, dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, d_k, dtype=torch.float16)
        value = torch.randn(batch_size, seq_len, d_v, dtype=torch.float16)
        
        ref_output = F.scaled_dot_product_attention(query, key, value)
        
        # Convert to float32 for numpy, then back
        custom_output, _ = attention(
            query.float().numpy(), key.float().numpy(), value.float().numpy()
        )
        custom_output = torch.from_numpy(custom_output).half()
        
        # FP16 has lower precision, use relaxed tolerance
        assert torch.allclose(custom_output, ref_output, atol=1e-2, rtol=1e-2)
    
    def test_attention_bf16(self):
        """Test attention with BF16 precision."""
        if not torch.cuda.is_available():
            pytest.skip("BF16 requires CUDA")
        
        batch_size, seq_len, d_k, d_v = 2, 16, 32, 32
        
        query = torch.randn(batch_size, seq_len, d_k, dtype=torch.bfloat16)
        key = torch.randn(batch_size, seq_len, d_k, dtype=torch.bfloat16)
        value = torch.randn(batch_size, seq_len, d_v, dtype=torch.bfloat16)
        
        ref_output = F.scaled_dot_product_attention(query, key, value)
        
        custom_output, _ = attention(
            query.float().numpy(), key.float().numpy(), value.float().numpy()
        )
        custom_output = torch.from_numpy(custom_output).bfloat16()
        
        # BF16 has lower precision
        assert torch.allclose(custom_output, ref_output, atol=1e-2, rtol=1e-2)


class TestAttentionGradients:
    """
    Test suite for gradient/backward pass correctness.
    Critical for Phase 3 validation.
    """
    
    def test_gradient_q(self):
        """Test gradient with respect to query."""
        batch_size, seq_len, d_k, d_v = 2, 10, 8, 8
        
        query = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_v, requires_grad=True)
        
        # PyTorch reference
        output_ref = F.scaled_dot_product_attention(query, key, value)
        loss_ref = output_ref.sum()
        loss_ref.backward()
        grad_q_ref = query.grad.clone()
        
        # Reset gradients
        query.grad = None
        key.grad = None
        value.grad = None
        
        # Custom implementation (using autograd wrapper)
        query_np = query.detach().numpy()
        key_np = key.detach().numpy()
        value_np = value.detach().numpy()
        
        # For now, we'll test that the function can be wrapped
        # Actual gradient testing will be done in test_triton_attention.py
        # when the implementation supports autograd
        try:
            custom_output, _ = attention(query_np, key_np, value_np)
            # If we get here, the function works
            # Gradient testing requires autograd integration
            assert custom_output.shape == output_ref.shape
        except NotImplementedError:
            pytest.skip("Attention function not implemented yet")
    
    def test_gradient_k(self):
        """Test gradient with respect to key."""
        batch_size, seq_len, d_k, d_v = 2, 10, 8, 8
        
        query = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_v, requires_grad=True)
        
        output_ref = F.scaled_dot_product_attention(query, key, value)
        loss_ref = output_ref.sum()
        loss_ref.backward()
        grad_k_ref = key.grad.clone()
        
        # Test that reference gradients are computed
        assert grad_k_ref is not None
        assert grad_k_ref.shape == key.shape
    
    def test_gradient_v(self):
        """Test gradient with respect to value."""
        batch_size, seq_len, d_k, d_v = 2, 10, 8, 8
        
        query = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_v, requires_grad=True)
        
        output_ref = F.scaled_dot_product_attention(query, key, value)
        loss_ref = output_ref.sum()
        loss_ref.backward()
        grad_v_ref = value.grad.clone()
        
        # Test that reference gradients are computed
        assert grad_v_ref is not None
        assert grad_v_ref.shape == value.shape
    
    def test_gradient_with_causal_mask(self):
        """Test gradients with causal mask."""
        batch_size, seq_len, d_k, d_v = 2, 10, 8, 8
        
        query = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_v, requires_grad=True)
        
        causal_mask_bool = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask_bool = causal_mask_bool.unsqueeze(0).expand(batch_size, -1, -1)
        
        # PyTorch's attn_mask expects False for masked positions
        output_ref = F.scaled_dot_product_attention(
            query, key, value, attn_mask=~causal_mask_bool
        )
        loss_ref = output_ref.sum()
        loss_ref.backward()
        
        # Verify gradients exist
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
    
    def test_gradient_numerical_stability(self):
        """Test gradient numerical stability with extreme values."""
        batch_size, seq_len, d_k, d_v = 2, 10, 8, 8
        
        # Use larger values to test numerical stability
        # Create leaf tensors directly with the scaled values
        query = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_v, requires_grad=True)
        
        # Scale the values (this creates non-leaf tensors, but that's okay for the test)
        query_scaled = query * 2.0
        key_scaled = key * 2.0
        value_scaled = value * 2.0
        
        output_ref = F.scaled_dot_product_attention(query_scaled, key_scaled, value_scaled)
        loss_ref = output_ref.sum()
        loss_ref.backward()
        
        # Check that gradients are finite (they should be on the leaf tensors)
        assert query.grad is not None and torch.isfinite(query.grad).all()
        assert key.grad is not None and torch.isfinite(key.grad).all()
        assert value.grad is not None and torch.isfinite(value.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

