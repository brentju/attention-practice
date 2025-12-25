"""
Test suite for Triton-based FlashAttention kernels.
Designed for Phase 2+ validation of fused attention implementations.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# Try to import triton - skip tests if not available
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Try to import custom Triton attention implementation
try:
    from triton_attention import flash_attn_forward, flash_attn_backward
    TRITON_ATTENTION_AVAILABLE = True
except ImportError:
    TRITON_ATTENTION_AVAILABLE = False


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonAttentionForward:
    """
    Test suite for Triton FlashAttention forward pass.
    For Phase 2 validation.
    """
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_small(self):
        """Test forward pass with small matrices."""
        batch_size, seq_len, d_k, d_v = 2, 32, 64, 64
        num_heads = 1
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16)
        
        # Triton implementation
        out_triton = flash_attn_forward(q, k, v)
        
        # PyTorch reference
        q_ref = q.transpose(1, 2)  # (B, H, N, D) -> (B, N, H, D)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_ref = out_ref.transpose(1, 2)  # Back to (B, H, N, D)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_medium(self):
        """Test forward pass with medium-sized matrices."""
        batch_size, seq_len, d_k, d_v = 4, 128, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16)
        
        out_triton = flash_attn_forward(q, k, v)
        
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_ref = out_ref.transpose(1, 2)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_long_sequence(self):
        """Test forward pass with long sequences (FlashAttention's strength)."""
        batch_size, seq_len, d_k, d_v = 2, 512, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16)
        
        out_triton = flash_attn_forward(q, k, v)
        
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_ref = out_ref.transpose(1, 2)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_very_long_sequence(self):
        """Test forward pass with very long sequences."""
        batch_size, seq_len, d_k, d_v = 1, 1024, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16)
        
        out_triton = flash_attn_forward(q, k, v)
        
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_ref = out_ref.transpose(1, 2)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_causal_mask(self):
        """Test forward pass with causal mask (Phase 2.1 requirement)."""
        batch_size, seq_len, d_k, d_v = 2, 128, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
        
        out_triton = flash_attn_forward(q, k, v, causal=True)
        
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        causal_mask_ref = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=causal_mask_ref)
        out_ref = out_ref.transpose(1, 2)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_head_dim_64(self):
        """Test forward pass with head_dim=64 (Phase 2.1 requirement)."""
        batch_size, seq_len, d_k, d_v = 2, 128, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16)
        
        out_triton = flash_attn_forward(q, k, v)
        
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_ref = out_ref.transpose(1, 2)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_head_dim_128(self):
        """Test forward pass with head_dim=128 (Phase 2.1 requirement)."""
        batch_size, seq_len, d_k, d_v = 2, 128, 128, 128
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16)
        
        out_triton = flash_attn_forward(q, k, v)
        
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_ref = out_ref.transpose(1, 2)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_forward_bf16(self):
        """Test forward pass with BF16 precision."""
        if not torch.cuda.is_available():
            pytest.skip("BF16 requires CUDA")
        
        batch_size, seq_len, d_k, d_v = 2, 128, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.bfloat16)
        
        out_triton = flash_attn_forward(q, k, v)
        
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
        out_ref = out_ref.transpose(1, 2)
        
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonAttentionBackward:
    """
    Test suite for Triton FlashAttention backward pass.
    Critical for Phase 3 validation.
    """
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_backward_small(self):
        """Test backward pass with small matrices."""
        batch_size, seq_len, d_k, d_v = 2, 32, 64, 64
        num_heads = 1
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16, requires_grad=True)
        
        # Triton implementation
        out_triton = flash_attn_forward(q, k, v)
        loss_triton = out_triton.sum()
        loss_triton.backward()
        
        grad_q_triton = q.grad.clone()
        grad_k_triton = k.grad.clone()
        grad_v_triton = v.grad.clone()
        
        # Reset gradients
        q.grad = None
        k.grad = None
        v.grad = None
        
        # PyTorch reference
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        
        q_ref_t = q_ref.transpose(1, 2)
        k_ref_t = k_ref.transpose(1, 2)
        v_ref_t = v_ref.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref_t, k_ref_t, v_ref_t)
        out_ref = out_ref.transpose(1, 2)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        
        grad_q_ref = q_ref.grad
        grad_k_ref = k_ref.grad
        grad_v_ref = v_ref.grad
        
        # Compare gradients
        assert torch.allclose(grad_q_triton, grad_q_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_k_triton, grad_k_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_v_triton, grad_v_ref, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_backward_medium(self):
        """Test backward pass with medium-sized matrices."""
        batch_size, seq_len, d_k, d_v = 4, 128, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out_triton = flash_attn_forward(q, k, v)
        loss_triton = out_triton.sum()
        loss_triton.backward()
        
        grad_q_triton = q.grad.clone()
        grad_k_triton = k.grad.clone()
        grad_v_triton = v.grad.clone()
        
        q.grad = None
        k.grad = None
        v.grad = None
        
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        
        q_ref_t = q_ref.transpose(1, 2)
        k_ref_t = k_ref.transpose(1, 2)
        v_ref_t = v_ref.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref_t, k_ref_t, v_ref_t)
        out_ref = out_ref.transpose(1, 2)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        
        assert torch.allclose(grad_q_triton, q_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_k_triton, k_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_v_triton, v_ref.grad, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_backward_causal_mask(self):
        """Test backward pass with causal mask."""
        batch_size, seq_len, d_k, d_v = 2, 128, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out_triton = flash_attn_forward(q, k, v, causal=True)
        loss_triton = out_triton.sum()
        loss_triton.backward()
        
        grad_q_triton = q.grad.clone()
        grad_k_triton = k.grad.clone()
        grad_v_triton = v.grad.clone()
        
        q.grad = None
        k.grad = None
        v.grad = None
        
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        q_ref_t = q_ref.transpose(1, 2)
        k_ref_t = k_ref.transpose(1, 2)
        v_ref_t = v_ref.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref_t, k_ref_t, v_ref_t, attn_mask=causal_mask)
        out_ref = out_ref.transpose(1, 2)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        
        assert torch.allclose(grad_q_triton, q_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_k_triton, k_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_v_triton, v_ref.grad, atol=1e-2, rtol=1e-2)
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_backward_long_sequence(self):
        """Test backward pass with long sequences."""
        batch_size, seq_len, d_k, d_v = 2, 512, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out_triton = flash_attn_forward(q, k, v)
        loss_triton = out_triton.sum()
        loss_triton.backward()
        
        grad_q_triton = q.grad.clone()
        grad_k_triton = k.grad.clone()
        grad_v_triton = v.grad.clone()
        
        q.grad = None
        k.grad = None
        v.grad = None
        
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        
        q_ref_t = q_ref.transpose(1, 2)
        k_ref_t = k_ref.transpose(1, 2)
        v_ref_t = v_ref.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_ref_t, k_ref_t, v_ref_t)
        out_ref = out_ref.transpose(1, 2)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        
        assert torch.allclose(grad_q_triton, q_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_k_triton, k_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_v_triton, v_ref.grad, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonAttentionAutograd:
    """
    Test suite for autograd integration with Triton kernels.
    For Phase 3 validation.
    """
    
    @pytest.mark.skipif(not TRITON_ATTENTION_AVAILABLE, reason="Triton attention not implemented")
    def test_autograd_function(self):
        """Test that Triton attention can be wrapped in autograd Function."""
        try:
            from triton_attention import FlashAttentionFunction
        except ImportError:
            pytest.skip("FlashAttentionFunction not implemented")
        
        batch_size, seq_len, d_k, d_v = 2, 64, 64, 64
        num_heads = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, d_v, device='cuda', dtype=torch.float16, requires_grad=True)
        
        # Use autograd Function
        out = FlashAttentionFunction.apply(q, k, v)
        loss = out.sum()
        loss.backward()
        
        # Verify gradients exist
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        
        # Verify gradients are finite
        assert torch.isfinite(q.grad).all()
        assert torch.isfinite(k.grad).all()
        assert torch.isfinite(v.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

