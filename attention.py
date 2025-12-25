"""
Attention mechanism implementation.
Implement the attention function to pass the test suite.
"""

import numpy as np
from typing import Optional, Tuple


def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention implementation.
    
    Args:
        query: Query array of shape (..., seq_len_q, d_k)
        key: Key array of shape (..., seq_len_k, d_k)
        value: Value array of shape (..., seq_len_v, d_v)
        mask: Optional attention mask of shape (..., seq_len_q, seq_len_k)
        scale: Optional scaling factor (defaults to 1/sqrt(d_k))
    
    Returns:
        output: Attention output array of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights array of shape (..., seq_len_q, seq_len_k)
    """
    # TODO: Implement scaled dot-product attention
    # Steps:
    # 1. Compute attention scores: Q @ K^T
    # 2. Scale by 1/sqrt(d_k) or provided scale factor
    # 3. Apply mask if provided (set masked positions to -inf)
    # 4. Apply softmax to get attention weights
    # 5. Apply attention weights to values: weights @ V
    
    raise NotImplementedError("Please implement the attention function")


