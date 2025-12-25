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
    scores = np.matmul(query, np.swapaxes(key, -2, -1))
    # 2. Scale by 1/sqrt(d_k) or provided scale factor
    if scale is None:
        scale = 1.0 / np.sqrt(query.shape[-1])
    scores *= scale
    # 3. Apply mask if provided (set masked positions to -inf)
    if mask is not None:
        scores = np.where(mask == 0, float('-inf'), scores)
    # 4. Apply softmax to get attention weights
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        # Subtracting the max for numerical stability
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    attention_weights = softmax(scores)
    # 5. Apply attention weights to values: weights @ V
    output = attention_weights @ value
    return (output, attention_weights)


