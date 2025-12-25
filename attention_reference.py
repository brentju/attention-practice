"""
Reference implementation of scaled dot-product attention.
This is a working implementation for reference - implement your own in attention.py!
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
    # Get the dimension of the key (d_k)
    d_k = query.shape[-1]
    
    # Set default scale if not provided
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)
    
    # Compute attention scores: Q @ K^T
    # query: (..., seq_len_q, d_k)
    # key: (..., seq_len_k, d_k)
    # scores: (..., seq_len_q, seq_len_k)
    scores = np.matmul(query, np.swapaxes(key, -2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        # Set masked positions to negative infinity
        scores = np.where(mask == 0, float('-inf'), scores)
    
    # Apply softmax to get attention weights
    # Subtract max for numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Apply attention weights to values
    # attention_weights: (..., seq_len_q, seq_len_k)
    # value: (..., seq_len_k, d_v)
    # output: (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

