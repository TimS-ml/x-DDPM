"""
Attention Module for Diffusion Models

This module provides an optimized attention mechanism that supports both
standard attention and Flash Attention for improved performance. It automatically
selects the best backend based on the available hardware (CPU/CUDA, GPU generation).
"""

from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from torch.nn.attention import SDPBackend

# constants

# Named tuple to store attention backend configurations
AttentionConfig = namedtuple('AttentionConfig', ['backends'])

# helpers

def exists(val):
    """
    Check if a value exists (is not None).

    Args:
        val: Value to check

    Returns:
        bool: True if value is not None, False otherwise
    """
    return val is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return the default.

    Args:
        val: Value to check
        d: Default value to return if val is None

    Returns:
        The value if it exists, otherwise the default
    """
    return val if exists(val) else d

def once(fn):
    """
    Decorator that ensures a function is only called once.
    Subsequent calls will return None without executing the function.

    Args:
        fn: Function to wrap

    Returns:
        Wrapped function that only executes once
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# Utility to print a message only once, useful for logging hardware detection
print_once = once(print)

# main class

class Attend(nn.Module):
    """
    Optimized Attention Module with Flash Attention Support

    This module implements multi-head attention with automatic backend selection.
    It supports both standard scaled dot-product attention and Flash Attention
    for improved memory efficiency and speed.

    The module automatically detects hardware capabilities and selects the most
    efficient attention backend (Flash Attention on A100+, memory-efficient
    attention on older GPUs, or standard attention as fallback).

    Args:
        dropout (float): Dropout probability for attention weights. Default: 0.0
        flash (bool): Whether to enable Flash Attention if available. Default: False
        scale (float, optional): Custom scaling factor for attention scores.
                                If None, uses 1/sqrt(d_k) scaling. Default: None
    """
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        # Flash Attention requires PyTorch 2.0 or above
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # Determine efficient attention configs for CUDA and CPU
        # CPU config tries Flash -> Math -> Efficient backends in order
        self.cpu_config = AttentionConfig([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        self.cuda_config = None

        # Skip CUDA config if CUDA is not available or flash is disabled
        if not torch.cuda.is_available() or not flash:
            return

        # Detect GPU generation to select optimal attention backend
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        # A100 (compute capability 8.0+) supports Flash Attention efficiently
        if device_version > version.parse('8.0'):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.FLASH_ATTENTION])
        else:
            # Older GPUs use memory-efficient or math backends
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])

    def flash_attn(self, q, k, v):
        """
        Flash Attention implementation using PyTorch's scaled_dot_product_attention.

        This method provides memory-efficient attention computation using hardware-
        optimized kernels. It automatically selects the best backend based on the
        device (CUDA/CPU) and GPU generation.

        Args:
            q (Tensor): Query tensor of shape (batch, heads, q_len, dim)
            k (Tensor): Key tensor of shape (batch, heads, k_len, dim)
            v (Tensor): Value tensor of shape (batch, heads, k_len, dim)

        Returns:
            Tensor: Attention output of shape (batch, heads, q_len, dim)
        """
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Apply custom scaling if provided
        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        # Ensure tensors are contiguous in memory for optimal performance
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Select the appropriate attention backend based on device
        config = self.cuda_config if is_cuda else self.cpu_config

        # Use PyTorch 2.0's scaled_dot_product_attention with the selected backend
        # This function handles the attention computation: softmax(QK^T / sqrt(d))V
        with torch.nn.attention.sdpa_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        Compute multi-head attention.

        This method computes the scaled dot-product attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

        Einstein notation used in this method:
        b - batch size
        h - number of attention heads
        i, j - sequence positions (query and key/value respectively)
        d - feature dimension per head

        Args:
            q (Tensor): Query tensor of shape (batch, heads, q_len, dim)
            k (Tensor): Key tensor of shape (batch, heads, k_len, dim)
            v (Tensor): Value tensor of shape (batch, heads, k_len, dim)

        Returns:
            Tensor: Attention output of shape (batch, heads, q_len, dim)
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # Use Flash Attention if enabled
        if self.flash:
            return self.flash_attn(q, k, v)

        # Standard attention computation
        # Use custom scale if provided, otherwise use 1/sqrt(d_k)
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # Compute similarity scores: QK^T scaled by sqrt(d_k)
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # Apply softmax to get attention weights
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # Aggregate values using attention weights
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
