"""
Classifier-Free Guidance for Diffusion Models

This module implements a denoising diffusion probabilistic model (DDPM) with classifier-free guidance.
Classifier-free guidance is a technique that enables conditional image generation without requiring
a separate classifier network. Instead, it trains a single model that can be conditioned on class labels
while also learning an unconditional distribution through random conditioning dropout.

Key Components:
- Unet: A U-Net architecture that serves as the noise prediction network, with class conditioning
- GaussianDiffusion: The main diffusion process implementing forward noising and reverse denoising
- Classifier-Free Guidance: Implemented via conditional dropout during training and guidance scaling during sampling

The diffusion process works by:
1. Forward process: Gradually adding Gaussian noise to images over T timesteps
2. Reverse process: Learning to denoise images step-by-step, conditioned on class labels
3. Sampling: Generating new images by denoising from pure noise, with guidance strength control

Mathematical Background:
- The forward process: q(x_t | x_0) = N(x_t; sqrt(α_t) * x_0, (1 - α_t) * I)
- The reverse process: p_θ(x_{t-1} | x_t, c) where c is the class condition
- Classifier-free guidance: output = unconditional + guidance_scale * (conditional - unconditional)
"""

import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

# constants

# Named tuple to store model predictions during sampling
# pred_noise: predicted noise component
# pred_x_start: predicted clean image (x_0)
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    """
    Check if a value is not None.

    Args:
        x: Value to check

    Returns:
        bool: True if x is not None, False otherwise
    """
    return x is not None

def default(val, d):
    """
    Return val if it exists, otherwise return default value d.

    Args:
        val: Value to check
        d: Default value or callable that returns default value

    Returns:
        val if it exists, otherwise d (or d() if d is callable)
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    """
    Identity function that returns its first argument unchanged.

    Args:
        t: Input value
        *args: Ignored positional arguments
        **kwargs: Ignored keyword arguments

    Returns:
        The input value t unchanged
    """
    return t

def cycle(dl):
    """
    Infinitely cycle through a dataloader.

    Args:
        dl: Dataloader or iterable to cycle through

    Yields:
        Items from the dataloader, repeating infinitely
    """
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    """
    Check if a number has an integer square root.

    Args:
        num: Number to check

    Returns:
        bool: True if sqrt(num) is an integer, False otherwise
    """
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    """
    Divide a number into groups of a given size.

    Args:
        num: Total number to divide
        divisor: Size of each group

    Returns:
        list: List of group sizes, with remainder in last group if needed

    Example:
        num_to_groups(10, 3) returns [3, 3, 3, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    """
    Convert image to a specific format if needed.

    Args:
        img_type: Target image format (e.g., 'RGB', 'L')
        image: PIL Image object

    Returns:
        PIL Image: Converted image or original if already in correct format
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def pack_one_with_inverse(x, pattern):
    """
    Pack a tensor according to a pattern and return both packed tensor and inverse function.
    Uses einops pack/unpack operations.

    Args:
        x: Tensor to pack
        pattern: Einops pattern string for packing (e.g., 'b *')

    Returns:
        tuple: (packed tensor, inverse function to restore original shape)
    """
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalize image from [0, 1] range to [-1, 1] range.

    Diffusion models typically work with images normalized to [-1, 1].

    Args:
        img: Image tensor with values in [0, 1]

    Returns:
        Tensor: Image with values in [-1, 1]
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize tensor from [-1, 1] range back to [0, 1] range.

    Used to convert generated images back to standard image range.

    Args:
        t: Tensor with values in [-1, 1]

    Returns:
        Tensor: Values in [0, 1]
    """
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    """
    Generate uniform random values in [0, 1].

    Args:
        shape: Shape of tensor to generate
        device: Device to create tensor on

    Returns:
        Tensor: Random values uniformly distributed in [0, 1]
    """
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    """
    Generate a boolean mask where each element is True with probability 'prob'.

    Used for classifier-free guidance to randomly drop conditioning during training.

    Args:
        shape: Shape of mask to generate
        prob: Probability of each element being True
        device: Device to create mask on

    Returns:
        Tensor: Boolean mask with shape 'shape'
    """
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def project(x, y):
    """
    Project vector x onto vector y and compute orthogonal component.

    Decomposes x into:
    - parallel: component of x parallel to y
    - orthogonal: component of x orthogonal to y

    This is used in advanced guidance techniques to remove or scale the parallel component.

    Args:
        x: Vector to project (any shape)
        y: Vector to project onto (any shape)

    Returns:
        tuple: (parallel component, orthogonal component), both with original shape
    """
    # Pack tensors to flatten all dimensions except batch
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    # Use double precision for numerical stability
    dtype = x.dtype
    x, y = x.double(), y.double()

    # Normalize y to get unit vector
    unit = F.normalize(y, dim = -1)

    # Compute parallel component: projection of x onto y
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    # Compute orthogonal component: x minus its projection
    orthogonal = x - parallel

    # Restore original shape and dtype
    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# small helper modules

class Residual(nn.Module):
    """
    Residual wrapper that adds the input to the output of a function.

    Implements: output = fn(x) + x

    This is the core building block of residual networks, allowing gradients
    to flow more easily through deep networks.

    Args:
        fn: Function or module to wrap with residual connection
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """Apply function and add input (residual connection)."""
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    """
    Create an upsampling layer that doubles spatial resolution.

    Uses nearest-neighbor upsampling followed by convolution.

    Args:
        dim: Input channels
        dim_out: Output channels (defaults to input channels)

    Returns:
        nn.Sequential: Upsampling module
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    """
    Create a downsampling layer that halves spatial resolution.

    Uses strided convolution for downsampling.

    Args:
        dim: Input channels
        dim_out: Output channels (defaults to input channels)

    Returns:
        nn.Conv2d: Downsampling convolution (kernel=4, stride=2, padding=1)
    """
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes by the root mean square of activations and applies
    a learnable scale parameter. It's more efficient than LayerNorm as it
    doesn't center the activations.

    Args:
        dim: Number of channels to normalize
    """
    def __init__(self, dim):
        super().__init__()
        # Learnable scale parameter
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        """
        Apply RMS normalization.

        Normalizes along the channel dimension and scales by sqrt(channels).
        """
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    """
    Apply normalization before a function/layer.

    This follows the pre-norm architecture where normalization is applied
    before the transformation, as opposed to post-norm.

    Args:
        dim: Number of channels
        fn: Function/module to apply after normalization
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """Apply normalization then the function."""
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for timesteps.

    Encodes continuous timestep values into high-dimensional vectors using
    sine and cosine functions at different frequencies. This is similar to
    the positional encoding in Transformers.

    The encoding allows the network to learn time-dependent behaviors across
    different diffusion timesteps.

    Args:
        dim: Dimension of the positional embedding
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Encode timesteps into sinusoidal embeddings.

        Args:
            x: Timestep values, shape (batch,)

        Returns:
            Tensor: Positional embeddings, shape (batch, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        # Create frequencies decreasing exponentially from 1 to 1/10000
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Multiply timesteps by frequencies
        emb = x[:, None] * emb[None, :]
        # Concatenate sine and cosine of the products
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Random or learned sinusoidal positional embeddings.

    Following @crowsonkb's approach with random (optionally learned) sinusoidal embeddings.
    Reference: https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8

    This variant uses random Fourier features with learnable or fixed frequencies,
    providing a different inductive bias than standard sinusoidal embeddings.

    Args:
        dim: Dimension of the positional embedding (must be even)
        is_random: If True, frequencies are fixed (not learned). If False, frequencies are learnable.
    """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        # Random weights that can optionally be learned
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        """
        Encode timesteps using random Fourier features.

        Args:
            x: Timestep values, shape (batch,)

        Returns:
            Tensor: Fourier embeddings, shape (batch, dim + 1)
                   Includes original x concatenated with sin/cos features
        """
        x = rearrange(x, 'b -> b 1')
        # Compute frequencies: x * weights * 2π
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # Apply sin and cos to get Fourier features
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # Concatenate original timestep with Fourier features
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    """
    Basic convolutional block with normalization and activation.

    Applies: Conv -> Norm -> [Optional Scale/Shift] -> Activation

    The optional scale and shift allow conditioning on timestep or class embeddings
    via adaptive normalization (similar to AdaIN/AdaGN).

    Args:
        dim: Input channels
        dim_out: Output channels
    """
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x, scale_shift = None):
        """
        Forward pass with optional adaptive normalization.

        Args:
            x: Input tensor, shape (batch, dim, height, width)
            scale_shift: Optional tuple (scale, shift) for adaptive normalization

        Returns:
            Tensor: Processed features, shape (batch, dim_out, height, width)
        """
        x = self.proj(x)
        x = self.norm(x)

        # Apply adaptive normalization if conditioning is provided
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """
    Residual block with timestep and class conditioning.

    A ResNet-style block that can be conditioned on both timestep embeddings
    and class embeddings. The conditioning is applied via adaptive normalization
    (scale and shift parameters) in the first block.

    Architecture:
        x -> Block1 (with conditioning) -> Block2 -> + residual -> output

    Args:
        dim: Input channels
        dim_out: Output channels
        time_emb_dim: Dimension of time embedding (optional)
        classes_emb_dim: Dimension of class embedding (optional)
    """
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None):
        super().__init__()
        # MLP to project time and class embeddings to scale/shift parameters
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        # 1x1 conv to match dimensions if needed
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):
        """
        Forward pass with optional timestep and class conditioning.

        Args:
            x: Input tensor, shape (batch, dim, height, width)
            time_emb: Time embedding, shape (batch, time_emb_dim)
            class_emb: Class embedding, shape (batch, classes_emb_dim)

        Returns:
            Tensor: Output features, shape (batch, dim_out, height, width)
        """
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            # Concatenate available conditioning embeddings
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            # Project to scale and shift parameters (2 * dim_out total)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            # Split into scale and shift
            scale_shift = cond_emb.chunk(2, dim = 1)

        # First block with conditioning
        h = self.block1(x, scale_shift = scale_shift)

        # Second block without conditioning
        h = self.block2(h)

        # Add residual connection
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(n) complexity instead of O(n²).

    Linear attention approximates standard attention by applying softmax
    to keys and queries separately, reducing computational complexity
    from quadratic to linear in the sequence length.

    This is more efficient for processing feature maps at higher resolutions.

    Args:
        dim: Input channel dimension
        heads: Number of attention heads
        dim_head: Dimension per attention head
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # Project to queries, keys, values
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        Apply linear attention.

        Args:
            x: Input features, shape (batch, dim, height, width)

        Returns:
            Tensor: Attended features, shape (batch, dim, height, width)
        """
        b, c, h, w = x.shape
        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # Apply softmax to queries and keys separately (linear attention trick)
        q = q.softmax(dim = -2)  # Softmax over feature dimension
        k = k.softmax(dim = -1)  # Softmax over spatial dimension

        q = q * self.scale

        # Compute context: k^T @ v (reduces spatial dimension first)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Apply queries: q @ context
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    """
    Standard multi-head self-attention mechanism.

    Computes attention with O(n²) complexity, where n is the sequence length.
    Used at lower resolutions in the U-Net architecture for better global
    feature aggregation.

    Attention formula: softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        dim: Input channel dimension
        heads: Number of attention heads
        dim_head: Dimension per attention head
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5  # Scaling factor: 1/sqrt(dim_head)
        self.heads = heads
        hidden_dim = dim_head * heads

        # Project to queries, keys, values
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Apply multi-head self-attention.

        Args:
            x: Input features, shape (batch, dim, height, width)

        Returns:
            Tensor: Attended features, shape (batch, dim, height, width)
        """
        b, c, h, w = x.shape
        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # Scale queries
        q = q * self.scale

        # Compute attention scores: Q @ K^T
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        # Apply softmax to get attention weights
        attn = sim.softmax(dim = -1)
        # Apply attention to values: attn @ V
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # Reshape back to spatial format
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    """
    U-Net architecture for diffusion models with classifier-free guidance.

    A U-Net is an encoder-decoder architecture with skip connections, widely used
    in diffusion models for noise prediction. This implementation supports:

    - Classifier-Free Guidance: Random conditioning dropout during training
    - Class Conditioning: Conditional generation based on class labels
    - Timestep Conditioning: Adaptive behavior across diffusion timesteps
    - Multi-resolution Processing: Hierarchical feature extraction and generation

    The U-Net predicts noise (or clean image) at a given timestep, conditioned on
    the noisy image and class label.

    Classifier-Free Guidance Training:
        During training, class conditioning is randomly dropped with probability
        cond_drop_prob. The model learns both conditional p(x|c) and unconditional
        p(x) distributions in a single network.

    Classifier-Free Guidance Sampling:
        During sampling, predictions are guided by:
        output = unconditional + scale * (conditional - unconditional)
        where scale > 1 strengthens the conditioning effect.

    Args:
        dim: Base channel dimension
        num_classes: Number of class labels for conditioning
        cond_drop_prob: Probability of dropping class conditioning during training (default: 0.5)
        init_dim: Initial convolution output channels (default: dim)
        out_dim: Final output channels (default: channels or 2*channels if learned_variance)
        dim_mults: Channel multipliers for each resolution level (default: (1,2,4,8))
        channels: Number of input/output image channels (default: 3 for RGB)
        learned_variance: Whether to predict variance (default: False)
        learned_sinusoidal_cond: Use learned sinusoidal embeddings (default: False)
        random_fourier_features: Use random Fourier features for time (default: False)
        learned_sinusoidal_dim: Dimension of learned sinusoidal embeddings (default: 16)
        attn_dim_head: Dimension per attention head (default: 32)
        attn_heads: Number of attention heads (default: 4)
    """
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        # classifier free guidance stuff
        # This probability controls how often class conditioning is dropped during training
        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        # Initial convolution to project image to feature space
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # Calculate dimensions for each resolution level
        # E.g., if dim=64 and dim_mults=(1,2,4,8), dims=[init_dim, 64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # Create pairs of (input_dim, output_dim) for each level
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        # Time embeddings encode the diffusion timestep t to condition the network

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # Choose between standard sinusoidal embeddings or random Fourier features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # MLP to process time embeddings: positional_encoding -> Linear -> GELU -> Linear
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings
        # Class embeddings enable conditional generation based on class labels

        # Learnable embedding for each class
        self.classes_emb = nn.Embedding(num_classes, dim)
        # Learnable "null" embedding used when dropping conditioning (classifier-free guidance)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        # MLP to process class embeddings
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers
        # Build the U-Net encoder (downsampling) and decoder (upsampling) paths

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Build downsampling path (encoder)
        # Each level: ResBlock -> ResBlock -> Attention -> Downsample
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),  # Linear attention for efficiency
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        # Bottleneck (middle of U-Net at lowest resolution)
        # Uses full attention here since spatial dimensions are smallest
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        # Build upsampling path (decoder)
        # Each level: ResBlock -> ResBlock -> Attention -> Upsample
        # Note: ResBlocks have doubled input channels due to skip connections from encoder
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                # dim_out + dim_in because of concatenation with skip connection
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # Final output layers
        # If learned_variance=True, output 2*channels (mean and variance)
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        **kwargs
    ):
        """
        Forward pass with classifier-free guidance scaling.

        This method implements classifier-free guidance by computing both conditional
        and unconditional predictions, then interpolating between them.

        Classifier-Free Guidance Formula:
            output = conditional + scale * (conditional - unconditional)
                   = unconditional + scale * update

        When cond_scale > 1, the conditioning signal is amplified, leading to
        samples that more strongly match the class label.

        Args:
            *args: Positional arguments passed to forward()
            cond_scale: Guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)
            rescaled_phi: Rescaling interpolation factor to prevent saturation (0-1)
            remove_parallel_component: Whether to remove parallel component in advanced guidance
            keep_parallel_frac: Fraction of parallel component to keep (0-1)
            **kwargs: Keyword arguments passed to forward()

        Returns:
            tuple: (guided_output, unconditional_output) if rescaled_phi or cond_scale > 1
                   else just guided_output
        """
        # Get conditional prediction (with class conditioning)
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        # If no guidance, return conditional prediction directly
        if cond_scale == 1:
            return logits

        # Get unconditional prediction (without class conditioning)
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        # Compute the difference (guidance direction)
        update = logits - null_logits

        # Optional: Remove component parallel to logits (advanced guidance technique)
        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        # Apply guidance scaling
        scaled_logits = logits + update * (cond_scale - 1.)

        if rescaled_phi == 0.:
            return scaled_logits, null_logits

        # Optional: Rescale to match standard deviation of original prediction
        # This helps prevent over-saturation at high guidance scales
        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        # Interpolate between rescaled and non-rescaled versions
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits, null_logits

    def forward(
        self,
        x,
        time,
        classes,
        cond_drop_prob = None
    ):
        """
        Forward pass through the U-Net.

        Predicts noise (or clean image) given a noisy input, timestep, and class label.

        The forward pass:
        1. Processes class and time embeddings
        2. Applies random dropout to class conditioning (for classifier-free guidance training)
        3. Encodes input through downsampling path, storing skip connections
        4. Processes bottleneck with full attention
        5. Decodes through upsampling path, using skip connections
        6. Outputs prediction (noise, x0, or v depending on objective)

        Args:
            x: Noisy input image, shape (batch, channels, height, width)
            time: Timestep values, shape (batch,)
            classes: Class labels, shape (batch,)
            cond_drop_prob: Probability of dropping conditioning (None = use self.cond_drop_prob)

        Returns:
            Tensor: Predicted noise/image, shape (batch, out_dim, height, width)
        """
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance
        # This is the key mechanism enabling classifier-free guidance

        # Get class embeddings from embedding table
        classes_emb = self.classes_emb(classes)

        # Randomly replace class embeddings with null embedding during training
        if cond_drop_prob > 0:
            # Create mask: True = keep class, False = drop to null
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            # Replace dropped samples with null embedding
            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        # Project class embeddings to conditioning dimension
        c = self.classes_mlp(classes_emb)

        # unet forward pass

        # Initial convolution
        x = self.init_conv(x)
        r = x.clone()  # Save for final skip connection

        # Process time embedding
        t = self.time_mlp(time)

        h = []  # List to store skip connections from encoder

        # Encoder (downsampling path)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)  # Store skip connection

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)  # Store skip connection

            x = downsample(x)  # Reduce spatial resolution

        # Bottleneck
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # Decoder (upsampling path)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)  # Add skip connection
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)  # Add skip connection
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)  # Increase spatial resolution

        # Final layers with long skip connection from input
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """
    Extract values from tensor 'a' at indices 't' and reshape for broadcasting.

    This is a helper function for indexing pre-computed diffusion coefficients
    (like alphas, betas) at specific timesteps and reshaping them to broadcast
    with image tensors.

    Args:
        a: Tensor of precomputed values (e.g., alphas_cumprod), shape (timesteps,)
        t: Timestep indices, shape (batch,)
        x_shape: Shape of the tensor to broadcast to (e.g., (batch, channels, height, width))

    Returns:
        Tensor: Extracted and reshaped values, shape (batch, 1, 1, 1, ...)
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    # Reshape to (batch, 1, 1, 1) for broadcasting with images
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    Create a linear schedule for beta values (noise schedule).

    Beta controls the amount of noise added at each timestep.
    Linear schedule: linearly increase from beta_start to beta_end.

    Args:
        timesteps: Number of diffusion timesteps

    Returns:
        Tensor: Beta values, shape (timesteps,)
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    Create a cosine schedule for beta values (noise schedule).

    The cosine schedule provides more gradual noise addition at the beginning
    and end of the diffusion process, which often leads to better sample quality.

    Reference: "Improved Denoising Diffusion Probabilistic Models"
    https://openreview.net/forum?id=-NEXDKk8gZ

    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small (default: 0.008)

    Returns:
        Tensor: Beta values, shape (timesteps,), clipped to [0, 0.999]
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    # Compute alpha_cumprod using cosine function
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # Derive betas from alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process with Classifier-Free Guidance.

    Implements the denoising diffusion probabilistic model (DDPM) framework, which:
    1. Gradually adds Gaussian noise to images (forward process)
    2. Learns to reverse this process (reverse process/denoising)
    3. Generates new images by denoising from pure noise

    Mathematical Background:
    - Forward process: q(x_t | x_0) = N(x_t; sqrt(α_t)*x_0, (1-α_t)*I)
    - Reverse process: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
    - Training objective: Predict noise ε, clean image x_0, or velocity v

    Three prediction objectives supported:
    - pred_noise: Predict the noise ε added to the image (original DDPM)
    - pred_x0: Directly predict the clean image x_0
    - pred_v: Predict velocity v (used in progressive distillation, Imagen Video)

    DDIM Sampling:
    - Deterministic or semi-deterministic sampling with fewer steps
    - Can generate samples in 50-100 steps vs 1000 for DDPM
    - Controlled by sampling_timesteps and ddim_sampling_eta parameters

    Args:
        model: U-Net model for noise prediction
        image_size: Size of images (assumes square images)
        timesteps: Number of diffusion timesteps for training (default: 1000)
        sampling_timesteps: Number of timesteps for sampling (None = same as timesteps)
        objective: Prediction target - 'pred_noise', 'pred_x0', or 'pred_v' (default: 'pred_noise')
        beta_schedule: Noise schedule - 'linear' or 'cosine' (default: 'cosine')
        ddim_sampling_eta: DDIM stochasticity (0=deterministic, 1=stochastic DDPM) (default: 1.0)
        offset_noise_strength: Strength of offset noise for better color/brightness diversity (default: 0.0)
        min_snr_loss_weight: Use min-SNR loss weighting (default: False)
        min_snr_gamma: Gamma value for min-SNR weighting (default: 5)
        use_cfg_plus_plus: Use CFG++ variant (https://arxiv.org/pdf/2406.08070) (default: False)
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        use_cfg_plus_plus = False # https://arxiv.org/pdf/2406.08070
    ):
        super().__init__()
        # Verify model configuration
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        self.objective = objective

        # Validate objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # Create noise schedule (beta values for each timestep)
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # Calculate alpha values
        # alpha_t = 1 - beta_t
        alphas = 1. - betas
        # alpha_cumprod_t = product of all alphas up to timestep t
        # This is ᾱ_t in the DDPM paper
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # alpha_cumprod for previous timestep (padded with 1.0 at start)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # use cfg++ when ddim sampling
        # CFG++ is an improved variant of classifier-free guidance

        self.use_cfg_plus_plus = use_cfg_plus_plus

        # sampling related parameters
        # DDIM allows faster sampling by skipping timesteps

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        # Buffers are saved with the model but not trained

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # Precompute coefficients for the forward diffusion process

        # For sampling x_t from x_0: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # For predicting x_0 from x_t and noise
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # This is used for the reverse diffusion process

        # Variance of the posterior distribution
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        # Coefficients for computing the posterior mean
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal
        # Offset noise helps with generating images with varied brightness/color

        self.offset_noise_strength = offset_noise_strength

        # loss weight
        # SNR (Signal-to-Noise Ratio) weighting for better training

        # SNR(t) = ᾱ_t / (1 - ᾱ_t)
        snr = alphas_cumprod / (1 - alphas_cumprod)

        # Min-SNR weighting: clip SNR to prevent extreme values
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        # Loss weighting depends on prediction objective
        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        """Get the device of the model."""
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict clean image x_0 from noisy image x_t and predicted noise.

        Uses the formula: x_0 = (x_t - sqrt(1-ᾱ_t)*ε) / sqrt(ᾱ_t)

        Args:
            x_t: Noisy image at timestep t
            t: Timestep values
            noise: Predicted noise

        Returns:
            Tensor: Predicted clean image x_0
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        Predict noise from noisy image x_t and clean image x_0.

        Uses the formula: ε = (x_t - sqrt(ᾱ_t)*x_0) / sqrt(1-ᾱ_t)

        Args:
            x_t: Noisy image at timestep t
            t: Timestep values
            x0: Clean image

        Returns:
            Tensor: Predicted noise
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """
        Compute velocity v from clean image and noise.

        V-parameterization combines both x_0 and noise predictions.
        Formula: v = sqrt(ᾱ_t)*ε - sqrt(1-ᾱ_t)*x_0

        Used in progressive distillation and Imagen Video.

        Args:
            x_start: Clean image x_0
            t: Timestep values
            noise: Noise added to image

        Returns:
            Tensor: Velocity v
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict clean image x_0 from noisy image x_t and velocity v.

        Formula: x_0 = sqrt(ᾱ_t)*x_t - sqrt(1-ᾱ_t)*v

        Args:
            x_t: Noisy image at timestep t
            t: Timestep values
            v: Predicted velocity

        Returns:
            Tensor: Predicted clean image x_0
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute posterior distribution q(x_{t-1} | x_t, x_0).

        The posterior is the distribution of x_{t-1} given both the noisy
        image x_t and the clean image x_0. This is used in the reverse
        diffusion process.

        Args:
            x_start: Clean image x_0
            x_t: Noisy image at timestep t
            t: Timestep values

        Returns:
            tuple: (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
        """
        Get model predictions with classifier-free guidance.

        Calls the model with guidance scaling and converts the output to both
        predicted noise and predicted clean image, regardless of the objective.

        CFG++ Variant:
        When use_cfg_plus_plus=True, uses the unconditional prediction for noise
        prediction in DDIM sampling, which can improve sample quality.

        Args:
            x: Noisy image at timestep t
            t: Timestep values
            classes: Class labels
            cond_scale: Classifier-free guidance scale (default: 6.0)
            rescaled_phi: Rescaling factor for guidance (default: 0.7)
            clip_x_start: Whether to clip predicted x_0 to [-1, 1] (default: False)

        Returns:
            ModelPrediction: Named tuple containing (pred_noise, pred_x_start)
        """
        # Get guided prediction from model
        model_output, model_output_null = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        # Convert model output to both noise and x_0 predictions based on objective
        if self.objective == 'pred_noise':
            # Model predicts noise directly
            # For CFG++, use unconditional noise for DDIM sampling
            pred_noise = model_output if not self.use_cfg_plus_plus else model_output_null

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            # Model predicts clean image directly
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start if not self.use_cfg_plus_plus else maybe_clip(model_output_null)

            # Derive noise from predicted x_0
            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            # Model predicts velocity
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start
            if self.use_cfg_plus_plus:
                x_start_for_pred_noise = self.predict_start_from_v(x, t, model_output_null)
                x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            # Derive noise from predicted x_0
            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised = True):
        """
        Compute mean and variance for reverse diffusion step.

        Predicts the distribution p(x_{t-1} | x_t) using the model's predictions.

        Args:
            x: Noisy image at timestep t
            t: Timestep values
            classes: Class labels
            cond_scale: Classifier-free guidance scale
            rescaled_phi: Rescaling factor for guidance
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            tuple: (model_mean, posterior_variance, posterior_log_variance, x_start)
        """
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # Compute posterior q(x_{t-1} | x_t, x_0) using predicted x_0
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        """
        Sample x_{t-1} from x_t using the reverse diffusion process (DDPM sampling).

        This is a single denoising step in the sampling process.

        Args:
            x: Noisy image at timestep t, shape (batch, channels, height, width)
            t: Current timestep (integer)
            classes: Class labels
            cond_scale: Classifier-free guidance scale (default: 6.0)
            rescaled_phi: Rescaling factor for guidance (default: 0.7)
            clip_denoised: Whether to clip predictions (default: True)

        Returns:
            tuple: (predicted image at t-1, predicted clean image)
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised)
        # Add noise except at the final step (t=0)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        # Sample from the predicted distribution
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7):
        """
        Generate samples using DDPM sampling (full diffusion chain).

        Starts from pure noise and iteratively denoises for all timesteps.
        This is the standard DDPM sampling procedure.

        Args:
            classes: Class labels for conditional generation
            shape: Shape of images to generate (batch, channels, height, width)
            cond_scale: Classifier-free guidance scale (default: 6.0)
            rescaled_phi: Rescaling factor for guidance (default: 0.7)

        Returns:
            Tensor: Generated images, normalized to [0, 1]
        """
        batch, device = shape[0], self.betas.device

        # Start from pure Gaussian noise
        img = torch.randn(shape, device=device)

        x_start = None

        # Iteratively denoise from T to 0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        # Convert from [-1, 1] to [0, 1]
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        """
        Generate samples using DDIM (Denoising Diffusion Implicit Models) sampling.

        DDIM is a faster sampling method that can skip timesteps, enabling generation
        in 50-100 steps instead of 1000. It provides a deterministic sampling process
        when eta=0, or stochastic when eta=1 (equivalent to DDPM).

        The DDIM update rule:
            x_{t-1} = sqrt(α_{t-1}) * x_0 + sqrt(1 - α_{t-1} - σ²) * ε + σ * z
        where:
            - x_0 is the predicted clean image
            - ε is the predicted noise
            - σ controls stochasticity (σ = η * sqrt((1 - α_{t-1})/(1 - α_t)) * sqrt(1 - α_t/α_{t-1}))
            - z is random noise

        Args:
            classes: Class labels for conditional generation
            shape: Shape of images to generate (batch, channels, height, width)
            cond_scale: Classifier-free guidance scale (default: 6.0)
            rescaled_phi: Rescaling factor for guidance (default: 0.7)
            clip_denoised: Whether to clip predictions to [-1, 1] (default: True)

        Returns:
            Tensor: Generated images, normalized to [0, 1]
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # Create timestep schedule for sampling
        # E.g., if total=1000 and sampling=50, we sample at [999, 979, 959, ..., 19, 0, -1]
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Start from pure Gaussian noise
        img = torch.randn(shape, device = device)

        x_start = None

        # DDIM sampling loop
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # Get model predictions (both noise and x_0)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

            # At the final step, just use the predicted clean image
            if time_next < 0:
                img = x_start
                continue

            # Get alpha values for current and next timesteps
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # Compute variance for this step
            # eta=0: deterministic, eta=1: stochastic (DDPM-like)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            # Coefficient for predicted noise
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # Random noise (only used if eta > 0)
            noise = torch.randn_like(img)

            # DDIM update: combine predicted x_0, predicted noise, and random noise
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # Convert from [-1, 1] to [0, 1]
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale = 6., rescaled_phi = 0.7):
        """
        Generate images from noise.

        Automatically selects DDPM or DDIM sampling based on sampling_timesteps.

        Args:
            classes: Class labels for conditional generation
            cond_scale: Classifier-free guidance scale (default: 6.0)
                       Values > 1 strengthen conditioning, typically 3-8 works well
            rescaled_phi: Rescaling factor for guidance (default: 0.7)

        Returns:
            Tensor: Generated images, shape (batch, channels, image_size, image_size)
        """
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, channels, image_size, image_size), cond_scale, rescaled_phi)

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        """
        Interpolate between two images in latent space.

        Adds noise to both images, interpolates in the noisy space,
        then denoises the interpolated result.

        Args:
            x1: First image
            x2: Second image
            classes: Class labels
            t: Timestep to noise to (None = maximum noise)
            lam: Interpolation factor, 0 = x1, 1 = x2 (default: 0.5)

        Returns:
            Tensor: Interpolated image
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        # Noise both images to timestep t
        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        # Interpolate in noisy space
        img = (1 - lam) * xt1 + lam * xt2

        # Denoise from timestep t to 0
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img, _ = self.p_sample(img, i, classes)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process q(x_t | x_0).

        Adds noise to a clean image according to the diffusion schedule.

        Formula: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Args:
            x_start: Clean image x_0
            t: Timestep values
            noise: Noise to add (None = sample random noise)

        Returns:
            Tensor: Noisy image x_t
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Optional: Add offset noise for better color/brightness diversity
        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # Apply forward diffusion formula
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise = None):
        """
        Compute training loss for the diffusion model.

        The loss is MSE between the model's prediction and the target,
        weighted by the loss_weight which depends on the objective.

        Args:
            x_start: Clean images
            t: Timestep values
            classes: Class labels
            noise: Noise to add (None = sample random noise)

        Returns:
            Tensor: Scalar loss value
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        # Add noise to images according to timestep t
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        # Get model prediction
        model_out = self.model(x, t, classes)

        # Determine target based on prediction objective
        if self.objective == 'pred_noise':
            target = noise  # Predict the noise that was added
        elif self.objective == 'pred_x0':
            target = x_start  # Predict the original clean image
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v  # Predict the velocity
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # Compute MSE loss
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # Apply loss weighting (SNR-based weighting for better training)
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Training forward pass.

        Randomly samples a timestep, adds noise, and computes the loss.

        Args:
            img: Batch of images, shape (batch, channels, height, width)
            *args: Additional arguments passed to p_losses (e.g., classes)
            **kwargs: Additional keyword arguments passed to p_losses

        Returns:
            Tensor: Scalar loss value
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # Randomly sample timesteps for each image in the batch
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Normalize images to [-1, 1]
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

# example usage

if __name__ == '__main__':
    """
    Example demonstrating how to use the classifier-free guidance diffusion model.

    This example shows:
    1. Creating a U-Net model with class conditioning
    2. Wrapping it in a GaussianDiffusion object
    3. Training on images with class labels
    4. Generating new images with classifier-free guidance
    5. Interpolating between images
    """

    # Configuration
    num_classes = 10  # Number of class labels (e.g., 10 for CIFAR-10)

    # Create U-Net model with classifier-free guidance
    model = Unet(
        dim = 64,                    # Base channel dimension
        dim_mults = (1, 2, 4, 8),   # Channel multipliers for each resolution
        num_classes = num_classes,   # Number of classes for conditioning
        cond_drop_prob = 0.5         # Probability of dropping conditioning (50% for CFG)
    )

    # Wrap model in diffusion process
    diffusion = GaussianDiffusion(
        model,
        image_size = 128,    # Image resolution (128x128)
        timesteps = 1000     # Number of diffusion timesteps
    ).cuda()

    # Training example
    # NOTE: In practice, images should be in range [0, 1]
    training_images = torch.randn(8, 3, 128, 128).cuda() # images are normalized from 0 to 1
    image_classes = torch.randint(0, num_classes, (8,)).cuda()    # say 10 classes

    # Compute loss for training
    loss = diffusion(training_images, classes = image_classes)
    loss.backward()

    # do above for many steps (typical training loop)

    # Sampling / Generation
    # Generate new images conditioned on class labels
    sampled_images = diffusion.sample(
        classes = image_classes,
        cond_scale = 6.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    sampled_images.shape # (8, 3, 128, 128)

    # interpolation
    # Smoothly interpolate between two images in the latent space
    interpolate_out = diffusion.interpolate(
        training_images[:1],    # First image
        training_images[:1],    # Second image
        image_classes[:1]       # Class label
    )

