"""
Denoising Diffusion Probabilistic Models (DDPM) - 2D Image Implementation

This module implements the standard DDPM framework for generating 2D images through
iterative denoising. DDPMs learn to generate images by reversing a diffusion process
that gradually adds noise to data.

Key Components:
    - GaussianDiffusion: Core diffusion model implementing forward (noise addition)
      and reverse (denoising) processes
    - Unet: Neural network backbone for predicting noise at each timestep
    - Trainer: Training loop with support for EMA, checkpointing, and FID evaluation
    - Dataset: Image dataset loader with augmentation support

Diffusion Process:
    Forward process: q(x_t | x_{t-1}) gradually adds Gaussian noise to images
    Reverse process: p(x_{t-1} | x_t) learned by neural network to denoise

Training Objectives:
    - pred_noise: Predict the noise added at each timestep (original DDPM)
    - pred_x0: Directly predict the clean image (alternative formulation)
    - pred_v: Predict velocity/v-parameterization (used in Imagen-Video)

Sampling Methods:
    - DDPM sampling: Iterative denoising through all timesteps
    - DDIM sampling: Deterministic/faster sampling with fewer steps

References:
    - DDPM paper: https://arxiv.org/abs/2006.11239
    - Improved DDPM: https://arxiv.org/abs/2102.09672
    - DDIM: https://arxiv.org/abs/2010.02502
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
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# constants

# Named tuple to store model predictions during sampling
# pred_noise: predicted noise at timestep t
# pred_x_start: predicted clean image (x_0) reconstructed from noisy image
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    """Check if a value is not None."""
    return x is not None

def default(val, d):
    """
    Return val if it exists, otherwise return default value d.
    If d is callable, call it to get the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    """
    Convert a value to a tuple of specified length.
    If already a tuple, return as-is. Otherwise, repeat the value.
    """
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    """Check if numer is evenly divisible by denom."""
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    """Identity function that returns its input unchanged."""
    return t

def cycle(dl):
    """
    Infinitely cycle through a DataLoader.
    Used to continuously sample batches during training.
    """
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    """Check if a number has an integer square root."""
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    """
    Divide num into groups of size divisor.
    Returns a list of group sizes, with remainder in final group if needed.
    Example: num_to_groups(10, 3) -> [3, 3, 3, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    """Convert PIL image to specified type (e.g., 'L', 'RGB', 'RGBA')."""
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalize image from [0, 1] to [-1, 1] range.
    This is the standard normalization for diffusion models.
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize from [-1, 1] back to [0, 1] range.
    Used after sampling to get valid image pixel values.
    """
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    """
    Upsampling layer used in UNet decoder.
    Doubles spatial resolution using nearest neighbor interpolation,
    then applies convolution for feature refinement.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    """
    Downsampling layer used in UNet encoder.
    Reduces spatial resolution by factor of 2 using space-to-depth rearrangement
    (pixelshuffle inverse), then 1x1 conv to adjust channels.
    This is more stable than strided convolution.
    """
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.
    More efficient alternative to LayerNorm, normalizes using RMS instead of mean and variance.
    Commonly used in modern vision transformers and diffusion models.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # Normalize across channel dimension, then scale
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    """
    Sinusoidal positional embeddings for timestep encoding.
    Encodes timestep t into a continuous vector representation using sin/cos functions
    at different frequencies. This allows the model to learn temporal patterns.

    Based on "Attention is All You Need" positional encoding.
    """
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta  # Base for frequency calculation

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # Create exponentially decaying frequencies
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Multiply timestep by frequencies
        emb = x[:, None] * emb[None, :]
        # Concatenate sin and cos embeddings
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """
    Random or learned sinusoidal positional embeddings.

    Alternative to standard sinusoidal embeddings that can optionally learn
    the frequency weights instead of using fixed exponential decay.

    Reference: @crowsonkb's v-diffusion-jax implementation
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        # Random weights that can optionally be learned during training
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        # Generate frequencies using random/learned weights
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # Fourier features: concatenate sin and cos
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # Concatenate with original input
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    """
    Basic convolutional block used in ResNet-style architecture.
    Consists of: Conv -> Norm -> Activation -> Dropout
    Supports adaptive normalization via scale_shift conditioning (for time embeddings).
    """
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()  # Smooth activation function (Swish)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        # Adaptive normalization: modulate features based on timestep
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    """
    ResNet-style block with skip connection and time embedding conditioning.

    The time embedding is processed through an MLP and used to modulate
    the features via adaptive normalization (scale and shift).
    This allows the network to adapt its processing based on the diffusion timestep.

    Structure: Block -> Block -> Add residual connection
    """
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        # MLP to project time embeddings to scale and shift parameters
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)  # *2 for scale and shift
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        # Residual connection: project input if dimensions don't match
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # Process time embedding and split into scale and shift
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        # Apply first block with time conditioning
        h = self.block1(x, scale_shift = scale_shift)

        # Apply second block (no conditioning)
        h = self.block2(h)

        # Add residual connection
        return h + self.res_conv(x)

class LinearAttention(Module):
    """
    Linear attention mechanism with O(n) complexity instead of O(n²).

    Uses kernel trick to avoid computing full attention matrix:
    Instead of softmax(QK^T)V, computes Q(K^TV) by applying softmax
    to Q and K separately.

    Includes learnable memory key-value pairs for enhanced expressiveness.
    Used in lower resolution layers for efficiency.
    """
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        # Learnable memory keys and values (global context)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # Add memory key-value pairs
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        # Apply softmax to queries and keys separately (linear attention trick)
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        # Compute attention: K^T @ V first (linear complexity)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Then multiply by Q
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    """
    Standard multi-head self-attention mechanism.

    Computes full O(n²) attention matrix for precise spatial relationships.
    Supports optional Flash Attention for memory efficiency.

    Includes learnable memory key-value pairs that act as global context
    tokens available to all spatial positions.

    Used in higher resolution layers and bottleneck for maximum expressiveness.
    """
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)  # Handles attention computation (with optional Flash Attention)

        # Learnable memory tokens (global context)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        # Add memory key-value pairs to provide global context
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # Compute standard attention
        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    """
    U-Net architecture for noise prediction in diffusion models.

    The U-Net is an encoder-decoder architecture with skip connections that
    processes noisy images and timesteps to predict either noise, clean images,
    or velocity (depending on the objective).

    Architecture:
        - Encoder (downs): Progressively downsamples the image while increasing channels
        - Bottleneck (mid): Processes the lowest resolution representation
        - Decoder (ups): Progressively upsamples while merging skip connections from encoder

    Each resolution level contains:
        - 2 ResNet blocks with time embedding conditioning
        - 1 Attention block (Linear or Full attention)
        - 1 Downsampling/Upsampling layer (except last)

    Time Conditioning:
        Timestep t is embedded using sinusoidal positional encoding and processed
        through an MLP. This embedding modulates the ResNet blocks via adaptive
        normalization, allowing the network to adapt its denoising based on noise level.

    Skip Connections:
        Features from encoder are concatenated with decoder features at matching
        resolutions, providing fine-grained spatial information for reconstruction.

    Args:
        dim: Base channel dimension
        init_dim: Initial channel dimension after first conv (defaults to dim)
        out_dim: Output channels (defaults to input channels or 2x for learned variance)
        dim_mults: Channel multipliers for each resolution level (e.g., (1,2,4,8))
        channels: Input image channels (3 for RGB)
        self_condition: Whether to use self-conditioning (concatenate previous prediction)
        learned_variance: Whether to predict variance in addition to mean
        learned_sinusoidal_cond: Use learned sinusoidal time embeddings
        random_fourier_features: Use random Fourier features for time
        learned_sinusoidal_dim: Dimension for learned sinusoidal embeddings
        sinusoidal_pos_emb_theta: Base frequency for sinusoidal embeddings
        dropout: Dropout rate
        attn_dim_head: Attention head dimension
        attn_heads: Number of attention heads
        full_attn: Which layers use full attention (defaults to only bottleneck)
        flash_attn: Use Flash Attention for efficiency
    """
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        # If self-conditioning, concatenate previous prediction with input
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # Initial convolution with large kernel to capture local patterns
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # Calculate dimensions for each resolution level
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # Choose time embedding type
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        # MLP to process time embeddings: SinusoidalEmbed -> Linear -> GELU -> Linear
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        # By default, only use full attention at bottleneck (lowest resolution)
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        # Build encoder (downsampling path)
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            # Choose attention type: Full for high-detail layers, Linear for efficiency
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            # Each encoder stage: ResBlock -> ResBlock -> Attention -> Downsample
            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        # Bottleneck (middle of U-Net, lowest resolution, highest channels)
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # Build decoder (upsampling path) - process in reverse order
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            # Each decoder stage: ResBlock -> ResBlock -> Attention -> Upsample
            # Note: ResBlocks take concatenated features (skip connection + upsampled)
            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),  # +dim_in for skip connection
                resnet_block(dim_out + dim_in, dim_out),  # +dim_in for skip connection
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # Output projection
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)  # *2 for final skip connection
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        """Total downsampling factor of the U-Net encoder."""
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        """
        Forward pass through U-Net.

        Args:
            x: Noisy input image [batch, channels, height, width]
            time: Timestep indices [batch]
            x_self_cond: Optional self-conditioning input (previous prediction)

        Returns:
            Predicted output (noise, x0, or v depending on objective)
        """
        # Ensure input dimensions are compatible with downsampling
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        # Self-conditioning: concatenate previous prediction with input
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        # Initial convolution
        x = self.init_conv(x)
        r = x.clone()  # Save for final skip connection

        # Process timestep embedding
        t = self.time_mlp(time)

        h = []  # Store intermediate features for skip connections

        # Encoder: progressively downsample while storing skip connections
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)  # Save for skip connection

            x = block2(x, t)
            x = attn(x) + x  # Attention with residual connection
            h.append(x)  # Save for skip connection

            x = downsample(x)

        # Bottleneck: process at lowest resolution
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x  # Attention with residual
        x = self.mid_block2(x, t)

        # Decoder: progressively upsample while merging skip connections
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)  # Merge skip connection
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)  # Merge skip connection
            x = block2(x, t)
            x = attn(x) + x  # Attention with residual

            x = upsample(x)

        # Final layers: merge with initial features and project to output
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """
    Extract values from array 'a' at indices 't' and reshape for broadcasting.

    This is a helper function to index into arrays like betas, alphas_cumprod, etc.
    based on timestep t, and reshape the output to broadcast with tensors of shape x_shape.

    Args:
        a: 1D array of values (e.g., betas, alphas_cumprod)
        t: Timestep indices [batch]
        x_shape: Shape of tensor to broadcast with

    Returns:
        Values from 'a' at indices 't', reshaped to [batch, 1, 1, ...]
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    Linear noise schedule - proposed in original DDPM paper.

    Linearly increases beta (noise variance) from ~0.0001 to ~0.02.
    Works well for images up to 64x64 resolution.

    Reference: https://arxiv.org/abs/2006.11239
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    Cosine noise schedule - proposed in Improved DDPM.

    Uses cosine function to create smoother noise schedule with less noise
    at beginning and end. Often produces better sample quality than linear schedule.

    Reference: https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    Sigmoid noise schedule.

    Uses sigmoid function for smoother transitions. Generally works better
    for higher resolution images (>64x64) compared to linear schedule.

    Reference: https://arxiv.org/abs/2212.11972 - Figure 8
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    """
    Gaussian Diffusion Model - implements the forward and reverse diffusion processes.

    This class handles:
    1. Forward diffusion: q(x_t | x_0) - gradually adds noise to clean images
    2. Reverse diffusion: p(x_{t-1} | x_t) - learned denoising process
    3. Training: computes loss for learning the reverse process
    4. Sampling: generates images from noise using learned reverse process

    The forward process is fixed (Gaussian noise addition), while the reverse
    process is learned by training a neural network (typically U-Net) to predict
    either the noise, the clean image, or a velocity parameterization.

    Key Concepts:
        - Beta schedule: controls how much noise is added at each timestep
        - Alpha: 1 - beta, the signal retention rate
        - Alpha_cumprod: cumulative product of alphas, determines total noise at timestep t
        - SNR: Signal-to-Noise Ratio, used for loss weighting

    Args:
        model: Neural network (U-Net) for predicting noise/x0/v
        image_size: Target image resolution (int or tuple)
        timesteps: Number of diffusion steps (default 1000)
        sampling_timesteps: Number of steps during sampling (can be < timesteps for DDIM)
        objective: What the model predicts ('pred_noise', 'pred_x0', or 'pred_v')
        beta_schedule: Noise schedule type ('linear', 'cosine', or 'sigmoid')
        schedule_fn_kwargs: Additional arguments for beta schedule function
        ddim_sampling_eta: Stochasticity parameter for DDIM (0 = deterministic)
        auto_normalize: Automatically normalize images to [-1, 1]
        offset_noise_strength: Strength of offset noise (for better dark/light generation)
        min_snr_loss_weight: Use min-SNR loss weighting (improves training stability)
        min_snr_gamma: Clipping value for min-SNR weighting
        immiscible: Use immiscible diffusion (experimental)
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # Select and initialize noise schedule
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        # Compute alpha values
        # alpha_t = 1 - beta_t (signal retention at step t)
        # alpha_cumprod_t = prod(alpha_i for i in 1..t) (cumulative signal retention)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        # For DDIM sampling, we can use fewer steps than training
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # These are precomputed coefficients for the forward diffusion process

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # Used during reverse diffusion to compute distribution of x_{t-1} given x_t and x_0

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion - experimental technique for better mode coverage

        self.immiscible = immiscible

        # offset noise strength - helps generate very dark/bright regions
        # Reference: https://www.crosslabs.org/blog/diffusion-with-offset-noise

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight based on Signal-to-Noise Ratio (SNR)
        # This reweights the loss at different timesteps for better training

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # Min-SNR loss weighting: https://arxiv.org/abs/2303.09556
        # Prevents the model from over-optimizing noisy timesteps

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        # Different loss weights for different objectives
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 (clean image) from x_t (noisy image) and predicted noise.
        Uses the reparameterization: x_0 = (x_t - sqrt(1-α̅_t) * noise) / sqrt(α̅_t)
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        Predict noise from x_t (noisy image) and x_0 (clean image).
        Inverse of predict_start_from_noise.
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """
        Compute v-parameterization target: v = sqrt(α̅_t) * noise - sqrt(1-α̅_t) * x_0
        V-parameterization is an alternative training objective that can improve stability.
        Reference: Progressive Distillation paper, used successfully in Imagen-Video
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict x_0 from x_t and v-parameterization.
        Inverse of predict_v.
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the posterior distribution q(x_{t-1} | x_t, x_0).

        Given the noisy image at timestep t and the predicted clean image,
        compute the mean and variance of the distribution for x_{t-1}.
        This is used during sampling to denoise step-by-step.

        Returns:
            posterior_mean: Mean of q(x_{t-1} | x_t, x_0)
            posterior_variance: Variance of the posterior
            posterior_log_variance_clipped: Log variance (clipped for numerical stability)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        """
        Get model predictions and convert to both noise and x_start predictions.

        This handles different prediction objectives (noise, x0, or v) and converts
        them to a standard format containing both pred_noise and pred_x_start.

        Args:
            x: Noisy image at timestep t
            t: Timestep
            x_self_cond: Optional self-conditioning from previous prediction
            clip_x_start: Clip predicted x_0 to [-1, 1]
            rederive_pred_noise: Recompute noise from clipped x_start (for DDIM)

        Returns:
            ModelPrediction with pred_noise and pred_x_start
        """
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        """
        Compute mean and variance for the reverse diffusion step p(x_{t-1} | x_t).

        Uses the model to predict x_0, then computes the posterior distribution
        for x_{t-1} conditioned on x_t and the predicted x_0.

        Args:
            x: Current noisy image
            t: Current timestep
            x_self_cond: Optional self-conditioning
            clip_denoised: Clip predicted x_0 to valid range

        Returns:
            model_mean, posterior_variance, posterior_log_variance, x_start
        """
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        """
        Single denoising step: sample x_{t-1} from p(x_{t-1} | x_t).

        This is the core reverse diffusion step. Given a noisy image at timestep t,
        predict the mean and variance, then sample from a Gaussian to get x_{t-1}.

        Args:
            x: Noisy image at timestep t
            t: Current timestep (scalar)
            x_self_cond: Optional self-conditioning from previous prediction

        Returns:
            pred_img: Sampled x_{t-1}
            x_start: Predicted clean image x_0
        """
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        """
        DDPM sampling loop - generates images from pure noise.

        Iteratively denoises starting from random Gaussian noise,
        going through all timesteps from T to 0.

        Args:
            shape: Shape of images to generate [batch, channels, height, width]
            return_all_timesteps: If True, return all intermediate steps

        Returns:
            Generated images (or all intermediate steps if return_all_timesteps=True)
        """
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        """
        DDIM sampling - faster deterministic/semi-deterministic sampling.

        DDIM allows sampling with fewer steps than training by using a
        non-Markovian process. Can be deterministic (eta=0) or add noise (eta>0).

        Reference: Denoising Diffusion Implicit Models (DDIM) - https://arxiv.org/abs/2010.02502

        Args:
            shape: Shape of images to generate
            return_all_timesteps: If True, return all intermediate steps

        Returns:
            Generated images
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # Create subsequence of timesteps for faster sampling
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # DDIM sampling formula
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            # Reconstruct x_{t-1} from x_0 prediction and noise
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        """
        Generate images from noise.
        Automatically uses DDPM or DDIM sampling based on configuration.
        """
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        """
        Interpolate between two images in latent space.

        Adds noise to both images to a specific timestep, interpolates,
        then denoises back to get a blended result.

        Args:
            x1, x2: Images to interpolate between
            t: Timestep to noise to (higher = more blending)
            lam: Interpolation weight (0=x1, 1=x2)

        Returns:
            Interpolated image
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        # Add noise to both images
        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        # Interpolate in noisy space
        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        # Denoise back to clean image
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        """
        Immiscible diffusion: assign noise to images using optimal transport.

        Instead of random noise pairing, finds optimal assignment between
        clean images and noise samples to potentially improve training.

        Experimental feature - not commonly used.
        """
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        """
        Forward diffusion: add noise to clean images.

        Implements q(x_t | x_0) = N(x_t; sqrt(α̅_t) * x_0, (1 - α̅_t) * I)

        This is the closed-form solution that allows jumping directly to
        any timestep t without iterating through all previous steps.

        Args:
            x_start: Clean images
            t: Timesteps to noise to
            noise: Optional pre-generated noise

        Returns:
            Noisy images at timestep t
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        """
        Compute training loss for the diffusion model.

        Training procedure:
        1. Sample random timesteps
        2. Add noise to clean images (forward diffusion)
        3. Predict noise/x0/v using the model
        4. Compute MSE loss against target
        5. Apply loss weighting based on timestep

        Args:
            x_start: Clean images
            t: Sampled timesteps
            noise: Optional pre-generated noise
            offset_noise_strength: Strength of offset noise augmentation

        Returns:
            Weighted MSE loss
        """
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - helps model learn to generate very dark/bright regions
        # Reference: https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample - forward diffusion

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # Self-conditioning: 50% of the time, condition on previous prediction
        # This technique slows training by ~25% but can significantly improve FID
        # Reference: https://arxiv.org/abs/2208.04202

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        # Set target based on objective
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # Compute loss
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # Apply timestep-dependent loss weighting (based on SNR)
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Training forward pass.

        Randomly samples timesteps and computes loss for a batch of images.

        Args:
            img: Batch of clean images

        Returns:
            Loss value
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    """
    Image dataset for training diffusion models.

    Loads images from a folder, applies transformations (resize, crop, flip),
    and converts to tensors in [0, 1] range.

    Args:
        folder: Path to folder containing images
        image_size: Target size for images (int or tuple)
        exts: List of image file extensions to load
        augment_horizontal_flip: Whether to randomly flip images horizontally
        convert_image_to: PIL image mode to convert to ('L', 'RGB', 'RGBA', etc.)
    """
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # Recursively find all image files with specified extensions
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # Image preprocessing pipeline
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),  # Convert image mode if needed
            T.Resize(image_size),  # Resize to target size
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),  # Data augmentation
            T.CenterCrop(image_size),  # Center crop to exact size
            T.ToTensor()  # Convert to tensor [0, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer:
    """
    Trainer for Denoising Diffusion Probabilistic Models.

    Handles the complete training loop including:
    - Multi-GPU training via Hugging Face Accelerate
    - Exponential Moving Average (EMA) of model weights
    - Gradient accumulation for large effective batch sizes
    - Automatic mixed precision (AMP) training
    - Periodic sampling and checkpointing
    - FID score evaluation
    - Best model selection

    Key Features:
        - EMA: Maintains running average of model parameters for better sample quality
        - Gradient Accumulation: Simulates larger batch sizes on limited memory
        - AMP: Mixed precision training for faster training and lower memory usage
        - Multi-GPU: Distributed training across multiple GPUs
        - FID Evaluation: Computes Fréchet Inception Distance for quality assessment

    Args:
        diffusion_model: GaussianDiffusion model to train
        folder: Path to folder containing training images
        train_batch_size: Batch size per GPU
        gradient_accumulate_every: Number of gradient accumulation steps
        augment_horizontal_flip: Apply random horizontal flip augmentation
        train_lr: Learning rate for Adam optimizer
        train_num_steps: Total number of training steps
        ema_update_every: Update EMA every N steps
        ema_decay: EMA decay rate (higher = slower update)
        adam_betas: Beta parameters for Adam optimizer
        save_and_sample_every: Save checkpoint and generate samples every N steps
        num_samples: Number of samples to generate (must have integer square root)
        results_folder: Directory to save checkpoints and samples
        amp: Enable automatic mixed precision training
        mixed_precision_type: Type of mixed precision ('fp16' or 'bf16')
        split_batches: Whether to split batches across GPUs (affects effective batch size)
        convert_image_to: PIL image mode to convert to
        calculate_fid: Whether to compute FID scores during training
        inception_block_idx: Inception block to use for FID (2048 is standard)
        max_grad_norm: Maximum gradient norm for clipping
        num_fid_samples: Number of samples to generate for FID computation
        save_best_and_latest_only: Only keep best and latest checkpoints (saves disk space)
    """
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator - handles multi-GPU, mixed precision, and distributed training

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # Effective batch size = train_batch_size * gradient_accumulate_every * num_gpus
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # Prepare dataloader for distributed training
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)  # Infinite iterator over dataloader

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # EMA - exponential moving average for better sample quality
        # Only maintained on main process to save memory

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        # This wraps them for distributed training, mixed precision, etc.

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation - measures sample quality
        # FID = Fréchet Inception Distance between generated and real images

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        """Get the device being used for training (CPU or CUDA)."""
        return self.accelerator.device

    def save(self, milestone):
        """
        Save model checkpoint.

        Saves model weights, optimizer state, EMA state, and training step.
        Only saves on the main process in distributed training.

        Args:
            milestone: Checkpoint identifier (usually step number or 'best'/'latest')
        """
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        """
        Load model checkpoint.

        Restores model weights, optimizer state, EMA state, and training step
        from a saved checkpoint.

        Args:
            milestone: Checkpoint identifier to load
        """
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        # Unwrap model from accelerator wrapper before loading
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # Restore gradient scaler state for mixed precision training
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        """
        Main training loop.

        Training procedure:
        1. Load batch of images
        2. Forward pass through diffusion model (add noise + predict)
        3. Compute loss
        4. Backward pass and gradient accumulation
        5. Update model parameters
        6. Update EMA model
        7. Periodically: generate samples, save checkpoint, compute FID

        The loop continues for train_num_steps iterations.

        Key features:
            - Gradient accumulation for effective larger batch sizes
            - Gradient clipping for training stability
            - EMA updates for better sample quality
            - Periodic sampling to visualize progress
            - FID score computation for quantitative evaluation
            - Checkpoint saving (all or best+latest only)
        """
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                # Gradient accumulation loop
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    # Forward pass with automatic mixed precision
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        # Scale loss by accumulation steps
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # Backward pass (accumulates gradients)
                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                # Synchronize across all GPUs before gradient clipping
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update parameters using accumulated gradients
                self.opt.step()
                self.opt.zero_grad()

                # Synchronize again after optimizer step
                accelerator.wait_for_everyone()

                self.step += 1

                # Update EMA model (only on main process)
                if accelerator.is_main_process:
                    self.ema.update()

                    # Periodic sampling and checkpointing
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        # Generate samples using EMA model
                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        # Save sample grid as image
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # Compute FID score for quantitative evaluation

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        # Save checkpoint (either all or best+latest)
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
