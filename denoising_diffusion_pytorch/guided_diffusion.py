"""
Guided Diffusion for Conditional Image Generation

This module implements classifier-guided diffusion models, which enable conditional image
generation by incorporating gradient guidance from an external classifier during the sampling process.

Key Concepts:
-------------
1. **Guided Diffusion Overview**:
   Guided diffusion extends standard diffusion models by steering the generation process toward
   desired attributes (e.g., generating images of a specific class). This is achieved by modifying
   the sampling process using gradients from a classifier that predicts the desired condition.

2. **How It Works**:
   - During training: The diffusion model is trained normally to denoise images
   - During sampling: At each denoising step, we:
     a) Predict the mean and variance for the next step
     b) Compute the classifier's gradient ∇_x log p(y|x_t) where y is the target class
     c) Adjust the predicted mean by adding variance * gradient
     d) This shifts the generation toward samples that the classifier recognizes as class y

3. **Benefits**:
   - **Conditional Generation**: Generate images with specific attributes without retraining
   - **Flexibility**: Use any pre-trained classifier to guide generation
   - **Control**: Adjust guidance strength via classifier_scale parameter
   - **Quality**: Often produces higher quality samples for desired conditions
   - **Composability**: Can combine multiple classifiers for multi-condition guidance

4. **Mathematical Foundation**:
   The guidance is based on Bayes' rule:
   p(x_t|y) ∝ p(x_t) * p(y|x_t)

   Taking the log and gradient:
   ∇_x log p(x_t|y) = ∇_x log p(x_t) + ∇_x log p(y|x_t)

   The second term is the classifier gradient, which steers generation toward class y.

5. **Implementation Details**:
   - The condition_mean() method implements gradient-based guidance
   - Guidance is applied in p_sample() during the denoising loop
   - The guidance_kwargs dict passes classifier and target class information
   - classifier_scale controls the strength of guidance (higher = stronger conditioning)

References:
-----------
- "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, 2021)
- Original implementation: https://github.com/openai/guided-diffusion
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
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

# from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    """
    Check if a value is not None.

    Args:
        x: Any value to check

    Returns:
        bool: True if x is not None, False otherwise
    """
    return x is not None

def default(val, d):
    """
    Return val if it exists, otherwise return default value d.

    Args:
        val: The value to check
        d: Default value or callable that returns default value

    Returns:
        val if it exists, otherwise d() if d is callable, else d
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    """
    Identity function that returns its input unchanged.

    Args:
        t: Input tensor or value
        *args: Additional positional arguments (ignored)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        The input t unchanged
    """
    return t

def cycle(dl):
    """
    Infinite iterator that cycles through a DataLoader.

    Args:
        dl: PyTorch DataLoader to cycle through

    Yields:
        Data batches from the DataLoader, cycling infinitely
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
    Divide a number into groups of size divisor, with a remainder group if needed.

    Args:
        num: Total number to divide
        divisor: Size of each group

    Returns:
        list: List of group sizes (e.g., [divisor, divisor, ..., remainder])
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    """
    Convert PIL image to specified type if needed.

    Args:
        img_type: Target image type (e.g., 'RGB', 'L')
        image: PIL Image to convert

    Returns:
        PIL Image in the specified format
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalize image from [0, 1] to [-1, 1] range.

    Args:
        img: Image tensor with values in [0, 1]

    Returns:
        Normalized image tensor with values in [-1, 1]
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize tensor from [-1, 1] to [0, 1] range.

    Args:
        t: Tensor with values in [-1, 1]

    Returns:
        Unnormalized tensor with values in [0, 1]
    """
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    """
    Residual wrapper that adds input to output of a function.

    This implements a skip connection: output = fn(x) + x
    """
    def __init__(self, fn):
        """
        Args:
            fn: The function/module to wrap with residual connection
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor
            *args: Additional positional arguments for fn
            **kwargs: Additional keyword arguments for fn

        Returns:
            fn(x) + x (residual connection)
        """
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    """
    Create 2x upsampling module using nearest neighbor interpolation + convolution.

    Args:
        dim: Input channel dimension
        dim_out: Output channel dimension (defaults to dim if None)

    Returns:
        nn.Sequential module that upsamples spatial dimensions by 2x
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    """
    Create 2x downsampling module using space-to-depth rearrangement + convolution.

    This reduces spatial dimensions by 2x while increasing channels by 4x,
    then applies 1x1 conv to get desired output channels.

    Args:
        dim: Input channel dimension
        dim_out: Output channel dimension (defaults to dim if None)

    Returns:
        nn.Sequential module that downsamples spatial dimensions by 2x
    """
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes using the root mean square statistic rather than mean and variance,
    which is more efficient while maintaining similar performance.
    """
    def __init__(self, dim):
        """
        Args:
            dim: Number of feature channels
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Normalized tensor scaled by learnable parameter g
        """
        return F.normalize(x, dim = 1) * self.g * (x.shape[-1] ** 0.5)

class PreNorm(nn.Module):
    """
    Wrapper that applies normalization before a function.

    This implements the Pre-LN (Pre-Layer Normalization) pattern.
    """
    def __init__(self, dim, fn):
        """
        Args:
            dim: Feature dimension for normalization
            fn: Function/module to apply after normalization
        """
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """
        Apply normalization then function.

        Args:
            x: Input tensor

        Returns:
            fn(norm(x))
        """
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for encoding timesteps.

    Uses sine and cosine functions of different frequencies to create
    unique embeddings for each timestep, similar to the Transformer positional encoding.
    """
    def __init__(self, dim):
        """
        Args:
            dim: Dimension of the positional embedding
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Create sinusoidal embeddings for timesteps.

        Args:
            x: Timestep tensor of shape [B]

        Returns:
            Positional embeddings of shape [B, dim]
        """
        device = x.device
        half_dim = self.dim // 2
        # Create exponentially spaced frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Apply timesteps to frequencies
        emb = x[:, None] * emb[None, :]
        # Concatenate sin and cos embeddings
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Random or learned sinusoidal positional embeddings.

    Following @crowsonkb's lead with random (optionally learned) sinusoidal pos emb.
    Reference: https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self, dim, is_random = False):
        """
        Args:
            dim: Dimension of the positional embedding (must be even)
            is_random: If True, frequencies are random and fixed; if False, they are learnable
        """
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        # Create random frequencies; learnable if is_random=False
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        """
        Create random/learned sinusoidal embeddings.

        Args:
            x: Timestep tensor of shape [B]

        Returns:
            Positional embeddings of shape [B, dim+1] (includes original timestep)
        """
        x = rearrange(x, 'b -> b 1')
        # Apply timesteps to random/learned frequencies
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # Create Fourier features with sin and cos
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # Concatenate original timestep with Fourier features
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    """
    Basic convolutional block with normalization and activation.

    Consists of: Conv -> Norm -> (optional scale/shift) -> Activation
    """
    def __init__(self, dim, dim_out):
        """
        Args:
            dim: Input channel dimension
            dim_out: Output channel dimension
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        """
        Forward pass with optional scale and shift modulation.

        Args:
            x: Input tensor of shape [B, C, H, W]
            scale_shift: Optional tuple of (scale, shift) tensors for adaptive normalization

        Returns:
            Processed tensor of shape [B, dim_out, H, W]
        """
        x = self.proj(x)
        x = self.norm(x)

        # Apply adaptive normalization if scale_shift is provided
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """
    Residual block with time embedding conditioning.

    This block applies two convolutional blocks with a residual connection,
    and optionally modulates features based on time embeddings using adaptive
    normalization (similar to AdaIN/FiLM).
    """
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        """
        Args:
            dim: Input channel dimension
            dim_out: Output channel dimension
            time_emb_dim: Dimension of time embeddings for conditioning (optional)
        """
        super().__init__()
        # MLP to process time embeddings into scale and shift parameters
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        # Residual connection that matches dimensions if needed
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        """
        Forward pass with optional time conditioning.

        Args:
            x: Input tensor of shape [B, dim, H, W]
            time_emb: Time embedding tensor of shape [B, time_emb_dim] (optional)

        Returns:
            Output tensor of shape [B, dim_out, H, W] with residual connection
        """
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # Process time embedding to get scale and shift parameters
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # Split into scale and shift for adaptive normalization
            scale_shift = time_emb.chunk(2, dim = 1)

        # Apply first block with time conditioning
        h = self.block1(x, scale_shift = scale_shift)

        # Apply second block
        h = self.block2(h)

        # Add residual connection
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """
    Linear-complexity attention mechanism.

    This is an efficient approximation to standard attention with O(n) complexity
    instead of O(n^2), achieved by applying softmax to keys and queries separately
    before computing attention.
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        """
        Args:
            dim: Input feature dimension
            heads: Number of attention heads
            dim_head: Dimension per attention head
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        Apply linear attention.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        b, c, h, w = x.shape
        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # Apply softmax to queries and keys separately (key to linear attention)
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        # Compute attention in linear complexity: first k*v, then q*(k*v)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    """
    Standard multi-head self-attention mechanism.

    Implements the classic scaled dot-product attention from "Attention is All You Need"
    adapted for 2D spatial features in diffusion models.
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        """
        Args:
            dim: Input feature dimension
            heads: Number of attention heads
            dim_head: Dimension per attention head
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        b, c, h, w = x.shape
        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # Scale queries
        q = q * self.scale

        # Compute attention scores (scaled dot-product)
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        # Apply attention to values
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    """
    U-Net architecture for diffusion models.

    This is the core denoising network that predicts noise/clean images given a noisy image
    and timestep. It uses a U-Net architecture with:
    - Downsampling path that progressively reduces spatial dimensions
    - Bottleneck with attention
    - Upsampling path that progressively increases spatial dimensions
    - Skip connections between corresponding down/up layers
    - Time embedding conditioning at each residual block

    The U-Net can predict different targets based on the objective:
    - pred_noise: Predict the noise added to the image
    - pred_x0: Predict the original clean image
    - pred_v: Predict the velocity parameterization
    """
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        """
        Args:
            dim: Base channel dimension
            init_dim: Initial channel dimension after first conv (defaults to dim)
            out_dim: Output channel dimension (defaults to channels or channels*2 if learned_variance)
            dim_mults: Channel multipliers for each resolution level (e.g., (1,2,4,8))
            channels: Number of input/output image channels
            self_condition: Whether to use self-conditioning (concatenate previous prediction)
            learned_variance: Whether to learn variance (doubles output channels)
            learned_sinusoidal_cond: Use learned sinusoidal time embeddings
            random_fourier_features: Use random Fourier features for time embeddings
            learned_sinusoidal_dim: Dimension for learned sinusoidal embeddings
        """
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        # Double input channels if using self-conditioning
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
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
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # MLP to process time embeddings
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Build downsampling layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        # Bottleneck with attention
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        # Build upsampling layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                # Input is concatenated with skip connection, hence dim_out + dim_in
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # Output projection
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        """
        Forward pass through U-Net.

        Args:
            x: Noisy image tensor of shape [B, channels, H, W]
            time: Timestep tensor of shape [B]
            x_self_cond: Previous prediction for self-conditioning [B, channels, H, W] (optional)

        Returns:
            Predicted noise/x0/v of shape [B, out_dim, H, W]
        """
        # Handle self-conditioning by concatenating previous prediction
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        # Initial convolution
        x = self.init_conv(x)
        r = x.clone()  # Save for final skip connection

        # Process time embeddings
        t = self.time_mlp(time)

        h = []  # List to store skip connections

        # Downsampling path
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)  # Save for skip connection

            x = block2(x, t)
            x = attn(x)
            h.append(x)  # Save for skip connection

            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling path with skip connections
        for block1, block2, attn, upsample in self.ups:
            # Concatenate with skip connection from downsampling path
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            # Concatenate with another skip connection
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        # Final skip connection from initial conv
        x = torch.cat((x, r), dim = 1)

        # Final processing
        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """
    Extract values from array a at indices t and reshape for broadcasting.

    This is a utility function to extract coefficients (like alphas, betas) at specific
    timesteps and reshape them to broadcast with image tensors.

    Args:
        a: 1D tensor of coefficients indexed by timestep
        t: Batch of timestep indices of shape [B]
        x_shape: Shape of the tensor to broadcast with (e.g., [B, C, H, W])

    Returns:
        Extracted values reshaped to [B, 1, 1, 1] for broadcasting
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
    


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process with Classifier Guidance Support.

    This class implements the Denoising Diffusion Probabilistic Model (DDPM) with support
    for classifier-guided sampling. It handles:

    1. **Forward Process (Training)**:
       - Gradually adds Gaussian noise to images over T timesteps
       - q(x_t | x_0) = N(x_t; sqrt(α_t)x_0, (1-α_t)I)

    2. **Reverse Process (Sampling)**:
       - Learns to denoise images step by step
       - p_θ(x_{t-1} | x_t) starts from pure noise and denoises to clean image

    3. **Classifier Guidance** (Key Feature):
       - During sampling, uses gradients from a classifier to guide generation
       - Modifies the mean at each step: μ' = μ + σ²∇_x log p(y|x)
       - Enables conditional generation without retraining the diffusion model
       - The gradient ∇_x log p(y|x) pushes samples toward desired class y

    The guidance mechanism in condition_mean() is the core innovation that enables
    classifier-guided diffusion, allowing flexible conditional generation.

    Supported Objectives:
    - pred_noise: Predict noise ε added to image (original DDPM)
    - pred_x0: Predict original clean image directly
    - pred_v: Predict velocity parameterization (progressive distillation)
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        """
        Args:
            model: U-Net denoising model
            image_size: Size of square images (H = W = image_size)
            timesteps: Number of diffusion steps T
            sampling_timesteps: Number of steps for DDIM sampling (defaults to timesteps)
            objective: Prediction target ('pred_noise', 'pred_x0', or 'pred_v')
            beta_schedule: Noise schedule ('linear', 'cosine', or 'sigmoid')
            schedule_fn_kwargs: Additional arguments for noise schedule
            ddim_sampling_eta: DDIM sampling parameter (0 = deterministic, 1 = DDPM)
            auto_normalize: Whether to auto-normalize images to [-1, 1]
            min_snr_loss_weight: Whether to use min-SNR loss weighting
            min_snr_gamma: Gamma parameter for min-SNR weighting
        """
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # Select noise schedule function
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # Generate noise schedule: β_t controls amount of noise at each timestep
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        # Calculate α_t = 1 - β_t and cumulative products ᾱ_t = ∏(1 to t) α_i
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        # DDIM allows sampling with fewer steps than training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Pre-compute values for forward diffusion q(x_t | x_0)
        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        # This posterior is used during the reverse process

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # Clipping prevents numerical issues at t=0 where variance is 0
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Compute loss weights based on SNR (Signal-to-Noise Ratio)

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            # Min-SNR weighting prevents over-emphasis on noisy timesteps
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        # Loss weighting depends on prediction objective
        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # Auto-normalization: convert images from [0, 1] to [-1, 1] for training

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 (original image) from x_t and predicted noise.

        Uses the formula: x_0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)

        Args:
            x_t: Noisy image at timestep t
            t: Timestep
            noise: Predicted noise

        Returns:
            Predicted original image x_0
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        Predict noise from x_t and x_0.

        Inverse of predict_start_from_noise.

        Args:
            x_t: Noisy image at timestep t
            t: Timestep
            x0: Original clean image

        Returns:
            Predicted noise that was added
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """
        Predict velocity parameterization v.

        v = sqrt(ᾱ_t) * ε - sqrt(1-ᾱ_t) * x_0

        Args:
            x_start: Original clean image
            t: Timestep
            noise: Noise

        Returns:
            Velocity v
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict x_0 from x_t and velocity v.

        Args:
            x_t: Noisy image at timestep t
            t: Timestep
            v: Velocity parameterization

        Returns:
            Predicted original image x_0
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute posterior distribution q(x_{t-1} | x_t, x_0).

        This is the true posterior in the forward process when x_0 is known.

        Args:
            x_start: Original clean image x_0
            x_t: Noisy image at timestep t
            t: Timestep

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        """
        Get model predictions and convert to both noise and x_start.

        Handles different prediction objectives and converts to a common format.

        Args:
            x: Current noisy image
            t: Timestep
            x_self_cond: Self-conditioning input (optional)
            clip_x_start: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            ModelPrediction namedtuple with (pred_noise, pred_x_start)
        """
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

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
        Compute mean and variance for reverse process step p(x_{t-1} | x_t).

        Args:
            x: Current noisy image at timestep t
            t: Timestep
            x_self_cond: Self-conditioning input (optional)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            Tuple of (model_mean, posterior_variance, posterior_log_variance, x_start)
        """
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
     
    def condition_mean(self, cond_fn, mean, variance, x, t, guidance_kwargs=None):
        """
        **CORE CLASSIFIER GUIDANCE METHOD**

        Adjust the predicted mean using classifier gradients to guide generation toward
        desired conditions. This is the heart of classifier-guided diffusion.

        Mathematical Background:
        -----------------------
        Standard sampling: p(x_{t-1} | x_t) = N(μ_θ(x_t), Σ_t)
        Guided sampling: p(x_{t-1} | x_t, y) ∝ p(x_{t-1} | x_t) * p(y | x_{t-1})

        Using Bayes' rule and gradient-based approximation:
        ∇_x log p(x_t | y) ≈ ∇_x log p(x_t) + ∇_x log p(y | x_t)

        The guidance modifies the mean:
        μ' = μ + σ² * ∇_x log p(y | x)

        Where:
        - μ is the predicted mean from the diffusion model
        - σ² is the variance at current timestep
        - ∇_x log p(y | x) is the classifier gradient (computed by cond_fn)
        - This gradient pushes the sample toward class y

        Why This Works:
        ---------------
        1. The classifier gradient ∇_x log p(y|x) points in the direction that increases
           the probability of class y
        2. Scaling by variance σ² adjusts the step size appropriately for each timestep
        3. Adding this to the mean shifts the distribution toward samples that the
           classifier recognizes as belonging to class y
        4. Repeating this at every timestep guides the entire generation process

        Implementation Note:
        -------------------
        This implementation uses the predicted mean (μ) instead of the current noisy
        sample (x_t) for computing gradients. This fixes a bug in the original OpenAI
        implementation: https://github.com/openai/guided-diffusion/issues/51

        Args:
            cond_fn: Classifier conditioning function that computes ∇_x log p(y|x)
                     Should take (x, t, **guidance_kwargs) and return gradient
            mean: Predicted mean μ_θ(x_t) for next step [B, C, H, W]
            variance: Variance σ² at current timestep [B, 1, 1, 1]
            x: Current noisy sample x_t [B, C, H, W] (not used in corrected version)
            t: Current timestep [B]
            guidance_kwargs: Dict with classifier and conditioning info (e.g., target class y)

        Returns:
            Adjusted mean μ' = μ + σ² * ∇_x log p(y|x) for guided sampling
        """
        # Compute classifier gradient: ∇_x log p(y | x)
        # This gradient points toward samples that look like class y
        gradient = cond_fn(mean, t, **guidance_kwargs)

        # Adjust mean by adding variance-scaled gradient
        # This shifts the distribution toward the desired condition
        new_mean = (
            mean.float() + variance * gradient.float()
        )

        # Debug output to monitor guidance strength
        print("gradient: ", (variance * gradient.float()).mean())

        return new_mean

        
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, cond_fn=None, guidance_kwargs=None):
        """
        Single denoising step from timestep t to t-1 with optional classifier guidance.

        This performs one step of the reverse diffusion process:
        1. Predict mean and variance using the U-Net model
        2. If guidance is enabled, adjust the mean using classifier gradients
        3. Sample from N(adjusted_mean, variance) to get x_{t-1}

        Guidance Application:
        --------------------
        When cond_fn and guidance_kwargs are provided:
        - Computes classifier gradient at the predicted mean
        - Adjusts mean toward desired condition
        - This happens at EVERY timestep during sampling
        - The cumulative effect steers generation toward target class

        Args:
            x: Current sample x_t of shape [B, C, H, W]
            t: Current timestep (integer from 0 to num_timesteps-1)
            x_self_cond: Previous prediction for self-conditioning (optional)
            cond_fn: Classifier gradient function for guidance (optional)
            guidance_kwargs: Dict with classifier and target class info (optional)

        Returns:
            Tuple of:
            - pred_img: Predicted sample at timestep t-1
            - x_start: Predicted original clean image x_0
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)

        # Get predicted mean and variance from diffusion model
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True
        )

        # Apply classifier guidance if provided
        # This is where guidance happens - mean is adjusted by classifier gradients
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(cond_fn, model_mean, variance, x, batched_times, guidance_kwargs)

        # Sample from Gaussian distribution (no noise at final step t=0)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        """
        Complete DDPM sampling loop with optional classifier guidance.

        Generates images from pure noise by iteratively denoising for T timesteps.
        At each step, optionally applies classifier guidance to steer generation.

        Sampling Process:
        ----------------
        1. Start with x_T ~ N(0, I) (pure Gaussian noise)
        2. For t = T-1 down to 0:
            a. Predict x_{t-1} from x_t using the model
            b. If guidance enabled, adjust prediction using classifier gradients
            c. Sample x_{t-1} ~ p(x_{t-1} | x_t, y) where y is target condition
        3. Return x_0 (generated image)

        Guidance Benefits:
        -----------------
        - Without guidance: Generates random samples from the training distribution
        - With guidance: Generates samples conditioned on desired attributes (e.g., class)
        - The classifier doesn't need to be trained jointly with the diffusion model
        - Can use any pre-trained classifier for the image domain

        Args:
            shape: Shape of images to generate [B, C, H, W]
            return_all_timesteps: If True, return all intermediate steps
            cond_fn: Classifier gradient function for guidance (optional)
            guidance_kwargs: Dict with classifier, target class y, and scale (optional)

        Returns:
            Generated images of shape [B, C, H, W] or [B, T, C, H, W] if return_all_timesteps
        """
        batch, device = shape[0], self.betas.device

        # Start from pure Gaussian noise
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # Reverse diffusion process: denoise from T to 0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            # Apply one denoising step with optional guidance
            img, x_start = self.p_sample(img, t, self_cond, cond_fn, guidance_kwargs)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # Convert from [-1, 1] back to [0, 1]
        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        """
        DDIM (Denoising Diffusion Implicit Models) sampling.

        DDIM is a faster sampling method that can generate images with fewer steps
        than the number of training timesteps. It uses a deterministic (when eta=0)
        or semi-stochastic sampling process.

        Note: This implementation currently does NOT support classifier guidance.
        The cond_fn and guidance_kwargs parameters are accepted but not used.
        For guided sampling, use p_sample_loop (DDPM sampling) instead.

        Benefits of DDIM:
        ----------------
        - Faster sampling: Can use 50 steps instead of 1000
        - Deterministic when eta=0: Same noise → same image
        - Quality-speed tradeoff: Fewer steps with minimal quality loss

        Args:
            shape: Shape of images to generate [B, C, H, W]
            return_all_timesteps: If True, return all intermediate steps
            cond_fn: Classifier gradient function (NOT CURRENTLY USED)
            guidance_kwargs: Guidance parameters (NOT CURRENTLY USED)

        Returns:
            Generated images of shape [B, C, H, W] or [B, T, C, H, W] if return_all_timesteps
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # Create time schedule for DDIM (can skip timesteps for faster sampling)
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), ..., (0, -1)]

        # Start from pure noise
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # DDIM sampling loop
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            # Predict noise and x_start
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)

            imgs.append(img)

            # At final step, just use predicted x_start
            if time_next < 0:
                img = x_start
                continue

            # DDIM update rule
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # Variance for stochasticity (eta=0 is deterministic, eta=1 is DDPM)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            # DDIM sampling step: x_{t-1} = sqrt(α_{t-1}) * x_0 + c * ε_θ + σ * noise
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        """
        Generate images from noise with optional classifier guidance.

        This is the main entry point for sampling. It automatically selects between
        DDPM (slower, supports guidance) and DDIM (faster, no guidance support) sampling
        based on the configuration.

        Usage Examples:
        --------------
        # Unconditional sampling (no guidance)
        images = diffusion.sample(batch_size=4)

        # Conditional sampling with classifier guidance
        images = diffusion.sample(
            batch_size=4,
            cond_fn=classifier_gradient_fn,
            guidance_kwargs={
                "classifier": my_classifier,
                "y": target_classes,  # e.g., torch.tensor([1, 5, 3, 7])
                "classifier_scale": 1.0  # guidance strength
            }
        )

        Guidance Parameters:
        -------------------
        - cond_fn: Function that computes classifier gradients
          Should accept (x, t, classifier, y, classifier_scale) and return gradient
        - guidance_kwargs should contain:
          - classifier: Pre-trained classifier model
          - y: Target classes (tensor of shape [batch_size])
          - classifier_scale: Guidance strength (typical range: 0.1 to 10.0)
            Higher values = stronger conditioning = more class-specific but less diverse

        Args:
            batch_size: Number of images to generate
            return_all_timesteps: If True, return all intermediate denoising steps
            cond_fn: Classifier gradient function for guidance (optional)
            guidance_kwargs: Dict with classifier, target class, and scale (optional)

        Returns:
            Generated images of shape [batch_size, channels, H, W]
            or [batch_size, T, channels, H, W] if return_all_timesteps=True
        """
        image_size, channels = self.image_size, self.channels
        # Choose sampling method based on configuration
        # Note: DDIM currently doesn't support guidance, so use DDPM for guided sampling
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps, cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        """
        Interpolate between two images in latent space.

        This adds noise to both images, interpolates in the noisy latent space,
        then denoises back to a clean image. Creates smooth transitions between images.

        Args:
            x1: First image [B, C, H, W]
            x2: Second image [B, C, H, W]
            t: Timestep to add noise to (defaults to num_timesteps - 1)
            lam: Interpolation weight (0 = all x1, 1 = all x2)

        Returns:
            Interpolated image [B, C, H, W]
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        # Add noise to both images
        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        # Interpolate in noisy latent space
        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        # Denoise the interpolated latent
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: Add noise to clean image according to schedule.

        Implements q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)

        This is the forward process that gradually adds Gaussian noise to images.

        Args:
            x_start: Clean image x_0 of shape [B, C, H, W]
            t: Timestep(s) of shape [B]
            noise: Gaussian noise (generated if not provided)

        Returns:
            Noisy image x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Apply noise according to the schedule
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        """
        Compute training loss for the diffusion model.

        Training procedure:
        1. Add noise to clean images using q_sample (forward process)
        2. Predict the noise/x_0/v using the U-Net model
        3. Compute MSE loss between prediction and target
        4. Apply SNR-based loss weighting

        Args:
            x_start: Clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Gaussian noise (optional, generated if not provided)

        Returns:
            Scalar loss value
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Forward diffusion: add noise to images
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # Self-conditioning: 50% of the time, use previous prediction as input
        # This improves quality but slows training by ~25%
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # Get model prediction
        model_out = self.model(x, t, x_self_cond)

        # Determine target based on objective
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # Compute MSE loss
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # Apply SNR-based loss weighting
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Forward pass for training.

        This is called during training to compute the loss.

        Args:
            img: Batch of clean images [B, C, H, W]
            *args, **kwargs: Additional arguments passed to p_losses

        Returns:
            Training loss (scalar)
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # Sample random timesteps for each image in batch
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Normalize images to [-1, 1]
        img = self.normalize(img)

        # Compute and return loss
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    """
    Image dataset for training diffusion models.

    Loads images from a folder, applies transformations, and provides
    batches for training.
    """
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        """
        Args:
            folder: Path to folder containing images
            image_size: Target size for images (will be resized and center cropped)
            exts: List of file extensions to include
            augment_horizontal_flip: Whether to randomly flip images horizontally
            convert_image_to: Convert images to this format (e.g., 'RGB', 'L')
        """
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # Recursively find all image files with specified extensions
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # Image preprocessing pipeline
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        """Return number of images in dataset."""
        return len(self.paths)

    def __getitem__(self, index):
        """
        Load and transform image at given index.

        Args:
            index: Index of image to load

        Returns:
            Transformed image tensor of shape [C, H, W]
        """
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    """
    Trainer for diffusion models with EMA and Accelerate support.

    Handles the training loop, checkpointing, sampling, and logging.
    Uses Accelerate for distributed training and mixed precision.

    Note: This trainer is for the unconditional diffusion model only.
    Classifier guidance is applied during sampling, not training.
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
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        """
        Args:
            diffusion_model: GaussianDiffusion model to train
            folder: Path to training images
            train_batch_size: Batch size for training
            gradient_accumulate_every: Steps to accumulate gradients
            augment_horizontal_flip: Whether to augment with horizontal flips
            train_lr: Learning rate
            train_num_steps: Total number of training steps
            ema_update_every: Update EMA every N steps
            ema_decay: EMA decay rate
            adam_betas: Betas for Adam optimizer
            save_and_sample_every: Save checkpoint and sample every N steps
            num_samples: Number of samples to generate
            results_folder: Folder to save results
            amp: Whether to use automatic mixed precision
            fp16: Whether to use FP16 mixed precision
            split_batches: Whether to split batches for distributed training
            convert_image_to: Convert images to this format
        """
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
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
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

if __name__ == '__main__':
    # =========================================================================
    # EXAMPLE: Classifier and Guidance Function for Classifier-Guided Diffusion
    # =========================================================================

    class Classifier(nn.Module):
        """
        Example time-aware classifier for guiding diffusion generation.

        This is a simple linear classifier that takes both the image and timestep
        as input. In practice, you would use a more sophisticated architecture
        (e.g., a ResNet or Vision Transformer).

        Important Design Considerations:
        --------------------------------
        1. **Time-Awareness**: The classifier should be trained to classify noisy images
           at different noise levels (timesteps). This allows it to provide useful
           gradients throughout the diffusion process.

        2. **Training**: The classifier should be trained on noisy images created
           using the same noise schedule as the diffusion model:
           - For each clean image x_0 and class label y
           - Sample a random timestep t
           - Create noisy image: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
           - Train classifier to predict y from (x_t, t)

        3. **Architecture**: Can be any architecture (CNN, ViT, etc.) as long as it:
           - Takes noisy images and timestep as input
           - Outputs class logits
           - Supports gradient computation with respect to input
        """
        def __init__(self, image_size, num_classes, t_dim=1) -> None:
            """
            Args:
                image_size: Size of square input images (H = W = image_size)
                num_classes: Number of classes to classify
                t_dim: Dimension for timestep encoding (1 for scalar timestep)
            """
            super().__init__()
            # Linear layers for timestep and image features
            # In practice, use a proper image encoder (ResNet, etc.)
            self.linear_t = nn.Linear(t_dim, num_classes)
            self.linear_img = nn.Linear(image_size * image_size * 3, num_classes)

        def forward(self, x, t):
            """
            Classify noisy image at given timestep.

            Args:
                x: Noisy image tensor [B, 3, H, W]
                t: Timestep tensor [B] indicating noise level

            Returns:
                logits: Class logits [B, num_classes]
            """
            B = x.shape[0]
            t = t.view(B, 1)
            # Combine image and timestep features
            # This simple example just adds them; real classifiers use more sophisticated fusion
            logits = self.linear_t(t.float()) + self.linear_img(x.view(x.shape[0], -1))
            return logits

    def classifier_cond_fn(x, t, classifier, y, classifier_scale=1):
        """
        Compute classifier gradient for guiding diffusion generation.

        This is the conditioning function (cond_fn) that gets passed to the
        diffusion model's sample() method. It computes the gradient:
        ∇_x log p(y | x, t)

        This gradient points in the direction that increases the probability
        that the classifier predicts class y for image x at timestep t.

        Mathematical Explanation:
        ------------------------
        We want to sample from p(x | y) ∝ p(x) * p(y | x)
        Taking logs: log p(x | y) = log p(x) + log p(y | x)
        Taking gradients: ∇_x log p(x | y) = ∇_x log p(x) + ∇_x log p(y | x)

        The diffusion model handles ∇_x log p(x), and this function computes
        ∇_x log p(y | x), which we add to guide generation toward class y.

        Implementation Details:
        ----------------------
        1. Enable gradient computation for input x
        2. Forward pass through classifier to get logits
        3. Convert to log probabilities with log_softmax
        4. Select log probability for target class y
        5. Compute gradient of log p(y|x) w.r.t. x using autograd
        6. Scale gradient by classifier_scale to control guidance strength

        Args:
            x: Current sample (noisy image) [B, C, H, W]
            t: Current timestep [B]
            classifier: Trained time-aware classifier model
            y: Target class labels [B] (integer class indices)
            classifier_scale: Guidance strength multiplier
                             - Higher values: stronger conditioning, more class-specific, less diverse
                             - Lower values: weaker conditioning, more diverse, less class-specific
                             - Typical range: 0.1 to 10.0

        Returns:
            gradient: ∇_x log p(y | x, t) scaled by classifier_scale [B, C, H, W]
        """
        assert y is not None, "Target class y must be provided for guidance"

        with torch.enable_grad():
            # Detach x and enable gradients to compute ∇_x log p(y|x)
            x_in = x.detach().requires_grad_(True)

            # Forward pass through classifier
            logits = classifier(x_in, t)

            # Convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Select log probability for target class y
            # This gives us log p(y | x, t) for each sample in batch
            selected = log_probs[range(len(logits)), y.view(-1)]

            # Compute gradient: ∇_x log p(y | x, t)
            # This gradient points toward samples that the classifier recognizes as class y
            grad = torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

            return grad
        

    # =========================================================================
    # EXAMPLE USAGE: Classifier-Guided Image Generation
    # =========================================================================

    # Step 1: Create the U-Net denoising model
    # This model learns to denoise images during training
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)  # Channel multipliers for each resolution
    )

    # Step 2: Wrap in GaussianDiffusion for training and sampling
    image_size = 128
    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000   # Number of diffusion steps
    )

    # Training phase (not shown here):
    # 1. Train the diffusion model on your image dataset (unconditional)
    # 2. Train a time-aware classifier on noisy images at different timesteps
    #    - For each (image, label) pair:
    #      - Sample random timestep t
    #      - Add noise: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    #      - Train classifier to predict label from (x_t, t)

    # Step 3: Create a trained time-aware classifier
    # In practice, this would be loaded from a checkpoint
    classifier = Classifier(image_size=image_size, num_classes=1000, t_dim=1)

    # Step 4: Generate images with classifier guidance
    batch_size = 4
    target_class = 1  # Generate images of class 1

    # Generate images conditioned on target_class using classifier gradients
    sampled_images = diffusion.sample(
        batch_size = batch_size,
        cond_fn = classifier_cond_fn,  # Function that computes classifier gradients
        guidance_kwargs = {
            "classifier": classifier,  # Trained classifier model
            "y": torch.full((batch_size,), target_class).long(),  # Target class for all samples
            "classifier_scale": 1.0,  # Guidance strength (try 0.1 to 10.0)
        }
    )

    # Result: Generated images shaped [4, 3, 128, 128]
    # These images are steered toward class 1 by the classifier gradients
    # Higher classifier_scale = stronger conditioning = more class-specific but less diverse
    print(f"Generated images shape: {sampled_images.shape}")  # (4, 3, 128, 128)

    # Advanced Usage Examples:
    # -----------------------

    # Example 1: Generate different classes in same batch
    # different_classes = torch.tensor([0, 1, 5, 9]).long()  # Different class per sample
    # sampled_images = diffusion.sample(
    #     batch_size=4,
    #     cond_fn=classifier_cond_fn,
    #     guidance_kwargs={"classifier": classifier, "y": different_classes, "classifier_scale": 2.0}
    # )

    # Example 2: Stronger guidance for more class-specific images
    # sampled_images = diffusion.sample(
    #     batch_size=4,
    #     cond_fn=classifier_cond_fn,
    #     guidance_kwargs={"classifier": classifier, "y": target_classes, "classifier_scale": 5.0}
    # )

    # Example 3: Weaker guidance for more diverse images
    # sampled_images = diffusion.sample(
    #     batch_size=4,
    #     cond_fn=classifier_cond_fn,
    #     guidance_kwargs={"classifier": classifier, "y": target_classes, "classifier_scale": 0.5}
    # )