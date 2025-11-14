"""
Simple Diffusion - A Streamlined Denoising Diffusion Probabilistic Model Implementation

This module provides a simplified implementation of denoising diffusion probabilistic models (DDPMs)
using a UViT (U-Net + Vision Transformer) architecture. This implementation differs from more complex
diffusion models in several key ways:

Key Simplifications:
    1. Uses a UViT architecture that combines U-Net's hierarchical feature learning with
       Vision Transformer's global attention at the bottleneck, rather than pure U-Net or pure ViT
    2. Implements log-SNR (Signal-to-Noise Ratio) noise scheduling with cosine schedule as the only option,
       whereas other implementations may offer multiple scheduling strategies
    3. Supports two prediction objectives: v-prediction (from progressive distillation) and
       epsilon-prediction (standard noise prediction), rather than additional objectives like x0 or score
    4. Uses RMSNorm instead of more common LayerNorm or GroupNorm for faster normalization
    5. Employs linear attention in the U-Net path and full attention in the transformer bottleneck
    6. Implements optional noise schedule shifting/interpolation for better multi-resolution handling

Main Components:
    - UViT: Hybrid U-Net + Vision Transformer denoising model
    - GaussianDiffusion: Manages the forward diffusion process and reverse denoising sampling
    - Various building blocks: ResNet blocks, attention layers, feedforward networks

Typical Usage:
    >>> model = UViT(dim=128, channels=3, dim_mults=(1,2,4,8))
    >>> diffusion = GaussianDiffusion(model, image_size=64, channels=3)
    >>> loss = diffusion(images)  # Training
    >>> samples = diffusion.sample(batch_size=16)  # Sampling
"""

import math
from functools import partial, wraps

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.amp import autocast

from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    """Check if a value is not None.

    Args:
        val: Any value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None

def identity(t):
    """Identity function that returns its input unchanged.

    Args:
        t: Any input value

    Returns:
        The input value unchanged
    """
    return t

def is_lambda(f):
    """Check if a function is a lambda function.

    Args:
        f: Function to check

    Returns:
        bool: True if f is a lambda function, False otherwise
    """
    return callable(f) and f.__name__ == "<lambda>"

def default(val, d):
    """Return val if it exists, otherwise return d (or d() if d is a lambda).

    This is useful for providing default values that may be expensive to compute,
    as lambda defaults are only evaluated when needed.

    Args:
        val: Primary value to return if it exists
        d: Default value or lambda function returning default value

    Returns:
        val if it exists, otherwise d or d()
    """
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def cast_tuple(t, l = 1):
    """Convert a value to a tuple of length l, or return unchanged if already a tuple.

    Args:
        t: Value to convert to tuple
        l: Length of tuple to create if t is not already a tuple

    Returns:
        tuple: Either (t,) * l if t is not a tuple, or t if it is
    """
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    """Append singleton dimensions to the end of a tensor's shape.

    This is useful for broadcasting operations where additional dimensions are needed.

    Args:
        t: Input tensor
        dims: Number of singleton dimensions to append

    Returns:
        torch.Tensor: Tensor with dims additional singleton dimensions
    """
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def l2norm(t):
    """L2 normalize a tensor along its last dimension.

    Args:
        t: Input tensor

    Returns:
        torch.Tensor: L2 normalized tensor
    """
    return F.normalize(t, dim = -1)

# u-vit related functions and modules

class Upsample(nn.Module):
    """Upsample feature maps using PixelShuffle.

    This upsampling method uses a 1x1 convolution to expand channels, followed by
    PixelShuffle to rearrange them into spatial dimensions. This is more efficient
    than transposed convolutions and helps avoid checkerboard artifacts.

    Args:
        dim (int): Input channel dimension
        dim_out (int, optional): Output channel dimension. Defaults to dim if None.
        factor (int): Upsampling factor. Defaults to 2.
    """
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = default(dim_out, dim)
        # Conv expands channels by factor^2, then PixelShuffle rearranges to spatial dims
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),  # Smooth activation function
            nn.PixelShuffle(factor)  # Rearrange (h, w, c*f^2) -> (h*f, w*f, c)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        """Initialize the convolution weights with proper scaling.

        Uses Kaiming initialization and replicates weights for each PixelShuffle group
        to ensure stable training from the start.

        Args:
            conv (nn.Conv2d): Convolution layer to initialize
        """
        o, i, h, w = conv.weight.shape
        # Initialize base weights, then replicate for all pixel shuffle groups
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        """Forward pass through upsampling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)

        Returns:
            torch.Tensor: Upsampled tensor of shape (batch, dim_out, height*factor, width*factor)
        """
        return self.net(x)

def Downsample(
    dim,
    dim_out = None,
    factor = 2
):
    """Downsample feature maps using space-to-depth rearrangement.

    This function creates a downsampling module that rearranges spatial information
    into channels (space-to-depth), then uses a 1x1 conv to adjust channel count.
    This is more efficient than strided convolutions and preserves all information.

    Args:
        dim (int): Input channel dimension
        dim_out (int, optional): Output channel dimension. Defaults to dim if None.
        factor (int): Downsampling factor. Defaults to 2.

    Returns:
        nn.Sequential: Sequential module performing downsampling
    """
    return nn.Sequential(
        # Rearrange spatial dims to channels: (h, w, c) -> (h/f, w/f, c*f^2)
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
        # Adjust channel count with 1x1 conv
        nn.Conv2d(dim * (factor ** 2), default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simpler and faster alternative to LayerNorm that normalizes
    using only the root mean square (no mean centering). This is more efficient
    and works well for many tasks.

    Unlike LayerNorm: RMS(x) = x / sqrt(mean(x^2) + eps)
    Instead of: LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)

    Args:
        dim (int): Dimension to normalize
        scale (bool): Whether to use a learnable scale parameter. Defaults to True.
        normalize_dim (int): Which dimension to normalize over. Defaults to 2.
    """
    def __init__(self, dim, scale = True, normalize_dim = 2):
        super().__init__()
        # Learnable scale parameter (gain)
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        """Apply RMS normalization.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: RMS normalized and scaled tensor
        """
        normalize_dim = self.normalize_dim
        # Reshape scale parameter to match input dimensions if learnable
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        # RMS normalize and scale by sqrt(dim) for stable magnitudes
        return F.normalize(x, dim = normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)

# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    """Learned Sinusoidal Positional Embeddings for time/noise level conditioning.

    This module creates positional embeddings using learned frequencies rather than
    fixed frequencies. This allows the model to learn the most useful frequency bands
    for encoding time information.

    Args:
        dim (int): Output embedding dimension (must be even)
    """
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        # Learnable frequencies (randomly initialized)
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        """Generate sinusoidal embeddings.

        Args:
            x (torch.Tensor): Input scalar values (e.g., timesteps) of shape (batch,)

        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch, dim+1)
                         Includes original x, sin(x*w*2π), and cos(x*w*2π)
        """
        x = rearrange(x, 'b -> b 1')
        # Compute learned frequencies: x * weights * 2π
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # Create Fourier features with sin and cos
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # Concatenate with original input for better conditioning
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    """Basic convolutional block with normalization and activation.

    This is a fundamental building block used in ResNet-style architectures.
    It applies convolution, normalization, optional adaptive conditioning, and activation.

    Args:
        dim (int): Input channel dimension
        dim_out (int): Output channel dimension
    """
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out, normalize_dim = 1)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        """Forward pass with optional adaptive normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, dim, height, width)
            scale_shift (tuple, optional): Tuple of (scale, shift) tensors for adaptive
                                          normalization (FiLM conditioning). None means no conditioning.

        Returns:
            torch.Tensor: Output tensor of shape (batch, dim_out, height, width)
        """
        x = self.proj(x)
        x = self.norm(x)

        # Apply adaptive normalization if provided (FiLM: Feature-wise Linear Modulation)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """Residual block with time embedding conditioning.

    A ResNet-style block that processes features through two convolutional blocks
    with a residual connection. Time embeddings are incorporated via adaptive
    normalization (FiLM) in the first block.

    Args:
        dim (int): Input channel dimension
        dim_out (int): Output channel dimension
        time_emb_dim (int, optional): Dimension of time embeddings. If None, no time conditioning.
    """
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        # MLP to project time embeddings to scale and shift parameters
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)  # *2 for scale and shift
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        # Residual connection projection (only needed if dimensions change)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        """Forward pass with optional time conditioning.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, dim, height, width)
            time_emb (torch.Tensor, optional): Time embeddings of shape (batch, time_emb_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, dim_out, height, width)
        """
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # Reshape to (batch, channels, 1, 1) for broadcasting
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # Split into scale and shift for adaptive normalization
            scale_shift = time_emb.chunk(2, dim = 1)

        # First block with time conditioning
        h = self.block1(x, scale_shift = scale_shift)

        # Second block without additional conditioning
        h = self.block2(h)

        # Add residual connection
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """Linear Attention mechanism with O(n) complexity.

    This is an efficient approximation of attention that scales linearly with sequence length
    instead of quadratically. It uses kernel feature maps (softmax) to compute attention
    efficiently via matrix association: (Q * K^T) * V ≈ Q * (K^T * V).

    Compared to standard attention which is O(n²) in memory and compute, this is O(n),
    making it much more efficient for processing spatial feature maps in U-Net paths.

    Args:
        dim (int): Input channel dimension
        heads (int): Number of attention heads. Defaults to 4.
        dim_head (int): Dimension per attention head. Defaults to 32.
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim = 1)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim = 1)
        )

    def forward(self, x):
        """Apply linear attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        residual = x

        b, c, h, w = x.shape

        x = self.norm(x)

        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # Apply softmax to create feature maps (kernel trick for linear attention)
        q = q.softmax(dim = -2)  # Softmax over feature dimension
        k = k.softmax(dim = -1)  # Softmax over spatial dimension

        q = q * self.scale

        # Compute attention via associative property: Q(KV) instead of (QK)V
        # This changes complexity from O(n²) to O(n)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out) + residual

class Attention(nn.Module):
    """Full Self-Attention mechanism with QK normalization.

    Standard scaled dot-product attention with O(n²) complexity. This version includes
    QK normalization (normalizing queries and keys before dot product) which provides
    more stable training and better gradient flow compared to standard attention.

    Used in the Vision Transformer bottleneck for global context modeling, where the
    sequence length is smaller and O(n²) complexity is manageable.

    Args:
        dim (int): Input dimension
        heads (int): Number of attention heads. Defaults to 4.
        dim_head (int): Dimension per head. Defaults to 32.
        scale (int): Attention scale factor. Defaults to 8.
        dropout (float): Dropout probability for attention weights. Defaults to 0.
    """
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 8, dropout = 0.):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        # Learnable scale parameters for queries and keys (for QK normalization)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(self, x):
        """Apply full self-attention with QK normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim)
        """
        x = self.norm(x)

        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # QK normalization: L2 normalize queries and keys for stability
        q, k = map(l2norm, (q, k))

        # Apply learnable scaling
        q = q * self.q_scale
        k = k * self.k_scale

        # Compute attention scores
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Softmax to get attention weights
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    """Feedforward network with adaptive normalization conditioned on time.

    A simple MLP with expansion and contraction, conditioned via FiLM (Feature-wise
    Linear Modulation) using time embeddings. The conditioning allows the network
    to adapt its behavior based on the noise level.

    Args:
        dim (int): Input and output dimension
        cond_dim (int): Conditioning dimension (time embedding dimension)
        mult (int): Hidden dimension multiplier. Defaults to 4.
        dropout (float): Dropout probability. Defaults to 0.
    """
    def __init__(
        self,
        dim,
        cond_dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = RMSNorm(dim, scale = False)
        dim_hidden = dim * mult

        # Project conditioning signal to scale and shift parameters
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange('b d -> b 1 d')  # Add sequence dimension
        )

        # Initialize conditioning projection to zeros for stable training start
        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        # Expansion layer
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias = False),
            nn.SiLU()
        )

        # Contraction layer
        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    def forward(self, x, t):
        """Forward pass with time conditioning.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim)
            t (torch.Tensor): Time conditioning of shape (batch, cond_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim)
        """
        x = self.norm(x)
        x = self.proj_in(x)

        # Apply adaptive normalization using time conditioning
        scale, shift = self.to_scale_shift(t).chunk(2, dim = -1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)

# vit

class Transformer(nn.Module):
    """Vision Transformer block with time conditioning.

    A stack of transformer layers (attention + feedforward) that processes
    the bottleneck features with global context. Each layer is conditioned
    on the diffusion timestep via the feedforward network.

    Args:
        dim (int): Feature dimension
        time_cond_dim (int): Time conditioning dimension
        depth (int): Number of transformer layers
        dim_head (int): Dimension per attention head. Defaults to 32.
        heads (int): Number of attention heads. Defaults to 4.
        ff_mult (int): Feedforward hidden dimension multiplier. Defaults to 4.
        dropout (float): Dropout probability. Defaults to 0.
    """
    def __init__(
        self,
        dim,
        time_cond_dim,
        depth,
        dim_head = 32,
        heads = 4,
        ff_mult = 4,
        dropout = 0.,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim = dim, mult = ff_mult, cond_dim = time_cond_dim, dropout = dropout)
            ]))

    def forward(self, x, t):
        """Forward pass through transformer layers.

        Args:
            x (torch.Tensor): Input features of shape (batch, seq_len, dim)
            t (torch.Tensor): Time embeddings of shape (batch, time_cond_dim)

        Returns:
            torch.Tensor: Transformed features of shape (batch, seq_len, dim)
        """
        for attn, ff in self.layers:
            # Pre-norm architecture with residual connections
            x = attn(x) + x
            x = ff(x, t) + x

        return x
# model

class UViT(nn.Module):
    """U-Net Vision Transformer - Hybrid architecture for diffusion models.

    UViT combines the hierarchical feature learning of U-Net with the global attention
    capabilities of Vision Transformers. The architecture consists of:
    1. Downsampling path: ResNet blocks + linear attention at each resolution
    2. Bottleneck: Vision Transformer for global context
    3. Upsampling path: ResNet blocks + linear attention, with skip connections

    This hybrid approach leverages U-Net's strong inductive biases for image structure
    while using transformers at the bottleneck for long-range dependencies where the
    spatial resolution is small enough to make full attention tractable.

    Args:
        dim (int): Base channel dimension
        init_dim (int, optional): Initial channel dimension after first conv. Defaults to dim.
        out_dim (int, optional): Output channel dimension. Defaults to input channels.
        dim_mults (tuple): Channel multipliers for each resolution level. Defaults to (1, 2, 4, 8).
        downsample_factor (int or tuple): Downsampling factor at each level. Defaults to 2.
        channels (int): Number of input/output image channels. Defaults to 3.
        vit_depth (int): Number of transformer layers in bottleneck. Defaults to 6.
        vit_dropout (float): Dropout probability in transformer. Defaults to 0.2.
        attn_dim_head (int): Dimension per attention head. Defaults to 32.
        attn_heads (int): Number of attention heads. Defaults to 4.
        ff_mult (int): Feedforward hidden dimension multiplier. Defaults to 4.
        learned_sinusoidal_dim (int): Dimension for learned sinusoidal embeddings. Defaults to 16.
        init_img_transform (callable, optional): Optional transform to apply to input (e.g., DWT).
        final_img_itransform (callable, optional): Inverse transform for output.
        patch_size (int): Patch size for patchification. Defaults to 1 (no patchification).
        dual_patchnorm (bool): Whether to use dual normalization for patches. Defaults to False.
    """
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        downsample_factor = 2,
        channels = 3,
        vit_depth = 6,
        vit_dropout = 0.2,
        attn_dim_head = 32,
        attn_heads = 4,
        ff_mult = 4,
        learned_sinusoidal_dim = 16,
        init_img_transform: callable = None,
        final_img_itransform: callable = None,
        patch_size = 1,
        dual_patchnorm = False
    ):
        super().__init__()

        # Optional initial image transform (e.g., DWT - Discrete Wavelet Transform)
        # Allows for processing in frequency domain or other transformed spaces

        if exists(init_img_transform) and exists(final_img_itransform):
            # Verify that transform and inverse transform are properly paired
            init_shape = torch.Size(1, 1, 32, 32)
            mock_tensor = torch.randn(init_shape)
            assert final_img_itransform(init_img_transform(mock_tensor)).shape == init_shape

        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # Optional patchification as an alternative to image transforms
        # Converts image into patches (similar to ViT) for different inductive bias

        self.unpatchify = identity

        input_channels = channels * (patch_size ** 2)
        needs_patch = patch_size > 1

        if needs_patch:
            if not dual_patchnorm:
                # Simple patching: strided convolution
                self.init_conv = nn.Conv2d(channels, init_dim, patch_size, stride = patch_size)
            else:
                # Dual normalization: normalize both in patch space and feature space
                # This can provide more stable training for larger patch sizes
                self.init_conv = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
                    nn.LayerNorm(input_channels),
                    nn.Linear(input_channels, init_dim),
                    nn.LayerNorm(init_dim),
                    Rearrange('b h w c -> b c h w')
                )

            # Unpatchify using transposed convolution
            self.unpatchify = nn.ConvTranspose2d(input_channels, channels, patch_size, stride = patch_size)

        # Determine channel dimensions at each resolution level
        # dims = [init_dim, dim*1, dim*2, dim*4, dim*8] for default dim_mults

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))  # Pairs of (input_dim, output_dim) for each level

        # Time embeddings for conditioning on diffusion timestep
        # Uses learned sinusoidal embeddings passed through an MLP

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1  # +1 for the original time value

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Configure downsampling factors for each resolution level
        # Can be a single int or a tuple with different factors per level

        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        assert len(downsample_factor) == len(dim_mults)

        # Build downsampling and upsampling paths

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Downsampling path: process at each resolution with ResNet + Attention
        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),  # First resnet block
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),  # Second resnet block
                LinearAttention(dim_in),  # Efficient attention for spatial features
                Downsample(dim_in, dim_out, factor = factor)  # Reduce spatial resolution
            ]))

        mid_dim = dims[-1]

        # Vision Transformer bottleneck for global context
        # At this point spatial resolution is smallest, making full attention tractable
        self.vit = Transformer(
            dim = mid_dim,
            time_cond_dim = time_dim,
            depth = vit_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            ff_mult = ff_mult,
            dropout = vit_dropout
        )

        # Upsampling path: reverse of downsampling with skip connections
        for ind, ((dim_in, dim_out), factor) in enumerate(zip(reversed(in_out), reversed(downsample_factor))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, factor = factor),  # Increase spatial resolution
                ResnetBlock(dim_in * 2, dim_in, time_emb_dim = time_dim),  # *2 for skip connection concat
                ResnetBlock(dim_in * 2, dim_in, time_emb_dim = time_dim),  # *2 for skip connection concat
                LinearAttention(dim_in),
            ]))

        # Final layers to produce output
        default_out_dim = input_channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim = time_dim)  # *2 for final skip
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward(self, x, time):
        """Forward pass through the UViT.

        Args:
            x (torch.Tensor): Input images of shape (batch, channels, height, width)
            time (torch.Tensor): Diffusion timesteps of shape (batch,)

        Returns:
            torch.Tensor: Predicted noise or v-prediction of shape (batch, channels, height, width)
        """
        # Apply optional input transform (e.g., DWT)
        x = self.init_img_transform(x)

        # Initial convolution to project to feature space
        x = self.init_conv(x)
        r = x.clone()  # Save for final skip connection

        # Encode timestep
        t = self.time_mlp(time)

        # List to store skip connections for U-Net upsampling path
        h = []

        # Downsampling path: progressively reduce spatial resolution
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)  # Store for skip connection

            x = block2(x, t)
            x = attn(x)  # Apply linear attention
            h.append(x)  # Store for skip connection

            x = downsample(x)  # Reduce spatial dimensions

        # Prepare for transformer: convert from spatial to sequence format
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')  # Flatten spatial dimensions into sequence

        # Vision Transformer bottleneck: global attention over all spatial positions
        x = self.vit(x, t)

        # Convert back to spatial format
        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')

        # Upsampling path: progressively increase spatial resolution
        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)  # Increase spatial dimensions

            # Concatenate with skip connection from downsampling path
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            # Concatenate with another skip connection
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)  # Apply linear attention

        # Final skip connection from very first feature map
        x = torch.cat((x, r), dim = 1)

        # Final processing
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        # Unpatchify if patchification was used
        x = self.unpatchify(x)

        # Apply optional output inverse transform
        return self.final_img_itransform(x)

# normalization functions

def normalize_to_neg_one_to_one(img):
    """Normalize image from [0, 1] to [-1, 1] range.

    Args:
        img (torch.Tensor): Image tensor with values in [0, 1]

    Returns:
        torch.Tensor: Normalized tensor with values in [-1, 1]
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """Unnormalize tensor from [-1, 1] to [0, 1] range.

    Args:
        t (torch.Tensor): Tensor with values in [-1, 1]

    Returns:
        torch.Tensor: Unnormalized tensor with values in [0, 1]
    """
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    """Pad dimensions of tensor t to match the number of dimensions in x.

    Adds singleton dimensions to the right of t until it has the same number
    of dimensions as x, enabling broadcasting in operations.

    Args:
        x (torch.Tensor): Reference tensor with target number of dimensions
        t (torch.Tensor): Tensor to pad

    Returns:
        torch.Tensor: Padded tensor with same ndim as x
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# logsnr schedules and shifting / interpolating decorators
# only cosine for now

def log(t, eps = 1e-20):
    """Safe logarithm with clamping to avoid log(0).

    Args:
        t (torch.Tensor): Input tensor
        eps (float): Minimum value for clamping. Defaults to 1e-20.

    Returns:
        torch.Tensor: Logarithm of clamped input
    """
    return torch.log(t.clamp(min = eps))

def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    """Cosine schedule for log signal-to-noise ratio (log-SNR).

    This schedule provides smooth, continuous noise scheduling that works well
    for diffusion models. The cosine schedule allocates more steps to medium
    noise levels compared to linear schedules.

    Args:
        t (torch.Tensor): Time values in [0, 1], where 0 is no noise and 1 is maximum noise
        logsnr_min (float): Minimum log-SNR value (high noise). Defaults to -15.
        logsnr_max (float): Maximum log-SNR value (low noise). Defaults to 15.

    Returns:
        torch.Tensor: Log-SNR values for the given timesteps
    """
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_shifted(fn, image_d, noise_d):
    """Shift a log-SNR schedule based on image and noise dimensions.

    This decorator shifts the noise schedule to account for different image resolutions.
    Smaller images need less noise, larger images need more. This is from the
    "Simple Diffusion" paper's resolution-dependent noise scheduling.

    Args:
        fn (callable): Base log-SNR scheduling function
        image_d (int): Image dimension (e.g., image size)
        noise_d (int): Reference noise dimension

    Returns:
        callable: Shifted scheduling function
    """
    shift = 2 * math.log(noise_d / image_d)
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner

def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    """Interpolate between two shifted log-SNR schedules.

    Creates a schedule that interpolates between schedules for different resolutions,
    allowing the model to handle multiple resolutions smoothly.

    Args:
        fn (callable): Base log-SNR scheduling function
        image_d (int): Image dimension
        noise_d_low (int): Lower bound noise dimension
        noise_d_high (int): Upper bound noise dimension

    Returns:
        callable: Interpolated scheduling function
    """
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        # Interpolate based on timestep t
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner

# main gaussian diffusion class

class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion process for training and sampling.

    This class implements the forward diffusion process (adding noise) and reverse
    denoising process (removing noise) for denoising diffusion probabilistic models.
    It uses log-SNR parameterization which is more numerically stable than the
    traditional alpha/beta parameterization.

    Key simplifications compared to other implementations:
    - Uses log-SNR formulation exclusively (no alpha_bar or beta schedules)
    - Supports only two prediction objectives: v-prediction and epsilon-prediction
    - Uses continuous time formulation (t in [0,1]) rather than discrete timesteps
    - Implements Min-SNR loss weighting for improved training stability

    Args:
        model (UViT): The denoising model (typically UViT)
        image_size (int): Size of input images (assumes square images)
        channels (int): Number of image channels. Defaults to 3.
        pred_objective (str): Prediction target - 'v' for v-prediction or 'eps' for noise.
                             Defaults to 'v'.
        noise_schedule (callable): Base noise scheduling function. Defaults to cosine schedule.
        noise_d (int, optional): Noise dimension for shifted schedule. Mutually exclusive with
                                noise_d_low/high.
        noise_d_low (int, optional): Lower noise dimension for interpolated schedule.
        noise_d_high (int, optional): Upper noise dimension for interpolated schedule.
        num_sample_steps (int): Number of denoising steps during sampling. Defaults to 500.
        clip_sample_denoised (bool): Whether to clamp denoised samples to [-1, 1]. Defaults to True.
        min_snr_loss_weight (bool): Whether to use Min-SNR loss weighting. Defaults to True.
        min_snr_gamma (float): Gamma parameter for Min-SNR weighting. Defaults to 5.
    """
    def __init__(
        self,
        model: UViT,
        *,
        image_size,
        channels = 3,
        pred_objective = 'v',
        noise_schedule = logsnr_schedule_cosine,
        noise_d = None,
        noise_d_low = None,
        noise_d_high = None,
        num_sample_steps = 500,
        clip_sample_denoised = True,
        min_snr_loss_weight = True,
        min_snr_gamma = 5
    ):
        super().__init__()
        assert pred_objective in {'v', 'eps'}, 'whether to predict v-space (progressive distillation paper) or noise'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # training objective: 'v' for v-prediction, 'eps' for noise prediction

        self.pred_objective = pred_objective

        # noise schedule configuration

        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        # Apply schedule shifting/interpolation for resolution-dependent noise

        self.log_snr = noise_schedule

        if exists(noise_d):
            # Shifted schedule: adjust noise level based on resolution
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            # Interpolated schedule: smooth interpolation between resolutions
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

        # sampling parameters

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # Min-SNR loss weighting: prevents over-weighting very noisy samples

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        """Get the device of the model.

        Returns:
            torch.device: Device where model parameters are stored
        """
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        """Compute the mean and variance for the reverse diffusion step.

        This function computes the posterior distribution p(x_{t-1} | x_t, x_0) where
        x_0 is predicted from the model. Uses the DDIM/DDPM formulation with log-SNR.

        Args:
            x (torch.Tensor): Noisy image at current timestep
            time (float): Current timestep (continuous, in [0, 1])
            time_next (float): Next timestep (closer to 0)

        Returns:
            tuple: (model_mean, posterior_variance) for the reverse step
        """
        # Get log-SNR at current and next timesteps
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)  # Coefficient for interpolation

        # Convert log-SNR to alpha and sigma values
        # alpha^2 = sigmoid(log_snr), sigma^2 = sigmoid(-log_snr)
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        # Get model prediction
        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred = self.model(x, batch_log_snr)

        # Predict x_0 from model output based on prediction objective
        if self.pred_objective == 'v':
            # v-prediction: v = alpha * noise - sigma * x_0
            # Solve for x_0: x_0 = alpha * x_t - sigma * v
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            # epsilon-prediction: x_t = alpha * x_0 + sigma * eps
            # Solve for x_0: x_0 = (x_t - sigma * eps) / alpha
            x_start = (x - sigma * pred) / alpha

        # Clamp predicted x_0 to valid range
        x_start.clamp_(-1., 1.)

        # Compute mean of reverse step using DDIM formulation
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        # Compute variance for stochastic sampling (DDPM) or 0 for deterministic (DDIM)
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        """Single reverse diffusion sampling step.

        Takes a noisy sample at time t and denoises it to time t-1.

        Args:
            x (torch.Tensor): Noisy image at current timestep
            time (float): Current timestep
            time_next (float): Next timestep (less noisy)

        Returns:
            torch.Tensor: Denoised image at next timestep
        """
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        # If we've reached the final step, return the mean without adding noise
        if time_next == 0:
            return model_mean

        # Add noise for stochastic sampling (DDPM)
        # For deterministic sampling (DDIM), model_variance would be 0
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """Complete sampling loop from pure noise to generated image.

        Progressively denoises random noise over num_sample_steps iterations.

        Args:
            shape (tuple): Shape of samples to generate (batch, channels, height, width)

        Returns:
            torch.Tensor: Generated images in [0, 1] range
        """
        batch = shape[0]

        # Start from pure noise
        img = torch.randn(shape, device = self.device)
        # Create timestep schedule from 1 (max noise) to 0 (no noise)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        # Iteratively denoise
        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        # Final clamping and normalization
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        """Generate samples from the diffusion model.

        Args:
            batch_size (int): Number of samples to generate. Defaults to 16.

        Returns:
            torch.Tensor: Generated images of shape (batch_size, channels, image_size, image_size)
        """
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # training related functions - noise prediction

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, times, noise = None):
        """Forward diffusion process: add noise to clean images.

        Implements q(x_t | x_0) - the forward noising process at arbitrary timesteps.
        Uses the closed-form solution: x_t = alpha_t * x_0 + sigma_t * noise

        Args:
            x_start (torch.Tensor): Clean images
            times (torch.Tensor): Continuous timesteps in [0, 1]
            noise (torch.Tensor, optional): Noise to add. Generated if None.

        Returns:
            tuple: (noised_images, log_snr) - noised images and log signal-to-noise ratios
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        # Pad log_snr dimensions for broadcasting
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        # Convert log-SNR to alpha and sigma
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        # Apply forward diffusion: x_t = alpha * x_0 + sigma * noise
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, x_start, times, noise = None):
        """Compute training loss for the diffusion model.

        Adds noise to images, predicts either noise or v-prediction, and computes
        MSE loss with Min-SNR weighting.

        Args:
            x_start (torch.Tensor): Clean images
            times (torch.Tensor): Random timesteps
            noise (torch.Tensor, optional): Noise to add. Generated if None.

        Returns:
            torch.Tensor: Weighted MSE loss (scalar)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Apply forward diffusion
        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        # Get model prediction
        model_out = self.model(x, log_snr)

        # Compute target based on prediction objective
        if self.pred_objective == 'v':
            # v-prediction target: v = alpha * noise - sigma * x_0
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == 'eps':
            # epsilon-prediction target: just the noise
            target = noise

        # Compute per-sample MSE loss
        loss = F.mse_loss(model_out, target, reduction = 'none')

        # Average over all dimensions except batch
        loss = reduce(loss, 'b ... -> b', 'mean')

        # Compute loss weighting using SNR
        snr = log_snr.exp()

        # Apply Min-SNR weighting to prevent over-weighting high-noise samples
        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max = self.min_snr_gamma)

        # Different weighting for different objectives
        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)

        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr

        # Apply weighting and average over batch
        return (loss * loss_weight).mean()

    def forward(self, img, *args, **kwargs):
        """Training forward pass.

        Randomly samples timesteps and computes training loss.

        Args:
            img (torch.Tensor): Batch of images in [0, 1] range
            *args: Additional arguments passed to p_losses
            **kwargs: Additional keyword arguments passed to p_losses

        Returns:
            torch.Tensor: Training loss (scalar)
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # Normalize to [-1, 1]
        img = normalize_to_neg_one_to_one(img)
        # Sample random timesteps uniformly from [0, 1]
        times = torch.zeros((img.shape[0],), device = self.device).float().uniform_(0, 1)

        return self.p_losses(img, times, *args, **kwargs)
