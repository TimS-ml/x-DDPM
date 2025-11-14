"""
3D Karras UNet: Magnitude-Preserving UNet for 3D Volumetric Data

This module implements a 3D variant of the magnitude-preserving UNet proposed in
"Analyzing and Improving the Training Dynamics of Diffusion Models"
by Karras et al. (https://arxiv.org/abs/2312.02696).

The architecture extends the 2D Karras UNet to handle volumetric data such as:
- Videos (temporal sequences of images)
- 3D medical imaging (CT scans, MRI volumes)
- Scientific simulations (fluid dynamics, climate data)
- Any 5D tensor data with shape (batch, channels, frames/depth, height, width)

Key Innovations for 3D:
----------------------
1. **3D Convolutions**: Uses Conv3d operations instead of Conv2d to process spatial
   and temporal/depth dimensions simultaneously.

2. **Factorized Space-Time Attention**: Optional separation of attention across
   spatial dimensions (height, width) and temporal/depth dimension for efficiency.

3. **Flexible Downsampling**: Independent control over downsampling in frame/depth
   vs spatial dimensions, allowing preservation of temporal resolution when needed.

4. **Magnitude Preservation**: Maintains signal magnitude throughout the network to
   improve training stability and gradient flow, critical for deep 3D networks.

Core Principles from Karras et al.:
-----------------------------------
- Bias-free design: No bias terms in any layers
- No normalization layers: Instead uses pixel normalization and weight normalization
- Magnitude-preserving operations: Custom Add, Cat operations preserve signal strength
- Adaptive Gain: Learnable scaling parameters for fine-grained control
- MP activations: SiLU activation scaled to preserve magnitude

This architecture is particularly effective for:
- Video generation and prediction
- 3D medical image synthesis and denoising
- Volumetric data generation in scientific domains
- Any task requiring consistent processing across space and time/depth
"""

import math
from math import sqrt, ceil
from functools import partial
from typing import Optional, Union, Tuple

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

from denoising_diffusion_pytorch.attend import Attend

# Helper functions for common operations

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
        val if it exists, otherwise d (or d() if d is callable)
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def xnor(x, y):
    """
    Logical XNOR operation - returns True if x and y are both True or both False.

    Useful for validating that optional features are used consistently
    (either both enabled or both disabled).

    Args:
        x: First boolean value
        y: Second boolean value

    Returns:
        bool: True if x and y have the same truth value
    """
    return not (x ^ y)

def append(arr, el):
    """
    Append element to array in-place.

    Args:
        arr: List to append to
        el: Element to append
    """
    arr.append(el)

def prepend(arr, el):
    """
    Insert element at the beginning of array in-place.

    Args:
        arr: List to prepend to
        el: Element to insert at index 0
    """
    arr.insert(0, el)

def pack_one(t, pattern):
    """
    Pack a single tensor according to einops pattern.

    Args:
        t: Tensor to pack
        pattern: Einops pattern string

    Returns:
        Packed tensor
    """
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """
    Unpack a single tensor according to einops pattern.

    Args:
        t: Tensor to unpack
        ps: Packed shapes
        pattern: Einops pattern string

    Returns:
        Unpacked tensor
    """
    return unpack(t, ps, pattern)[0]

def cast_tuple(t, length = 1):
    """
    Convert value to tuple of specified length if not already a tuple.

    If t is already a tuple, return it as-is.
    If t is not a tuple, return a tuple with t repeated length times.

    Args:
        t: Value to convert or tuple to return
        length: Number of times to repeat t if not a tuple

    Returns:
        Tuple of length 'length'
    """
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    """
    Check if numer is evenly divisible by denom.

    Args:
        numer: Numerator (dividend)
        denom: Denominator (divisor)

    Returns:
        bool: True if remainder is 0
    """
    return (numer % denom) == 0

# in paper, they use eps 1e-4 for pixelnorm

def l2norm(t, dim = -1, eps = 1e-12):
    """
    L2 normalization along specified dimension.

    Normalizes the tensor such that the L2 norm along the specified dimension is 1.

    Args:
        t: Input tensor
        dim: Dimension along which to normalize (default: -1)
        eps: Small epsilon for numerical stability (default: 1e-12)

    Returns:
        Normalized tensor
    """
    return F.normalize(t, dim = dim, eps = eps)

# Magnitude-preserving activations
# Section 2.5 of the Karras paper

class MPSiLU(Module):
    """
    Magnitude-Preserving SiLU (Swish) activation function.

    Standard SiLU activation scaled by a constant factor (1/0.596) to preserve
    the expected magnitude of the input. This prevents magnitude collapse that
    can occur with standard activations in very deep networks.

    The scaling factor 0.596 is empirically derived to maintain E[||x||^2] ≈ E[||SiLU(x)||^2]
    for zero-mean unit-variance input.

    Forward pass:
        output = SiLU(x) / 0.596
    """
    def forward(self, x):
        """
        Apply magnitude-preserving SiLU activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Activated tensor with preserved magnitude, same shape as input
        """
        return F.silu(x) / 0.596

# Gain - learnable layer scaling

class Gain(Module):
    """
    Learnable gain (scaling) layer.

    A simple learnable scalar that multiplies the input. Initialized to 0,
    allowing the layer to start as identity and gradually learn to scale.

    This is used at the output of certain blocks to allow fine-grained control
    over the contribution of that block to the overall network output.
    """
    def __init__(self):
        """Initialize gain parameter to 0."""
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        """
        Scale input by learnable gain.

        Args:
            x: Input tensor of any shape

        Returns:
            Scaled tensor: x * gain
        """
        return x * self.gain

# Magnitude-preserving concatenation
# Equation (103) from the paper - default to t=0.5, which they recommended

class MPCat(Module):
    """
    Magnitude-Preserving Concatenation.

    Concatenates two tensors along a dimension while preserving the overall magnitude.
    This prevents the magnitude from growing as sqrt(2) which would happen with naive
    concatenation.

    The operation balances the contributions of both tensors using a mixing parameter t,
    and applies a correction factor C to maintain the expected magnitude.

    Math (Equation 103):
        C = sqrt((Na + Nb) / ((1-t)^2 + t^2))
        output = C * concat((1-t)*a/sqrt(Na), t*b/sqrt(Nb))

    where Na and Nb are the sizes of a and b along the concatenation dimension.

    Args:
        t: Mixing parameter between 0 and 1 (default: 0.5 for equal weighting)
        dim: Dimension along which to concatenate (default: -1)
    """
    def __init__(self, t = 0.5, dim = -1):
        """
        Initialize magnitude-preserving concatenation.

        Args:
            t: Weight for second tensor, 1-t used for first tensor (default: 0.5)
            dim: Concatenation dimension (default: -1, typically the channel dimension)
        """
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        """
        Concatenate tensors a and b with magnitude preservation.

        Args:
            a: First tensor
            b: Second tensor (must have same shape as a except along dim)

        Returns:
            Concatenated tensor with preserved magnitude
        """
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        # Compute correction factor to preserve magnitude
        C = sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

        # Scale inputs by their mixing weights and dimension sizes
        a = a * (1. - t) / sqrt(Na)
        b = b * t / sqrt(Nb)

        return C * torch.cat((a, b), dim = dim)

# Magnitude-preserving addition (residual connection)
# Equation (88) from the paper
# Empirically, they found t=0.3 optimal for encoder/decoder/attention residuals
# and t=0.5 for embedding residuals

class MPAdd(Module):
    """
    Magnitude-Preserving Addition for residual connections.

    Combines the main path and residual path while preserving magnitude.
    This is crucial for deep networks to maintain stable gradient flow.

    Unlike standard residual addition (x + res), this applies weighted averaging
    with a correction factor to prevent magnitude growth:

    Math (Equation 88):
        output = ((1-t)*x + t*res) / sqrt((1-t)^2 + t^2)

    The denominator ensures that if x and res have similar magnitudes,
    the output will also have similar magnitude (not sqrt(2) times larger).

    Recommended values:
        t=0.3 for encoder/decoder/attention residuals (favors main path)
        t=0.5 for embedding residuals (equal weighting)

    Args:
        t: Weight for residual connection, 1-t used for main path
    """
    def __init__(self, t):
        """
        Initialize magnitude-preserving addition.

        Args:
            t: Residual mixing weight (0.3 for most layers, 0.5 for embeddings)
        """
        super().__init__()
        self.t = t

    def forward(self, x, res):
        """
        Add residual to main path with magnitude preservation.

        Args:
            x: Main path tensor
            res: Residual path tensor (must have same shape as x)

        Returns:
            Combined tensor with preserved magnitude
        """
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

# PixelNorm normalization
# Equation (30) from the paper

class PixelNorm(Module):
    """
    Pixel Normalization layer.

    Normalizes the input along a specified dimension (typically the channel dimension)
    such that the L2 norm equals sqrt(dimension_size).

    Unlike LayerNorm or BatchNorm, PixelNorm:
    - Has no learnable parameters
    - Normalizes each sample independently
    - Uses a higher epsilon (1e-4) for stability

    Math:
        output = (x / ||x||_2) * sqrt(dim_size)

    where ||x||_2 is computed along the specified dimension.

    This normalization helps maintain consistent magnitudes throughout the network
    without introducing the training instabilities that can come from learned
    normalization parameters.

    Args:
        dim: Dimension along which to normalize (typically 1 for channel dimension)
        eps: Epsilon for numerical stability (default: 1e-4, higher than typical)
    """
    def __init__(self, dim, eps = 1e-4):
        """
        Initialize PixelNorm layer.

        Args:
            dim: Dimension to normalize along (typically 1 for channels)
            eps: Numerical stability epsilon (paper uses 1e-4, higher than usual)
        """
        super().__init__()
        # high epsilon for the pixel norm in the paper
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        """
        Apply pixel normalization.

        Args:
            x: Input tensor of any shape

        Returns:
            Normalized tensor with L2 norm = sqrt(dim_size) along specified dimension
        """
        dim = self.dim
        return l2norm(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

# Weight-normalized Conv3d and Linear layers
# Algorithm 1 in the paper - forced weight normalization

def normalize_weight(weight, eps = 1e-4):
    """
    Normalize weight tensor to unit norm with correction factor.

    Implements weight normalization from Algorithm 1 of the paper:
    1. Flatten weight to (output_channels, -1)
    2. L2 normalize along the flattened dimension
    3. Scale by sqrt(total_params / output_channels) to preserve magnitude

    Args:
        weight: Weight tensor of any shape (typically conv or linear weights)
        eps: Small epsilon for numerical stability (default: 1e-4)

    Returns:
        Normalized weight tensor with same shape as input
    """
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

class Conv3d(Module):
    """
    3D Convolution with forced weight normalization and no bias.

    Key features for 3D volumetric processing:
    - Processes 5D tensors: (batch, channels, frames/depth, height, width)
    - Weight normalization applied every forward pass for training stability
    - Bias-free design as per Karras architecture
    - He initialization scaled by fan-in
    - Optional concatenation of ones channel for expressivity

    The 3D convolution operates across all spatial dimensions (frame/time, height, width)
    simultaneously, making it ideal for:
    - Video data (temporal coherence)
    - 3D medical imaging (volumetric structures)
    - Scientific simulations (3D spatial dynamics)

    Args:
        dim_in: Number of input channels
        dim_out: Number of output channels
        kernel_size: Size of the 3D kernel (applied to all 3 dimensions)
        eps: Epsilon for weight normalization (default: 1e-4)
        concat_ones_to_input: If True, add a channel of ones to input for
            additional expressivity without bias (used in input block)
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        """
        Initialize 3D convolution layer.

        Args:
            dim_in: Input channels
            dim_out: Output channels
            kernel_size: Kernel size (same for all 3 dimensions)
            eps: Weight normalization epsilon
            concat_ones_to_input: Whether to concatenate ones channel to input
        """
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 3
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        """
        Apply 3D convolution with weight normalization.

        During training, weights are normalized in-place for stability.
        Weights are then scaled by 1/sqrt(fan_in) for proper initialization.

        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)

        Returns:
            Convolved tensor of shape (batch, out_channels, frames, height, width)
        """
        # During training, update stored weights to their normalized version
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        # Normalize and scale weights for forward pass
        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)

        # Optionally add ones channel for expressivity without bias
        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 1, 0), value = 1.)

        return F.conv3d(x, weight, padding='same')

class Linear(Module):
    """
    Linear layer with forced weight normalization and no bias.

    Implements a fully-connected layer following Karras principles:
    - Weight normalization every forward pass
    - No bias term
    - Scaled by 1/sqrt(fan_in) for proper magnitude

    Used primarily for processing embeddings (time, class conditions).

    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        eps: Epsilon for weight normalization (default: 1e-4)
    """
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        """
        Initialize linear layer.

        Args:
            dim_in: Input feature dimension
            dim_out: Output feature dimension
            eps: Weight normalization epsilon
        """
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        """
        Apply linear transformation with weight normalization.

        Args:
            x: Input tensor of shape (..., dim_in)

        Returns:
            Output tensor of shape (..., dim_out)
        """
        # During training, update stored weights to their normalized version
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        # Normalize and scale weights for forward pass
        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

# Magnitude-preserving Fourier embeddings for time/noise level

class MPFourierEmbedding(Module):
    """
    Magnitude-Preserving Fourier Feature Embedding.

    Embeds scalar values (like time or noise level) into high-dimensional
    sinusoidal features for conditioning the UNet. This allows the network
    to distinguish between different time steps in the diffusion process.

    Process:
    1. Multiply input by random frequencies
    2. Apply sin and cos to get periodic features
    3. Scale by sqrt(2) to preserve magnitude

    The random frequencies are fixed (not learned) and provide a diverse
    set of periodic basis functions for representing the scalar input.

    Math:
        freqs = x * weights * 2π
        output = sqrt(2) * [sin(freqs), cos(freqs)]

    This produces a dim-dimensional embedding where adjacent time steps
    have similar but not identical embeddings, allowing the network to
    learn smooth time-dependent behavior.

    Args:
        dim: Embedding dimension (must be even, since half sin and half cos)
    """
    def __init__(self, dim):
        """
        Initialize Fourier embedding.

        Args:
            dim: Output embedding dimension (must be divisible by 2)
        """
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        # Fixed random frequencies (not trainable)
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        """
        Embed scalar input into Fourier features.

        Args:
            x: Scalar input tensor of shape (batch,)

        Returns:
            Embedded features of shape (batch, dim)
        """
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # Concatenate sin and cos, scale by sqrt(2) for magnitude preservation
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)

# Building block modules for 3D UNet

class Encoder(Module):
    """
    Encoder block for 3D UNet (downsampling path).

    Each encoder block processes volumetric data through:
    1. Optional downsampling (trilinear interpolation + conv) for multi-resolution
    2. Pixel normalization for magnitude control
    3. Two convolutional blocks with MP-SiLU activation
    4. Optional time/class conditioning via adaptive gain
    5. Residual connection with magnitude preservation
    6. Optional self-attention for long-range dependencies

    3D-Specific Features:
    ---------------------
    - Processes 5D tensors: (batch, channels, frames, height, width)
    - Flexible downsampling: can downsample in frame/time dimension independently
      from spatial dimensions (height, width)
    - Optional factorized space-time attention: separates attention across
      spatial dims vs temporal dim for computational efficiency

    Downsampling Configuration:
    --------------------------
    The downsample_config tuple controls which dimensions to downsample:
    - (True, True, True): Downsample all dimensions (frames, height, width)
    - (False, True, True): Downsample spatial only, preserve temporal resolution
    - (True, False, False): Downsample temporal only (rare)

    Args:
        dim: Input channel dimension
        dim_out: Output channel dimension (default: same as dim)
        emb_dim: Dimension of time/class embedding for conditioning
        dropout: Dropout probability for regularization
        mp_add_t: Mixing weight for magnitude-preserving residual (default: 0.3)
        has_attn: Whether to include self-attention
        attn_dim_head: Dimension per attention head
        attn_res_mp_add_t: Mixing weight for attention residual
        attn_flash: Whether to use flash attention for efficiency
        factorize_space_time_attn: If True, separate attention for space and time
        downsample: Whether this block performs downsampling
        downsample_config: Which dimensions to downsample (frame, height, width)
    """
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        factorize_space_time_attn = False,
        downsample = False,
        downsample_config: Tuple[bool, bool, bool] = (True, True, True)
    ):
        """
        Initialize encoder block.

        Args:
            dim: Input channels
            dim_out: Output channels (defaults to dim)
            emb_dim: Time/class embedding dimension
            dropout: Dropout rate
            mp_add_t: Residual mixing weight
            has_attn: Include attention layer
            attn_dim_head: Attention head dimension
            attn_res_mp_add_t: Attention residual mixing
            attn_flash: Use flash attention
            factorize_space_time_attn: Separate space/time attention
            downsample: Enable downsampling
            downsample_config: Dimensions to downsample (frame, h, w)
        """
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_config = downsample_config

        self.downsample_conv = None

        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv3d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        self.pixel_norm = PixelNorm(dim = 1)

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv3d(curr_dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv3d(dim_out, dim_out, 3)
        )

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        self.factorized_attn = factorize_space_time_attn

        if has_attn:
            attn_kwargs = dict(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

            if factorize_space_time_attn:
                self.attn = nn.ModuleList([
                    Attention(**attn_kwargs, only_space = True),
                    Attention(**attn_kwargs, only_time = True),
                ])
            else:
                self.attn = Attention(**attn_kwargs)

    def forward(
        self,
        x,
        emb = None
    ):
        """
        Forward pass through encoder block.

        Processing steps:
        1. Optionally downsample via trilinear interpolation
        2. Apply pixel normalization
        3. Save residual connection
        4. First conv block with activation
        5. Apply time/class conditioning (adaptive gain)
        6. Second conv block with dropout
        7. Add residual with magnitude preservation
        8. Optionally apply self-attention (space-time or factorized)

        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)
            emb: Optional conditioning embedding (time/class) of shape (batch, emb_dim)

        Returns:
            Processed tensor of shape (batch, out_channels, out_frames, out_h, out_w)
            where spatial dimensions may be downsampled by factor of 2
        """
        # Downsample using trilinear interpolation based on config
        if self.downsample:
            t, h, w = x.shape[-3:]
            resize_factors = tuple((2 if downsample else 1) for downsample in self.downsample_config)
            interpolate_shape = tuple(shape // factor for shape, factor in zip((t, h, w), resize_factors))

            x = F.interpolate(x, interpolate_shape, mode = 'trilinear')
            x = self.downsample_conv(x)

        # Normalize to control magnitude
        x = self.pixel_norm(x)

        # Save for residual connection
        res = x.clone()

        # First convolutional block
        x = self.block1(x)

        # Apply time/class conditioning if provided
        if exists(emb):
            scale = self.to_emb(emb) + 1  # +1 for identity when embedding is 0
            x = x * rearrange(scale, 'b c -> b c 1 1 1')

        # Second convolutional block with dropout
        x = self.block2(x)

        # Add residual with magnitude preservation
        x = self.res_mp_add(x, res)

        # Apply self-attention if enabled
        if exists(self.attn):
            if self.factorized_attn:
                # Factorized: separate attention for space and time
                attn_space, attn_time = self.attn
                x = attn_space(x)
                x = attn_time(x)
            else:
                # Full 3D attention
                x = self.attn(x)

        return x

class Decoder(Module):
    """
    Decoder block for 3D UNet (upsampling path).

    Mirrors the encoder but works in reverse: processes features from deeper
    layers and optionally upsamples to higher resolutions. Used for both the
    middle blocks and the reconstruction path of the UNet.

    Each decoder block processes volumetric data through:
    1. Optional upsampling (trilinear interpolation) for multi-resolution
    2. Two convolutional blocks with MP-SiLU activation
    3. Optional time/class conditioning via adaptive gain
    4. Residual connection with magnitude preservation
    5. Optional self-attention for long-range dependencies

    Key Differences from Encoder:
    -----------------------------
    - Upsamples instead of downsamples (increases resolution)
    - Can accept skip connections from encoder (via MPCat in main UNet)
    - No pixel normalization at start (encoder output is already normalized)

    3D-Specific Features:
    ---------------------
    - Processes 5D tensors: (batch, channels, frames, height, width)
    - Flexible upsampling: can upsample in frame/time dimension independently
      from spatial dimensions (height, width)
    - Maintains temporal coherence during upsampling via trilinear interpolation

    Upsampling Configuration:
    ------------------------
    The upsample_config tuple controls which dimensions to upsample:
    - (True, True, True): Upsample all dimensions (frames, height, width)
    - (False, True, True): Upsample spatial only, preserve temporal resolution
    - (True, False, False): Upsample temporal only

    Args:
        dim: Input channel dimension
        dim_out: Output channel dimension (default: same as dim)
        emb_dim: Dimension of time/class embedding for conditioning
        dropout: Dropout probability for regularization
        mp_add_t: Mixing weight for magnitude-preserving residual (default: 0.3)
        has_attn: Whether to include self-attention
        attn_dim_head: Dimension per attention head
        attn_res_mp_add_t: Mixing weight for attention residual
        attn_flash: Whether to use flash attention for efficiency
        factorize_space_time_attn: If True, separate attention for space and time
        upsample: Whether this block performs upsampling
        upsample_config: Which dimensions to upsample (frame, height, width)
    """
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        factorize_space_time_attn = False,
        upsample = False,
        upsample_config: Tuple[bool, bool, bool] = (True, True, True)
    ):
        """
        Initialize decoder block.

        Args:
            dim: Input channels
            dim_out: Output channels (defaults to dim)
            emb_dim: Time/class embedding dimension
            dropout: Dropout rate
            mp_add_t: Residual mixing weight
            has_attn: Include attention layer
            attn_dim_head: Attention head dimension
            attn_res_mp_add_t: Attention residual mixing
            attn_flash: Use flash attention
            factorize_space_time_attn: Separate space/time attention
            upsample: Enable upsampling
            upsample_config: Dimensions to upsample (frame, h, w)
        """
        super().__init__()
        dim_out = default(dim_out, dim)

        self.upsample = upsample
        self.upsample_config = upsample_config

        self.needs_skip = not upsample

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv3d(dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv3d(dim_out, dim_out, 3)
        )

        self.res_conv = Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        self.factorized_attn = factorize_space_time_attn

        if has_attn:
            attn_kwargs = dict(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

            if factorize_space_time_attn:
                self.attn = nn.ModuleList([
                    Attention(**attn_kwargs, only_space = True),
                    Attention(**attn_kwargs, only_time = True),
                ])
            else:
                self.attn = Attention(**attn_kwargs)

    def forward(
        self,
        x,
        emb = None
    ):
        """
        Forward pass through decoder block.

        Processing steps:
        1. Optionally upsample via trilinear interpolation (2x in selected dims)
        2. Create residual connection (with 1x1 conv if dim changes)
        3. First conv block with activation
        4. Apply time/class conditioning (adaptive gain)
        5. Second conv block with dropout
        6. Add residual with magnitude preservation
        7. Optionally apply self-attention (space-time or factorized)

        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)
            emb: Optional conditioning embedding (time/class) of shape (batch, emb_dim)

        Returns:
            Processed tensor of shape (batch, out_channels, out_frames, out_h, out_w)
            where spatial dimensions may be upsampled by factor of 2
        """
        # Upsample using trilinear interpolation based on config
        if self.upsample:
            t, h, w = x.shape[-3:]
            resize_factors = tuple((2 if upsample else 1) for upsample in self.upsample_config)
            interpolate_shape = tuple(shape * factor for shape, factor in zip((t, h, w), resize_factors))

            x = F.interpolate(x, interpolate_shape, mode = 'trilinear')

        # Residual path (with optional 1x1 conv for dimension matching)
        res = self.res_conv(x)

        # First convolutional block
        x = self.block1(x)

        # Apply time/class conditioning if provided
        if exists(emb):
            scale = self.to_emb(emb) + 1  # +1 for identity when embedding is 0
            x = x * rearrange(scale, 'b c -> b c 1 1 1')

        # Second convolutional block with dropout
        x = self.block2(x)

        # Add residual with magnitude preservation
        x = self.res_mp_add(x, res)

        # Apply self-attention if enabled
        if exists(self.attn):
            if self.factorized_attn:
                # Factorized: separate attention for space and time
                attn_space, attn_time = self.attn
                x = attn_space(x)
                x = attn_time(x)
            else:
                # Full 3D attention
                x = self.attn(x)

        return x

# Self-attention mechanism for 3D volumetric data

class Attention(Module):
    """
    Multi-head self-attention for 3D volumetric data with magnitude preservation.

    This attention mechanism is adapted for 3D data with special features:

    1. **Factorized Space-Time Attention**: Can operate separately on spatial
       dimensions (height, width) vs temporal dimension (frames) for efficiency.
       - only_space=True: Attention across spatial locations for each frame
       - only_time=True: Attention across frames for each spatial location
       - Both False: Full 3D attention (expensive but most expressive)

    2. **Memory Keys/Values**: Learnable memory tokens that attend to all positions,
       helping capture global patterns without explicit conditioning.

    3. **Pixel Normalization**: Applied to Q, K, V for magnitude preservation.

    4. **Magnitude-Preserving Residual**: Output is added to input via MPAdd.

    Factorized Attention Benefits for 3D:
    -------------------------------------
    - Spatial-only: Captures within-frame patterns (objects, textures)
    - Temporal-only: Captures motion, temporal dynamics
    - Combined: Sequential application gives O(T*H*W) instead of O((T*H*W)^2)
    - Critical for high-resolution videos where full attention is prohibitive

    Args:
        dim: Channel dimension
        heads: Number of attention heads
        dim_head: Dimension per attention head
        num_mem_kv: Number of learnable memory key-value pairs
        flash: Whether to use flash attention for efficiency
        mp_add_t: Mixing weight for residual connection (default: 0.3)
        only_space: If True, only attend across spatial dimensions
        only_time: If True, only attend across temporal dimension
    """
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        flash = False,
        mp_add_t = 0.3,
        only_space = False,
        only_time = False
    ):
        """
        Initialize 3D attention module.

        Args:
            dim: Input/output channel dimension
            heads: Number of attention heads
            dim_head: Channels per head
            num_mem_kv: Number of memory tokens
            flash: Use flash attention implementation
            mp_add_t: Residual mixing weight
            only_space: Restrict attention to spatial dimensions only
            only_time: Restrict attention to temporal dimension only
        """
        super().__init__()
        # Can't be both spatial-only and temporal-only
        assert (int(only_space) + int(only_time)) <= 1

        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim = -1)

        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv3d(dim, hidden_dim * 3, 1)
        self.to_out = Conv3d(hidden_dim, dim, 1)

        self.mp_add = MPAdd(t = mp_add_t)

        self.only_space = only_space
        self.only_time = only_time

    def forward(self, x):
        """
        Apply self-attention with optional space-time factorization.

        Process flow:
        1. Generate Q, K, V from input via 1x1 conv
        2. Optionally reshape for factorized attention (space-only or time-only)
        3. Split into attention heads
        4. Concatenate learnable memory keys/values
        5. Apply pixel normalization to Q, K, V
        6. Compute attention and aggregate values
        7. Reshape back to original 3D structure
        8. Project output and add residual

        Factorized Attention Modes:
        ---------------------------
        - Full 3D (default): All voxels attend to all voxels
          Shape: (b, c, t, h, w) -> attend over (t*h*w) positions

        - Spatial only: Each frame attends within itself
          Shape: (b, c, t, h, w) -> (b*t, c, h, w) -> attend over (h*w) positions
          Use case: Capture spatial structures independent of time

        - Temporal only: Each spatial location attends across time
          Shape: (b, c, t, h, w) -> (b*h*w, c, t) -> attend over t positions
          Use case: Capture motion and temporal dynamics

        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)

        Returns:
            Output tensor of same shape with attention-refined features
        """
        # Save input for residual connection and get dimensions
        res, orig_shape = x, x.shape
        b, c, t, h, w = orig_shape

        # Generate queries, keys, values
        qkv = self.to_qkv(x)

        # Reshape for factorized attention if needed
        if self.only_space:
            # Treat each frame independently: (b*t) separate attention operations
            qkv = rearrange(qkv, 'b c t x y -> (b t) c x y')
        elif self.only_time:
            # Treat each spatial location independently: (b*h*w) separate operations
            qkv = rearrange(qkv, 'b c t x y -> (b x y) c t')

        # Split into Q, K, V
        qkv = qkv.chunk(3, dim = 1)

        # Reshape to multi-head format: (batch, heads, positions, dim_per_head)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h (...) c', h = self.heads), qkv)

        # Add learnable memory tokens (global context)
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = k.shape[0]), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # Normalize for magnitude preservation
        q, k, v = map(self.pixel_norm, (q, k, v))

        # Compute attention
        out = self.attend(q, k, v)

        # Merge heads
        out = rearrange(out, 'b h n d -> b (h d) n')

        # Reshape back from factorized attention if needed
        if self.only_space:
            # Combine batch and time back: (b*t, c, h*w) -> (b, c, t*h*w)
            out = rearrange(out, '(b t) c n -> b c (t n)', t = t)
        elif self.only_time:
            # Combine batch and spatial back: (b*h*w, c, t) -> (b, c, t*h*w)
            out = rearrange(out, '(b x y) c n -> b c (n x y)', x = h, y = w)

        # Reshape to original 5D structure
        out = out.reshape(orig_shape)

        # Project to output dimension
        out = self.to_out(out)

        # Add residual with magnitude preservation
        return self.mp_add(out, res)

# Main 3D UNet architecture proposed by Karras
# Bias-less, no group-norms, with magnitude preserving operations

class KarrasUnet3D(Module):
    """
    3D Karras UNet for diffusion models on volumetric data.

    This is a 3D extension of the magnitude-preserving UNet from Karras et al. (2023),
    adapted to process volumetric data like videos, 3D medical images, and scientific
    simulations. The architecture follows Figure 21 Config G from the paper.

    Architecture Overview:
    ---------------------
    - Encoder-decoder UNet structure with skip connections
    - Multiple resolution scales via downsampling/upsampling
    - Self-attention at specified resolutions for long-range dependencies
    - Time and optional class conditioning via embeddings
    - All operations preserve signal magnitude for stable training

    3D-Specific Adaptations:
    -----------------------
    1. **3D Convolutions**: All convs operate on (frames, height, width)
    2. **Flexible Downsampling**: Control spatial vs temporal downsampling independently
    3. **Factorized Attention**: Optional space-time separation for efficiency
    4. **Trilinear Interpolation**: For smooth up/downsampling in all dimensions

    Input/Output:
    ------------
    - Input: 5D tensor (batch, channels, frames, height, width)
    - Time: Scalar timestep for diffusion (batch,)
    - Optional class labels: (batch,) or (batch, num_classes)
    - Output: Denoised 5D tensor, same shape as input

    Typical Use Cases:
    -----------------
    - Video generation and prediction
    - 3D medical image synthesis (CT, MRI)
    - Scientific simulation (fluid dynamics, weather)
    - Any volumetric data generation task

    Key Parameters:
    --------------
    image_size: Spatial resolution (assumes square images)
    frames: Temporal/depth resolution
    dim: Base channel dimension (doubles each downsample up to dim_max)
    num_downsamples: Number of resolution scales
    downsample_types: Control which dimensions to downsample at each stage
        - 'all': Downsample frames, height, width
        - 'image': Downsample height, width only
        - 'frame': Downsample frames only
    attn_res: Resolutions where self-attention is applied
    factorize_space_time_attn: Separate spatial and temporal attention
    num_classes: Enable class conditioning (e.g., 1000 for ImageNet)
    self_condition: Enable self-conditioning (input previous prediction)

    Example:
    -------
    >>> unet = KarrasUnet3D(
    ...     image_size=64,
    ...     frames=32,
    ...     dim=128,
    ...     num_downsamples=3,
    ...     downsample_types=('image', 'frame', 'image'),
    ...     factorize_space_time_attn=True
    ... )
    >>> video = torch.randn(2, 4, 32, 64, 64)  # batch=2, channels=4
    >>> time = torch.rand(2)
    >>> denoised = unet(video, time)
    """

    def __init__(
        self,
        *,
        image_size,
        frames,
        dim = 192,
        dim_max = 768,            # channels will double every downsample and cap out to this value
        num_classes = None,       # in paper, they do 1000 classes for a popular benchmark
        channels = 4,             # 4 channels in paper for some reason, must be alpha channel?
        num_downsamples = 3,
        num_blocks_per_stage: Union[int, Tuple[int, ...]] = 4,
        downsample_types: Optional[Tuple[str, ...]] = None,
        attn_res = (16, 8),
        fourier_dim = 16,
        attn_dim_head = 64,
        attn_flash = False,
        mp_cat_t = 0.5,
        mp_add_emb_t = 0.5,
        attn_res_mp_add_t = 0.3,
        resnet_mp_add_t = 0.3,
        dropout = 0.1,
        self_condition = False,
        factorize_space_time_attn = False
    ):
        """
        Initialize 3D Karras UNet.

        Args:
            image_size: Spatial resolution (height and width, assumes square)
            frames: Number of frames (temporal/depth dimension)
            dim: Base channel dimension (default: 192)
            dim_max: Maximum channel dimension after downsampling (default: 768)
            num_classes: Number of classes for conditioning (None = no class conditioning)
            channels: Number of input/output channels (default: 4)
            num_downsamples: Number of downsampling stages (default: 3)
            num_blocks_per_stage: Encoder/decoder blocks per resolution (int or tuple)
            downsample_types: Type of downsampling at each stage ('all', 'image', 'frame')
            attn_res: Spatial resolutions where attention is applied (default: (16, 8))
            fourier_dim: Dimension of Fourier time embedding (default: 16)
            attn_dim_head: Dimension per attention head (default: 64)
            attn_flash: Use flash attention for efficiency (default: False)
            mp_cat_t: Mixing parameter for magnitude-preserving concat (default: 0.5)
            mp_add_emb_t: Mixing parameter for embedding residuals (default: 0.5)
            attn_res_mp_add_t: Mixing parameter for attention residuals (default: 0.3)
            resnet_mp_add_t: Mixing parameter for conv block residuals (default: 0.3)
            dropout: Dropout probability (default: 0.1)
            self_condition: Enable self-conditioning (default: False)
            factorize_space_time_attn: Use factorized space-time attention (default: False)
        """
        super().__init__()

        self.self_condition = self_condition

        # determine dimensions

        self.channels = channels
        self.frames = frames
        self.image_size = image_size

        input_channels = channels * (2 if self_condition else 1)

        # input and output blocks

        self.input_block = Conv3d(input_channels, dim, 3, concat_ones_to_input = True)

        self.output_block = nn.Sequential(
            Conv3d(dim, channels, 3),
            Gain()
        )

        # time embedding

        emb_dim = dim * 4

        self.to_time_emb = nn.Sequential(
            MPFourierEmbedding(fourier_dim),
            Linear(fourier_dim, emb_dim)
        )

        # class embedding

        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes

        if self.needs_class_labels:
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t = mp_add_emb_t)

        # final embedding activations

        self.emb_activation = MPSiLU()

        # number of downsamples

        self.num_downsamples = num_downsamples

        # specifying downsample types (either image, frames, or both)

        downsample_types = default(downsample_types, 'all')
        downsample_types = cast_tuple(downsample_types, num_downsamples)

        assert len(downsample_types) == num_downsamples
        assert all([t in {'all', 'frame', 'image'} for t in downsample_types])

        # number of blocks per downsample

        num_blocks_per_stage = cast_tuple(num_blocks_per_stage, num_downsamples)

        if len(num_blocks_per_stage) == num_downsamples:
            first, *_ = num_blocks_per_stage
            num_blocks_per_stage = (first, *num_blocks_per_stage)

        assert len(num_blocks_per_stage) == (num_downsamples + 1)
        assert all([num_blocks >= 1 for num_blocks in num_blocks_per_stage])

        # attention

        attn_res = set(cast_tuple(attn_res))

        # resnet block

        block_kwargs = dict(
            dropout = dropout,
            emb_dim = emb_dim,
            attn_dim_head = attn_dim_head,
            attn_res_mp_add_t = attn_res_mp_add_t,
            attn_flash = attn_flash
        )

        # unet encoder and decoders

        self.downs = ModuleList([])
        self.ups = ModuleList([])

        curr_dim = dim
        curr_image_res = image_size
        curr_frame_res = frames

        self.skip_mp_cat = MPCat(t = mp_cat_t, dim = 1)

        # take care of skip connection for initial input block and first three encoder blocks

        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))

        init_num_blocks_per_stage, *rest_num_blocks_per_stage = num_blocks_per_stage

        for _ in range(init_num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)

            append(self.downs, enc)
            prepend(self.ups, dec)

        # stages

        for _, layer_num_blocks_per_stage, layer_downsample_type in zip(range(self.num_downsamples), rest_num_blocks_per_stage, downsample_types):

            dim_out = min(dim_max, curr_dim * 2)

            downsample_image = layer_downsample_type in {'all', 'image'}
            downsample_frame = layer_downsample_type in {'all', 'frame'}

            assert not (downsample_image and not divisible_by(curr_image_res, 2))
            assert not (downsample_frame and not divisible_by(curr_frame_res, 2))

            down_and_upsample_config = (
                downsample_frame,
                downsample_image,
                downsample_image
            )

            upsample = Decoder(
                dim_out,
                curr_dim,
                has_attn = curr_image_res in attn_res,
                upsample = True,
                upsample_config = down_and_upsample_config,
                factorize_space_time_attn = factorize_space_time_attn,
                **block_kwargs
            )

            if downsample_image:
                curr_image_res //= 2

            if downsample_frame:
                curr_frame_res //= 2

            has_attn = curr_image_res in attn_res

            downsample = Encoder(
                curr_dim,
                dim_out,
                downsample = True,
                downsample_config = down_and_upsample_config,
                has_attn = has_attn,
                factorize_space_time_attn = factorize_space_time_attn,
                **block_kwargs
            )

            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs))

            for _ in range(layer_num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn = has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs)

                append(self.downs, enc)
                prepend(self.ups, dec)

            curr_dim = dim_out

        # take care of the two middle decoders

        mid_has_attn = curr_image_res in attn_res

        self.mids = ModuleList([
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
        ])

        self.out_dim = channels

    @property
    def downsample_factor(self):
        """
        Total downsampling factor from input to bottleneck.

        Returns:
            int: 2^num_downsamples (e.g., 8 for 3 downsamples)
        """
        return 2 ** self.num_downsamples

    def forward(
        self,
        x,
        time,
        self_cond = None,
        class_labels = None
    ):
        """
        Forward pass through the 3D UNet.

        Processing pipeline:
        1. Validate input shape
        2. Optionally concatenate self-conditioning (previous prediction)
        3. Embed time via Fourier features
        4. Optionally embed and add class labels
        5. Process through encoder (downsampling path) with skip connections
        6. Process through middle blocks (bottleneck)
        7. Process through decoder (upsampling path) with skip connections
        8. Output final prediction

        Args:
            x: Input volumetric data of shape (batch, channels, frames, height, width)
               Must match the configured channels, frames, and image_size
            time: Diffusion timestep of shape (batch,)
                  Typically in range [0, 1] or [0, num_diffusion_steps]
            self_cond: Optional self-conditioning input (previous prediction)
                      Same shape as x, used only if self_condition=True
            class_labels: Optional class labels of shape (batch,) for integer labels
                         or (batch, num_classes) for one-hot labels
                         Required if num_classes was specified at initialization

        Returns:
            Denoised/predicted output of shape (batch, channels, frames, height, width)
            Same shape as input x

        Example:
            >>> unet = KarrasUnet3D(image_size=64, frames=16, channels=3)
            >>> x = torch.randn(4, 3, 16, 64, 64)  # batch of 4 videos
            >>> t = torch.rand(4)  # random timesteps
            >>> output = unet(x, t)
            >>> output.shape
            torch.Size([4, 3, 16, 64, 64])
        """
        # Validate input shape matches network configuration
        assert x.shape[1:] == (self.channels, self.frames, self.image_size, self.image_size), \
            f"Expected shape (*, {self.channels}, {self.frames}, {self.image_size}, {self.image_size}), got {x.shape}"

        # Self conditioning: concatenate previous prediction if enabled
        if self.self_condition:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim = 1)
        else:
            assert not exists(self_cond), "self_cond provided but self_condition not enabled"

        # Embed time via Fourier features and linear projection
        time_emb = self.to_time_emb(time)

        # Class conditioning: embed and add to time embedding if provided
        assert xnor(exists(class_labels), self.needs_class_labels), \
            "class_labels provided but num_classes not set, or vice versa"

        if self.needs_class_labels:
            # Convert integer labels to one-hot if needed
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)

            assert class_labels.shape[-1] == self.num_classes
            # Scale by sqrt(num_classes) for magnitude preservation
            class_labels = class_labels.float() * sqrt(self.num_classes)

            class_emb = self.to_class_emb(class_labels)

            # Combine time and class embeddings via magnitude-preserving add
            time_emb = self.add_class_emb(time_emb, class_emb)

        # Apply final activation to combined embedding
        emb = self.emb_activation(time_emb)

        # Skip connections storage for U-Net
        skips = []

        # Input block: initial 3x3x3 conv
        x = self.input_block(x)
        skips.append(x)

        # Encoder: downsampling path with skip connection storage
        for encoder in self.downs:
            x = encoder(x, emb = emb)
            skips.append(x)

        # Middle: bottleneck processing at lowest resolution
        for decoder in self.mids:
            x = decoder(x, emb = emb)

        # Decoder: upsampling path with skip connections
        for decoder in self.ups:
            if decoder.needs_skip:
                # Retrieve and concatenate skip connection with magnitude preservation
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)

            x = decoder(x, emb = emb)

        # Output block: final 3x3x3 conv to get back to input channels
        return self.output_block(x)

# Magnitude-Preserving Transformer components (experimental)

class MPFeedForward(Module):
    """
    Magnitude-Preserving Feed-Forward Network for transformers.

    A simple 2-layer MLP with magnitude preservation, following the transformer
    architecture pattern but adapted for the Karras magnitude-preserving approach.

    Architecture:
    1. Pixel normalization
    2. Expand to inner dimension (dim * mult) via 1x1x1 conv
    3. MP-SiLU activation
    4. Project back to original dimension via 1x1x1 conv
    5. Add residual with magnitude preservation

    This can be used as an alternative to or in combination with the standard
    UNet encoder/decoder blocks.

    Args:
        dim: Input/output channel dimension
        mult: Expansion multiplier for hidden dimension (default: 4)
        mp_add_t: Mixing weight for residual connection (default: 0.3)
    """
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        mp_add_t = 0.3
    ):
        """
        Initialize magnitude-preserving feed-forward network.

        Args:
            dim: Channel dimension
            mult: Hidden dimension multiplier
            mp_add_t: Residual mixing weight
        """
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            PixelNorm(dim = 1),
            Conv3d(dim, dim_inner, 1),
            MPSiLU(),
            Conv3d(dim_inner, dim, 1)
        )

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        """
        Apply feed-forward transformation with residual.

        Args:
            x: Input tensor of shape (batch, dim, frames, height, width)

        Returns:
            Output tensor of same shape
        """
        res = x
        out = self.net(x)
        return self.mp_add(out, res)

class MPImageTransformer(Module):
    """
    Magnitude-Preserving Vision Transformer for 3D volumetric data.

    A transformer-style architecture using magnitude-preserving operations.
    Each layer consists of multi-head self-attention followed by feed-forward,
    both with magnitude-preserving residual connections.

    This can serve as an alternative to the convolutional UNet architecture,
    or can be used within the UNet as a processing block.

    Architecture per layer:
    1. Multi-head self-attention with MP residual
    2. Feed-forward network with MP residual

    3D Capability:
    - Processes full 3D volumes via the Attention module
    - Can use factorized space-time attention for efficiency
    - Useful for global context modeling in volumetric data

    Args:
        dim: Channel dimension throughout the transformer
        depth: Number of transformer layers
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 8)
        num_mem_kv: Number of memory key-value pairs (default: 4)
        ff_mult: Feed-forward expansion multiplier (default: 4)
        attn_flash: Use flash attention (default: False)
        residual_mp_add_t: Mixing weight for all residuals (default: 0.3)
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_mem_kv = 4,
        ff_mult = 4,
        attn_flash = False,
        residual_mp_add_t = 0.3
    ):
        """
        Initialize magnitude-preserving transformer.

        Args:
            dim: Channel dimension
            depth: Number of transformer layers
            dim_head: Dimension per attention head
            heads: Number of attention heads
            num_mem_kv: Number of memory tokens
            ff_mult: Feed-forward hidden dimension multiplier
            attn_flash: Use flash attention
            residual_mp_add_t: Residual connection mixing weight
        """
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, num_mem_kv = num_mem_kv, flash = attn_flash, mp_add_t = residual_mp_add_t),
                MPFeedForward(dim = dim, mult = ff_mult, mp_add_t = residual_mp_add_t)
            ]))

    def forward(self, x):
        """
        Process input through transformer layers.

        Args:
            x: Input tensor of shape (batch, dim, frames, height, width)

        Returns:
            Transformed tensor of same shape
        """
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

# Example usage demonstrating 3D UNet for video/volumetric data

if __name__ == '__main__':
    """
    Example: 3D Karras UNet for class-conditional video generation.

    This example demonstrates:
    - Video processing (32 frames, 64x64 resolution)
    - Alternating spatial and temporal downsampling
    - Class conditioning (1000 classes, e.g., for action recognition)
    - Factorized space-time attention for efficiency
    - 6 downsampling stages for multi-scale processing
    """

    # Initialize 3D UNet with video-specific configuration
    unet = KarrasUnet3D(
        frames = 32,              # Temporal dimension (number of video frames)
        image_size = 64,          # Spatial resolution (64x64 per frame)
        dim = 8,                  # Base channel dimension (kept small for example)
        dim_max = 768,            # Maximum channels after downsampling
        num_downsamples = 6,      # 6 resolution scales (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        num_blocks_per_stage = (4, 3, 2, 2, 2, 2),  # Decreasing blocks at lower resolutions
        downsample_types = (
            'image',   # Stage 1: Downsample spatial only (preserve temporal resolution)
            'frame',   # Stage 2: Downsample temporal only (preserve spatial resolution)
            'image',   # Stage 3: Downsample spatial
            'frame',   # Stage 4: Downsample temporal
            'image',   # Stage 5: Downsample spatial
            'frame',   # Stage 6: Downsample temporal
        ),
        attn_dim_head = 8,        # Small attention heads for this example
        num_classes = 1000,       # Class-conditional (e.g., 1000 action classes)
        factorize_space_time_attn = True  # Separate spatial and temporal attention for efficiency
    )

    # Create example video input: (batch=2, channels=4, frames=32, height=64, width=64)
    video = torch.randn(2, 4, 32, 64, 64)

    # Run denoising with time and class conditioning
    denoised_video = unet(
        video,
        time = torch.ones(2,),                    # Time step for diffusion (same for both samples)
        class_labels = torch.randint(0, 1000, (2,))  # Random class labels
    )

    print(f"Input shape: {video.shape}")
    print(f"Output shape: {denoised_video.shape}")
    print(f"Shape matches: {video.shape == denoised_video.shape}")
