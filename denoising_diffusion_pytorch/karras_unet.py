"""
Karras et al. Magnitude-Preserving UNet Architecture

This module implements the magnitude-preserving UNet architecture from the paper:
"Elucidating the Design Space of Diffusion-Based Generative Models"
by Tero Karras et al. (2023) - https://arxiv.org/abs/2312.02696

Key Innovations from the Paper:
-----------------------------

1. Magnitude Preservation (MP):
   - All operations are designed to preserve signal magnitude throughout the network
   - Removes the need for normalization layers (GroupNorm, BatchNorm)
   - Ensures stable training dynamics without biases
   - Based on the principle that signal variance should remain constant

2. Weight Normalization:
   - All linear and convolutional layers use forced weight normalization (Algorithm 1)
   - Weights are normalized at each forward pass to unit norm
   - Scaled by 1/sqrt(fan_in) to maintain magnitude

3. Magnitude-Preserving Operations:
   - MPSiLU: Modified SiLU activation that preserves magnitude
   - MPAdd: Magnitude-preserving residual connections (Equation 88)
   - MPCat: Magnitude-preserving concatenation (Equation 103)
   - PixelNorm: Per-pixel normalization (Equation 30)

4. Fourier Features for Time Conditioning:
   - Uses learnable Fourier embeddings for noise level conditioning
   - More expressive than sinusoidal embeddings

5. Bias-Free Architecture:
   - All biases removed from convolutional and linear layers
   - Input block uses concatenated ones to preserve expressivity

6. FIR (Finite Impulse Response) Filtering:
   - Uses bilinear interpolation for up/downsampling
   - Provides smoother feature transitions

7. Architecture Improvements:
   - Attention at specific resolutions (typically 16x16 and 8x8)
   - Memory key-values in attention for increased capacity
   - Gain layers for learnable residual scaling

This implementation follows "Config G" from Figure 21 of the paper, which
demonstrated state-of-the-art results on ImageNet and other benchmarks.
"""

import math
from math import sqrt, ceil
from functools import partial

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

from denoising_diffusion_pytorch.attend import Attend

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
        val: Value to check
        d: Default value (can be a callable or a value)

    Returns:
        val if it exists, otherwise d() if d is callable, else d
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def xnor(x, y):
    """
    Logical XNOR operation - returns True if x and y have the same truth value.

    Args:
        x: First boolean value
        y: Second boolean value

    Returns:
        bool: True if both are True or both are False
    """
    return not (x ^ y)

def append(arr, el):
    """
    Append element to end of array (in-place).

    Args:
        arr: List to append to
        el: Element to append
    """
    arr.append(el)

def prepend(arr, el):
    """
    Prepend element to beginning of array (in-place).

    Args:
        arr: List to prepend to
        el: Element to prepend
    """
    arr.insert(0, el)

def pack_one(t, pattern):
    """
    Pack a single tensor using einops pattern.

    Args:
        t: Tensor to pack
        pattern: Einops pattern string

    Returns:
        Packed tensor
    """
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """
    Unpack a single tensor using einops pattern.

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
    Cast value to tuple of specified length.

    Args:
        t: Value to cast (tuple or single value)
        length: Desired tuple length

    Returns:
        tuple: Original tuple if t is already a tuple, otherwise (t,) * length
    """
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    """
    Check if numerator is evenly divisible by denominator.

    Args:
        numer: Numerator
        denom: Denominator

    Returns:
        bool: True if numer % denom == 0
    """
    return (numer % denom) == 0

# in paper, they use eps 1e-4 for pixelnorm

def l2norm(t, dim = -1, eps = 1e-12):
    """
    L2 normalization along specified dimension.

    Args:
        t: Input tensor
        dim: Dimension to normalize along (default: -1)
        eps: Small constant for numerical stability (default: 1e-12)

    Returns:
        Normalized tensor with unit L2 norm along specified dimension
    """
    return F.normalize(t, dim = dim, eps = eps)

# mp activations
# section 2.5

class MPSiLU(Module):
    """
    Magnitude-Preserving SiLU (Swish) Activation.

    Implements a magnitude-preserving variant of the SiLU activation function.
    The standard SiLU reduces signal magnitude, so this divides by 0.596 to
    compensate and maintain unit variance.

    From Section 2.5 of the Karras paper. The scaling factor 0.596 is computed
    such that if the input has unit variance, the output also has unit variance.

    Forward pass:
        output = SiLU(x) / 0.596 = (x * sigmoid(x)) / 0.596
    """
    def forward(self, x):
        """
        Args:
            x: Input tensor of any shape

        Returns:
            Magnitude-preserved SiLU activation of x
        """
        return F.silu(x) / 0.596

# gain - layer scaling

class Gain(Module):
    """
    Learnable Gain Layer for Output Scaling.

    Multiplies input by a learnable scalar parameter, initialized to 0.
    This is used in the output block to allow the network to learn the
    appropriate scaling for its predictions. Starting from 0 means the
    network initially outputs zeros and gradually learns the correct scale.

    The gain parameter is learned during training and allows for adaptive
    scaling of residual connections or output predictions.
    """
    def __init__(self):
        """
        Initialize Gain layer with parameter set to 0.
        """
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        """
        Args:
            x: Input tensor of any shape

        Returns:
            Tensor scaled by learnable gain parameter
        """
        return x * self.gain

# magnitude preserving concat
# equation (103) - default to 0.5, which they recommended

class MPCat(Module):
    """
    Magnitude-Preserving Concatenation.

    Implements Equation 103 from the Karras paper. Standard concatenation
    increases the total variance when combining tensors. This operation
    rescales the concatenated tensors to preserve unit variance.

    The operation balances the contributions from both tensors using a
    weighting parameter t, and applies a correction factor C to ensure
    the output has the same expected variance as the inputs.

    Args:
        t: Weighting parameter between 0 and 1 (default: 0.5)
           Controls the relative contribution of each tensor
        dim: Dimension along which to concatenate (default: -1)

    Mathematical formulation:
        C = sqrt((Na + Nb) / ((1-t)^2 + t^2))
        output = C * concat(a * (1-t) / sqrt(Na), b * t / sqrt(Nb))

    where Na and Nb are the dimensions of tensors a and b along dim.
    """
    def __init__(self, t = 0.5, dim = -1):
        """
        Args:
            t: Weighting parameter (default: 0.5 as recommended in paper)
            dim: Concatenation dimension (default: -1)
        """
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        """
        Concatenate two tensors while preserving magnitude.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            Magnitude-preserved concatenation of a and b along dim
        """
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        C = sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

        a = a * (1. - t) / sqrt(Na)
        b = b * t / sqrt(Nb)

        return C * torch.cat((a, b), dim = dim)

# magnitude preserving sum
# equation (88)
# empirically, they found t=0.3 for encoder / decoder / attention residuals
# and for embedding, t=0.5

class MPAdd(Module):
    """
    Magnitude-Preserving Addition for Residual Connections.

    Implements Equation 88 from the Karras paper. Standard residual addition
    (x + residual) increases variance. This operation combines two tensors
    while preserving unit variance through careful weighting and normalization.

    Used for all residual connections in the network. The weighting parameter t
    controls the balance between the main path and residual:
    - t=0.3: Used for encoder/decoder/attention residuals (empirically optimal)
    - t=0.5: Used for embedding additions

    Mathematical formulation:
        output = (a * (1-t) + b * t) / sqrt((1-t)^2 + t^2)

    This ensures that if inputs have unit variance, output also has unit variance.

    Args:
        t: Weighting parameter between 0 and 1
           Higher t gives more weight to the residual
    """
    def __init__(self, t):
        """
        Args:
            t: Weighting parameter for residual connection
        """
        super().__init__()
        self.t = t

    def forward(self, x, res):
        """
        Add two tensors while preserving magnitude.

        Args:
            x: Main path tensor
            res: Residual tensor to add

        Returns:
            Magnitude-preserved sum of x and res
        """
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

# pixelnorm
# equation (30)

class PixelNorm(Module):
    """
    Pixel Normalization Layer.

    Implements Equation 30 from the Karras paper. Normalizes activations
    along a specified dimension to unit norm, then scales by sqrt(dim_size)
    to preserve the expected magnitude.

    Unlike standard normalization layers (LayerNorm, BatchNorm), PixelNorm:
    - Normalizes to unit L2 norm (not zero mean, unit variance)
    - Uses higher epsilon (1e-4) for stability as recommended in paper
    - Preserves magnitude through sqrt(dim_size) scaling

    This is applied after convolutions and before attention to maintain
    magnitude preservation throughout the network.

    Args:
        dim: Dimension to normalize along (e.g., 1 for channels)
        eps: Epsilon for numerical stability (default: 1e-4, higher than usual)
    """
    def __init__(self, dim, eps = 1e-4):
        """
        Args:
            dim: Dimension index to normalize
            eps: Small constant for stability (paper uses 1e-4)
        """
        super().__init__()
        # high epsilon for the pixel norm in the paper
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        """
        Apply pixel normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor with preserved magnitude
        """
        dim = self.dim
        return l2norm(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

# forced weight normed conv2d and linear
# algorithm 1 in paper

def normalize_weight(weight, eps = 1e-4):
    """
    Weight Normalization following Algorithm 1 from the Karras paper.

    Normalizes weight tensor to unit norm per output channel, then scales
    by sqrt(fan_in) to preserve magnitude. This is a key component of the
    magnitude-preserving architecture.

    The normalization:
    1. Flattens weight to (output_channels, fan_in)
    2. L2 normalizes each output channel to unit norm
    3. Scales by sqrt(fan_in) to preserve expected variance

    This ensures that regardless of initialization, weights have consistent
    magnitude and the network maintains stable gradients.

    Args:
        weight: Weight tensor of shape (out_channels, in_channels, *kernel_size)
                or (out_features, in_features) for linear layers
        eps: Epsilon for numerical stability in normalization (default: 1e-4)

    Returns:
        Normalized weight tensor with same shape as input
    """
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

class Conv2d(Module):
    """
    Magnitude-Preserving 2D Convolution Layer.

    Implements bias-free convolution with forced weight normalization.
    All weights are normalized at every forward pass to maintain magnitude
    preservation throughout the network.

    Key features:
    - No bias term (removed for magnitude preservation)
    - Weight normalization applied every forward pass
    - Weights scaled by 1/sqrt(fan_in) to preserve variance
    - During training, weights are normalized in-place for efficiency
    - Optional concatenation of ones channel for input expressivity

    The concat_ones_to_input option is used only in the first layer to
    compensate for potential loss of expressivity from removing biases,
    though the paper found minimal impact.

    Args:
        dim_in: Number of input channels
        dim_out: Number of output channels
        kernel_size: Size of the convolutional kernel
        eps: Epsilon for weight normalization (default: 1e-4)
        concat_ones_to_input: If True, concatenate channel of ones to input
                              (used only in input block)
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
        Initialize magnitude-preserving Conv2d layer.

        Args:
            dim_in: Input channels
            dim_out: Output channels
            kernel_size: Convolution kernel size
            eps: Normalization epsilon
            concat_ones_to_input: Whether to add ones channel to input
        """
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        """
        Forward pass with weight normalization.

        During training, weights are normalized in-place for efficiency.
        The normalization is applied again for the actual computation
        and scaled by 1/sqrt(fan_in).

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Convolved output with same spatial dimensions (padding='same')
        """

        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value = 1.)

        return F.conv2d(x, weight, padding='same')

class Linear(Module):
    """
    Magnitude-Preserving Linear Layer.

    Implements bias-free fully-connected layer with forced weight normalization.
    Similar to Conv2d but for linear transformations.

    Key features:
    - No bias term (removed for magnitude preservation)
    - Weight normalization applied every forward pass
    - Weights scaled by 1/sqrt(fan_in) to preserve variance
    - During training, weights are normalized in-place for efficiency

    Used for embedding projections and attention transformations.

    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        eps: Epsilon for weight normalization (default: 1e-4)
    """
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        """
        Initialize magnitude-preserving Linear layer.

        Args:
            dim_in: Input features
            dim_out: Output features
            eps: Normalization epsilon
        """
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        """
        Forward pass with weight normalization.

        Args:
            x: Input tensor of shape (batch, ..., dim_in)

        Returns:
            Output tensor of shape (batch, ..., dim_out)
        """
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

# mp fourier embeds

class MPFourierEmbedding(Module):
    """
    Magnitude-Preserving Fourier Feature Embedding.

    Embeds scalar inputs (e.g., noise levels, timesteps) into higher-dimensional
    space using random Fourier features. This is more expressive than standard
    sinusoidal embeddings used in original transformer/diffusion models.

    The embedding uses learnable (but frozen) random frequencies to project
    the input through sine and cosine functions, creating a rich representation
    of the continuous input value.

    Process:
    1. Generate random frequencies (frozen, not trainable)
    2. Compute freqs = input * frequencies * 2π
    3. Return [sin(freqs), cos(freqs)] scaled by sqrt(2)

    The sqrt(2) scaling preserves magnitude: each component (sin/cos) has
    variance 0.5, so their concatenation has total variance 1.0.

    Used for time/noise conditioning in the UNet.

    Args:
        dim: Output embedding dimension (must be even)
    """
    def __init__(self, dim):
        """
        Initialize Fourier embedding with random frequencies.

        Args:
            dim: Embedding dimension (must be divisible by 2)
        """
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        """
        Embed scalar input using Fourier features.

        Args:
            x: Input tensor of shape (batch,) containing scalar values

        Returns:
            Embedded tensor of shape (batch, dim) with Fourier features
        """
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)

# building block modules

class Encoder(Module):
    """
    Encoder Block for the Karras UNet.

    A residual encoder block that optionally downsamples the input and applies
    self-attention. The block consists of:

    1. Optional downsampling via bilinear interpolation + 1x1 conv
    2. Pixel normalization
    3. First conv block: MPSiLU -> Conv2d
    4. Optional time/class embedding injection (via adaptive scaling)
    5. Second conv block: MPSiLU -> Dropout -> Conv2d
    6. Magnitude-preserving residual addition
    7. Optional self-attention

    All operations maintain magnitude preservation for stable training.

    Args:
        dim: Input channels
        dim_out: Output channels (defaults to dim)
        emb_dim: Embedding dimension for time/class conditioning (if None, no conditioning)
        dropout: Dropout probability (default: 0.1)
        mp_add_t: MPAdd weighting for residual connection (default: 0.3)
        has_attn: Whether to include self-attention (default: False)
        attn_dim_head: Dimension per attention head (default: 64)
        attn_res_mp_add_t: MPAdd weighting for attention residual (default: 0.3)
        attn_flash: Use flash attention if available (default: False)
        downsample: Whether to downsample spatially by 2x (default: False)
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
        downsample = False
    ):
        """
        Initialize Encoder block.

        Args:
            dim: Input channel dimension
            dim_out: Output channel dimension
            emb_dim: Time/class embedding dimension
            dropout: Dropout rate
            mp_add_t: Magnitude-preserving add parameter
            has_attn: Include attention layer
            attn_dim_head: Attention head dimension
            attn_res_mp_add_t: Attention residual MPAdd parameter
            attn_flash: Use flash attention
            downsample: Apply 2x spatial downsampling
        """
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv2d(curr_dim, dim_out, 1)
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
            Conv2d(curr_dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    def forward(
        self,
        x,
        emb = None
    ):
        """
        Forward pass through encoder block.

        Processing steps:
        1. Downsample if configured (bilinear interpolation + 1x1 conv)
        2. Apply pixel normalization
        3. Save residual connection
        4. First conv block (activation + conv)
        5. Apply time/class conditioning via adaptive feature modulation
        6. Second conv block (activation + dropout + conv)
        7. Add residual with magnitude preservation
        8. Apply self-attention if configured

        Args:
            x: Input feature map of shape (batch, channels, height, width)
            emb: Optional conditioning embedding of shape (batch, emb_dim)

        Returns:
            Processed feature map, potentially downsampled and with attention applied
        """
        if self.downsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h // 2, w // 2), mode = 'bilinear')
            x = self.downsample_conv(x)

        x = self.pixel_norm(x)

        res = x.clone()

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

class Decoder(Module):
    """
    Decoder Block for the Karras UNet.

    A residual decoder block that optionally upsamples the input and applies
    self-attention. The block is structurally similar to Encoder but:
    - Upsamples instead of downsampling (when upsample=True)
    - Always processes skip connections from encoder (when needs_skip=True)
    - No pixel norm at input (since skip concat already normalized)

    The block consists of:
    1. Optional upsampling via bilinear interpolation
    2. First conv block: MPSiLU -> Conv2d
    3. Optional time/class embedding injection (via adaptive scaling)
    4. Second conv block: MPSiLU -> Dropout -> Conv2d
    5. Magnitude-preserving residual addition
    6. Optional self-attention

    Args:
        dim: Input channels
        dim_out: Output channels (defaults to dim)
        emb_dim: Embedding dimension for time/class conditioning (if None, no conditioning)
        dropout: Dropout probability (default: 0.1)
        mp_add_t: MPAdd weighting for residual connection (default: 0.3)
        has_attn: Whether to include self-attention (default: False)
        attn_dim_head: Dimension per attention head (default: 64)
        attn_res_mp_add_t: MPAdd weighting for attention residual (default: 0.3)
        attn_flash: Use flash attention if available (default: False)
        upsample: Whether to upsample spatially by 2x (default: False)
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
        upsample = False
    ):
        """
        Initialize Decoder block.

        Args:
            dim: Input channel dimension
            dim_out: Output channel dimension
            emb_dim: Time/class embedding dimension
            dropout: Dropout rate
            mp_add_t: Magnitude-preserving add parameter
            has_attn: Include attention layer
            attn_dim_head: Attention head dimension
            attn_res_mp_add_t: Attention residual MPAdd parameter
            attn_flash: Use flash attention
            upsample: Apply 2x spatial upsampling
        """
        super().__init__()
        dim_out = default(dim_out, dim)

        self.upsample = upsample
        self.needs_skip = not upsample

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    def forward(
        self,
        x,
        emb = None
    ):
        """
        Forward pass through decoder block.

        Processing steps:
        1. Upsample if configured (bilinear interpolation 2x)
        2. Save residual connection (with channel adjustment if needed)
        3. First conv block (activation + conv)
        4. Apply time/class conditioning via adaptive feature modulation
        5. Second conv block (activation + dropout + conv)
        6. Add residual with magnitude preservation
        7. Apply self-attention if configured

        Args:
            x: Input feature map of shape (batch, channels, height, width)
            emb: Optional conditioning embedding of shape (batch, emb_dim)

        Returns:
            Processed feature map, potentially upsampled and with attention applied
        """
        if self.upsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h * 2, w * 2), mode = 'bilinear')

        res = self.res_conv(x)

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

# attention

class Attention(Module):
    """
    Magnitude-Preserving Multi-Head Self-Attention.

    Implements self-attention with magnitude preservation and memory keys/values.
    Key features from the Karras paper:

    1. Pixel normalization of Q, K, V for magnitude preservation
    2. Memory keys/values: Additional learnable K/V pairs that attend to all positions
       - Increases model capacity without additional input processing
       - Allows attention to learn global context patterns
    3. Magnitude-preserving residual connection
    4. Optional flash attention for efficiency

    The attention mechanism:
    - Projects input to Q, K, V using 1x1 convolutions
    - Flattens spatial dimensions to sequence
    - Concatenates learnable memory K/V to keys and values
    - Applies pixel normalization to Q, K, V
    - Computes attention and projects back to spatial format
    - Adds to input via MPAdd

    Args:
        dim: Input/output feature dimension
        heads: Number of attention heads (default: 4)
        dim_head: Dimension per head (default: 64)
        num_mem_kv: Number of memory key/value pairs per head (default: 4)
        flash: Use flash attention implementation (default: False)
        mp_add_t: MPAdd parameter for residual connection (default: 0.3)
    """
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        flash = False,
        mp_add_t = 0.3
    ):
        """
        Initialize attention module.

        Args:
            dim: Channel dimension
            heads: Number of attention heads
            dim_head: Dimension per attention head
            num_mem_kv: Number of learnable memory key-value pairs
            flash: Use flash attention
            mp_add_t: Residual connection MPAdd parameter
        """
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim = -1)

        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1)
        self.to_out = Conv2d(hidden_dim, dim, 1)

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        """
        Apply self-attention with magnitude preservation.

        Processing steps:
        1. Save residual connection
        2. Project to Q, K, V via 1x1 conv
        3. Reshape spatial (H, W) to sequence dimension
        4. Prepend learnable memory K/V to keys and values
        5. Apply pixel normalization to Q, K, V
        6. Compute attention (softmax(QK^T)V)
        7. Reshape back to spatial format
        8. Project output via 1x1 conv
        9. Add to residual with magnitude preservation

        Args:
            x: Input feature map of shape (batch, channels, height, width)

        Returns:
            Attention output added to input via MPAdd
        """
        res, b, c, h, w = x, *x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        return self.mp_add(out, res)

# unet proposed by karras
# bias-less, no group-norms, with magnitude preserving operations

class KarrasUnet(Module):
    """
    Magnitude-Preserving UNet from Karras et al. (2023).

    This is the main UNet architecture implementing "Config G" from Figure 21 of
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Architecture Overview:
    ---------------------
    The UNet follows an encoder-decoder structure with skip connections:

    1. Input Block: Conv2d with concatenated ones channel
    2. Encoder: Multiple residual blocks with optional attention and downsampling
       - Blocks per stage at same resolution
       - Downsample by 2x between stages via bilinear interpolation
       - Channels double at each downsample (up to dim_max)
    3. Middle: Two decoder blocks at lowest resolution
    4. Decoder: Mirror of encoder with upsampling and skip connections
       - Skip connections use MPCat for magnitude preservation
       - Upsample by 2x via bilinear interpolation
    5. Output Block: Conv2d + Gain layer

    Key Features:
    -------------
    - Magnitude Preservation: All operations preserve signal magnitude
    - No Biases: All conv/linear layers are bias-free
    - No Normalization Layers: No BatchNorm, GroupNorm, or LayerNorm
    - Fourier Time Embedding: Rich time/noise conditioning
    - Optional Class Conditioning: One-hot class labels scaled by sqrt(num_classes)
    - Attention at Specific Resolutions: Typically 16x16 and 8x8
    - Self-Conditioning: Optional input of previous prediction

    The architecture ensures stable training through:
    - Consistent signal magnitudes via MP operations
    - Proper weight initialization and normalization
    - Careful residual connection weighting
    - Adaptive feature modulation for conditioning

    Args:
        image_size: Input image spatial dimension (assumed square)
        dim: Base channel dimension (default: 192)
        dim_max: Maximum channels after downsampling (default: 768)
        num_classes: Number of classes for conditioning (None for unconditional)
        channels: Number of input/output channels (default: 4)
        num_downsamples: Number of 2x downsampling stages (default: 3)
        num_blocks_per_stage: Residual blocks at each resolution (default: 4)
        attn_res: Resolutions where attention is applied (default: (16, 8))
        fourier_dim: Fourier embedding dimension for time (default: 16)
        attn_dim_head: Dimension per attention head (default: 64)
        attn_flash: Use flash attention (default: False)
        mp_cat_t: MPCat parameter for skip connections (default: 0.5)
        mp_add_emb_t: MPAdd parameter for embedding addition (default: 0.5)
        attn_res_mp_add_t: MPAdd parameter for attention residuals (default: 0.3)
        resnet_mp_add_t: MPAdd parameter for conv residuals (default: 0.3)
        dropout: Dropout probability (default: 0.1)
        self_condition: Enable self-conditioning input (default: False)
    """

    def __init__(
        self,
        *,
        image_size,
        dim = 192,
        dim_max = 768,            # channels will double every downsample and cap out to this value
        num_classes = None,       # in paper, they do 1000 classes for a popular benchmark
        channels = 4,             # 4 channels in paper for some reason, must be alpha channel?
        num_downsamples = 3,
        num_blocks_per_stage = 4,
        attn_res = (16, 8),
        fourier_dim = 16,
        attn_dim_head = 64,
        attn_flash = False,
        mp_cat_t = 0.5,
        mp_add_emb_t = 0.5,
        attn_res_mp_add_t = 0.3,
        resnet_mp_add_t = 0.3,
        dropout = 0.1,
        self_condition = False
    ):
        super().__init__()

        self.self_condition = self_condition

        # determine dimensions

        self.channels = channels
        self.image_size = image_size
        input_channels = channels * (2 if self_condition else 1)

        # input and output blocks

        self.input_block = Conv2d(input_channels, dim, 3, concat_ones_to_input = True)

        self.output_block = nn.Sequential(
            Conv2d(dim, channels, 3),
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
        curr_res = image_size

        self.skip_mp_cat = MPCat(t = mp_cat_t, dim = 1)

        # take care of skip connection for initial input block and first three encoder blocks

        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))

        assert num_blocks_per_stage >= 1

        for _ in range(num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)

            append(self.downs, enc)
            prepend(self.ups, dec)

        # stages

        for _ in range(self.num_downsamples):
            dim_out = min(dim_max, curr_dim * 2)
            upsample = Decoder(dim_out, curr_dim, has_attn = curr_res in attn_res, upsample = True, **block_kwargs)

            curr_res //= 2
            has_attn = curr_res in attn_res

            downsample = Encoder(curr_dim, dim_out, downsample = True, has_attn = has_attn, **block_kwargs)

            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs))

            for _ in range(num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn = has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs)

                append(self.downs, enc)
                prepend(self.ups, dec)

            curr_dim = dim_out

        # take care of the two middle decoders

        mid_has_attn = curr_res in attn_res

        self.mids = ModuleList([
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
        ])

        self.out_dim = channels

    @property
    def downsample_factor(self):
        """
        Total spatial downsampling factor of the UNet.

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
        Forward pass through the UNet.

        Processing pipeline:
        1. Validate input shape
        2. Apply self-conditioning if enabled (concatenate previous prediction)
        3. Embed time/noise level via Fourier features
        4. Embed class labels if provided (one-hot scaled by sqrt(num_classes))
        5. Combine time and class embeddings via MPAdd
        6. Process through input block
        7. Encode through downsampling path, saving skip connections
        8. Process through middle blocks at lowest resolution
        9. Decode through upsampling path, using skip connections with MPCat
        10. Generate output via output block with Gain

        Args:
            x: Input images of shape (batch, channels, image_size, image_size)
            time: Noise level or timestep of shape (batch,)
            self_cond: Previous prediction for self-conditioning (batch, channels, H, W)
                       Only used if self_condition=True during initialization
            class_labels: Class labels of shape (batch,) as integers or
                         (batch, num_classes) as one-hot vectors
                         Only used if num_classes was specified during initialization

        Returns:
            Denoised prediction of shape (batch, channels, image_size, image_size)
        """
        # validate image shape

        assert x.shape[1:] == (self.channels, self.image_size, self.image_size)

        # self conditioning

        if self.self_condition:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim = 1)
        else:
            assert not exists(self_cond)

        # time condition

        time_emb = self.to_time_emb(time)

        # class condition

        assert xnor(exists(class_labels), self.needs_class_labels)

        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)

            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * sqrt(self.num_classes)

            class_emb = self.to_class_emb(class_labels)

            time_emb = self.add_class_emb(time_emb, class_emb)

        # final mp-silu for embedding

        emb = self.emb_activation(time_emb)

        # skip connections

        skips = []

        # input block

        x = self.input_block(x)

        skips.append(x)

        # down

        for encoder in self.downs:
            x = encoder(x, emb = emb)
            skips.append(x)

        # mid

        for decoder in self.mids:
            x = decoder(x, emb = emb)

        # up

        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)

            x = decoder(x, emb = emb)

        # output block

        return self.output_block(x)

# improvised MP Transformer

class MPFeedForward(Module):
    """
    Magnitude-Preserving Feedforward Network.

    A simple two-layer MLP with magnitude preservation, used in transformer blocks.
    Follows the standard transformer feedforward pattern but with MP operations:

    Architecture:
    - PixelNorm for input normalization
    - 1x1 Conv expanding to mult * dim (default 4x expansion)
    - MPSiLU activation
    - 1x1 Conv projecting back to dim
    - MPAdd for residual connection

    This is an improvised MP variant not from the original Karras paper,
    but follows the same magnitude preservation principles.

    Args:
        dim: Input/output channel dimension
        mult: Expansion factor for hidden dimension (default: 4)
        mp_add_t: MPAdd parameter for residual (default: 0.3)
    """
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        mp_add_t = 0.3
    ):
        """
        Initialize feedforward network.

        Args:
            dim: Channel dimension
            mult: Hidden dimension multiplier
            mp_add_t: Residual MPAdd parameter
        """
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            PixelNorm(dim = 1),
            Conv2d(dim, dim_inner, 1),
            MPSiLU(),
            Conv2d(dim_inner, dim, 1)
        )

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        """
        Forward pass through feedforward network.

        Args:
            x: Input tensor of shape (batch, dim, height, width)

        Returns:
            Output with residual connection via MPAdd
        """
        res = x
        out = self.net(x)
        return self.mp_add(out, res)

class MPImageTransformer(Module):
    """
    Magnitude-Preserving Image Transformer.

    A stack of transformer layers (attention + feedforward) with magnitude preservation.
    Each layer consists of:
    1. Self-attention with memory keys/values
    2. Feedforward network

    Both components use magnitude-preserving operations and residual connections.
    This is an improvised extension applying MP principles to vision transformers,
    not from the original Karras paper.

    Can be used as an alternative to or in combination with convolutional blocks
    for processing image features while maintaining magnitude preservation.

    Args:
        dim: Channel dimension for all layers
        depth: Number of transformer layers
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 8)
        num_mem_kv: Memory key-value pairs per head (default: 4)
        ff_mult: Feedforward hidden dimension multiplier (default: 4)
        attn_flash: Use flash attention (default: False)
        residual_mp_add_t: MPAdd parameter for all residuals (default: 0.3)
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
        Initialize transformer with specified depth and architecture.

        Args:
            dim: Channel dimension
            depth: Number of transformer blocks
            dim_head: Attention head dimension
            heads: Number of attention heads
            num_mem_kv: Number of memory K/V pairs
            ff_mult: Feedforward expansion multiplier
            attn_flash: Use flash attention
            residual_mp_add_t: Residual connection MPAdd parameter
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
        Forward pass through all transformer layers.

        Args:
            x: Input feature map of shape (batch, dim, height, width)

        Returns:
            Transformed features of same shape as input
        """

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

# works best with inverse square root decay schedule

def InvSqrtDecayLRSched(
    optimizer,
    t_ref = 70000,
    sigma_ref = 0.01
):
    """
    Inverse Square Root Learning Rate Decay Scheduler.

    Implements the learning rate schedule from Equation 67 and Table 1 of the
    Karras et al. paper. This scheduler works particularly well with the
    magnitude-preserving architecture.

    Learning rate formula:
        lr(t) = sigma_ref / sqrt(max(t / t_ref, 1))

    Where:
    - t is the current training step
    - t_ref is a reference step (typically 70000 in the paper)
    - sigma_ref is the reference learning rate (typically 0.01)

    Schedule behavior:
    - For t < t_ref: Learning rate decays as 1/sqrt(t/t_ref)
    - For t >= t_ref: Learning rate continues to decay as 1/sqrt(t/t_ref)
    - At t = t_ref: Learning rate equals sigma_ref
    - At t = 0: Learning rate is at maximum (sigma_ref)

    This provides a smooth decay that is gentler than exponential decay but
    more aggressive than linear decay. The inverse square root schedule has
    been shown to work well for training diffusion models, particularly when
    combined with the magnitude-preserving operations that ensure stable
    gradients throughout training.

    The default values (t_ref=70000, sigma_ref=0.01) are recommended in the
    paper for ImageNet-scale experiments, but may need adjustment for different
    datasets or batch sizes.

    Args:
        optimizer: PyTorch optimizer to schedule
        t_ref: Reference training step for learning rate normalization (default: 70000)
               Controls the scale of the decay schedule
        sigma_ref: Reference learning rate at t=t_ref (default: 0.01)
                   The learning rate at the reference step

    Returns:
        LambdaLR scheduler: PyTorch learning rate scheduler implementing inverse sqrt decay

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> scheduler = InvSqrtDecayLRSched(optimizer, t_ref=70000, sigma_ref=0.01)
        >>> for epoch in range(num_epochs):
        ...     train_one_epoch()
        ...     scheduler.step()
    """
    def inv_sqrt_decay_fn(t: int):
        return sigma_ref / sqrt(max(t / t_ref, 1.))

    return LambdaLR(optimizer, lr_lambda = inv_sqrt_decay_fn)

# example

if __name__ == '__main__':
    unet = KarrasUnet(
        image_size = 64,
        dim = 192,
        dim_max = 768,
        num_classes = 1000,
    )

    images = torch.randn(2, 4, 64, 64)

    denoised_images = unet(
        images,
        time = torch.ones(2,),
        class_labels = torch.randint(0, 1000, (2,))
    )

    assert denoised_images.shape == images.shape
