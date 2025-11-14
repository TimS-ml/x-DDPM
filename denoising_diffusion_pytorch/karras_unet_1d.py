"""
Karras UNet 1D - Magnitude-Preserving UNet for 1D Sequential Data

This module implements a 1D variant of the magnitude-preserving UNet architecture
proposed by Karras et al. in "Analyzing and Improving the Training Dynamics of
Diffusion Models" (https://arxiv.org/abs/2312.02696).

Key Adaptations for 1D:
    - Uses Conv1d instead of Conv2d for processing sequential/temporal data
    - Spatial dimensions (height/width) replaced with sequence length
    - Downsampling/upsampling operations work along the temporal dimension
    - Attention operates over sequence positions rather than spatial locations

Magnitude-Preserving Properties:
    - All operations preserve the expected magnitude of activations
    - No biases throughout the network (replaced with magnitude-preserving operations)
    - Weight normalization applied to all convolutional and linear layers
    - Custom activation functions (MPSiLU) that maintain magnitude
    - Specialized residual connections (MPAdd) and concatenations (MPCat)

Architecture Features:
    - Multi-scale U-Net structure with encoder-decoder design
    - Self-attention at specified resolutions for capturing long-range dependencies
    - Time embedding via Fourier features for diffusion timestep conditioning
    - Optional class conditioning for class-conditional generation
    - Self-conditioning support for improved sampling quality

This 1D variant is suitable for:
    - Audio waveform generation
    - Time series modeling
    - Sequential data synthesis
    - Any task requiring diffusion models on 1D data
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

# Helper Functions

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
    Return val if it exists, otherwise return d (or call d if it's callable).

    Args:
        val: The value to check
        d: Default value or callable that returns a default value

    Returns:
        val if it exists, otherwise d() if d is callable, else d
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def xnor(x, y):
    """
    Logical XNOR operation - returns True if both are True or both are False.

    Args:
        x: First boolean value
        y: Second boolean value

    Returns:
        bool: True if x and y have the same truth value
    """
    return not (x ^ y)

def append(arr, el):
    """
    Append element to the end of array (in-place).

    Args:
        arr: List to append to
        el: Element to append
    """
    arr.append(el)

def prepend(arr, el):
    """
    Insert element at the beginning of array (in-place).

    Args:
        arr: List to prepend to
        el: Element to insert at position 0
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
    Cast value to tuple of specified length if not already a tuple.

    Args:
        t: Value to cast (if already tuple, returned as-is)
        length: Length of tuple to create if t is not a tuple

    Returns:
        tuple: Original tuple or tuple with t repeated length times
    """
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    """
    Check if numerator is evenly divisible by denominator.

    Args:
        numer: Numerator value
        denom: Denominator value

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
        eps: Epsilon for numerical stability (default: 1e-12)

    Returns:
        L2-normalized tensor
    """
    return F.normalize(t, dim = dim, eps = eps)

def interpolate_1d(x, length, mode = 'bilinear'):
    """
    Interpolate 1D sequence to a target length.

    This function adapts 2D interpolation for 1D data by temporarily adding
    and then removing a dummy spatial dimension.

    Args:
        x: Input tensor of shape (batch, channels, time)
        length: Target sequence length
        mode: Interpolation mode (default: 'bilinear')

    Returns:
        Interpolated tensor of shape (batch, channels, length)
    """
    x = rearrange(x, 'b c t -> b c t 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, 'b c t 1 -> b c t')

# Magnitude-Preserving Activations
# section 2.5

class MPSiLU(Module):
    """
    Magnitude-Preserving SiLU (Swish) activation function.

    Standard SiLU reduces the magnitude of activations. This version scales
    the output by 1/0.596 to preserve expected magnitude, as derived in
    section 2.5 of the Karras et al. paper.

    The scaling factor 0.596 is the empirically determined reduction factor
    of standard SiLU activation.
    """
    def forward(self, x):
        """
        Apply magnitude-preserving SiLU activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor with preserved magnitude
        """
        return F.silu(x) / 0.596

# Gain - Layer Scaling

class Gain(Module):
    """
    Learnable scalar gain parameter for layer output scaling.

    Initialized to 0, this allows certain layers (like embeddings) to start
    with no contribution and gradually learn their importance during training.
    This is particularly useful for time and class embeddings in diffusion models.
    """
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        """
        Apply learnable gain to input.

        Args:
            x: Input tensor

        Returns:
            Scaled tensor: x * gain
        """
        return x * self.gain

# Magnitude-Preserving Concatenation
# equation (103) - default to 0.5, which they recommended

class MPCat(Module):
    """
    Magnitude-Preserving Concatenation operation.

    Standard concatenation can alter the magnitude of gradients and activations.
    This implements equation (103) from the paper, which scales the inputs before
    concatenation to preserve magnitude statistics.

    The parameter t controls the relative weighting:
    - t=0.5 (default): Equal weighting, recommended by the paper
    - Higher t: More weight to second tensor b
    - Lower t: More weight to first tensor a

    This is crucial in the 1D UNet for skip connections where encoder features
    are concatenated with decoder features along the channel dimension.

    Args:
        t: Interpolation parameter (default: 0.5)
        dim: Dimension along which to concatenate (default: -1)
    """
    def __init__(self, t = 0.5, dim = -1):
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        """
        Concatenate tensors a and b with magnitude preservation.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            Magnitude-preserving concatenation of a and b
        """
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        C = sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

        a = a * (1. - t) / sqrt(Na)
        b = b * t / sqrt(Nb)

        return C * torch.cat((a, b), dim = dim)

# Magnitude-Preserving Sum
# equation (88)
# empirically, they found t=0.3 for encoder / decoder / attention residuals
# and for embedding, t=0.5

class MPAdd(Module):
    """
    Magnitude-Preserving Addition for residual connections.

    Standard residual connections (x + res) can alter activation magnitudes.
    This implements equation (88) from the paper for magnitude-preserving
    weighted addition of two tensors.

    Empirically determined optimal values:
    - t=0.3 for encoder/decoder/attention residuals
    - t=0.5 for embedding additions

    This ensures stable training dynamics by maintaining consistent activation
    scales throughout the 1D UNet, which is critical for deep networks.

    Args:
        t: Interpolation parameter controlling the relative weighting
    """
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        """
        Add two tensors with magnitude preservation.

        Args:
            x: Primary input tensor
            res: Residual tensor to add

        Returns:
            Magnitude-preserving weighted sum of x and res
        """
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

# PixelNorm
# equation (30)

class PixelNorm(Module):
    """
    Pixel Normalization layer from the paper.

    Normalizes features to unit length along a specified dimension, then
    scales by sqrt(dimension_size) to preserve magnitude. This is equation (30)
    from the paper.

    The paper uses a higher epsilon (1e-4) compared to standard normalization
    for improved numerical stability.

    For 1D sequences:
    - dim=1 normalizes across channels at each time step
    - dim=-1 normalizes across time for each channel

    Args:
        dim: Dimension along which to normalize
        eps: Epsilon for numerical stability (default: 1e-4 as in paper)
    """
    def __init__(self, dim, eps = 1e-4):
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

# Forced Weight Normalization for Conv1d and Linear layers
# Algorithm 1 in paper

def normalize_weight(weight, eps = 1e-4):
    """
    Normalize weight tensor according to Algorithm 1 in the paper.

    This performs L2 normalization of weights and scales them to preserve
    the expected magnitude. Applied to both convolutional and linear layers
    to ensure magnitude-preserving properties throughout the network.

    The normalization is done per output channel (first dimension), with
    scaling by sqrt(total_params / num_output_channels).

    Args:
        weight: Weight tensor of shape (out_channels, in_channels, ...)
        eps: Epsilon for numerical stability (default: 1e-4)

    Returns:
        Normalized weight tensor with same shape as input
    """
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

class Conv1d(Module):
    """
    Magnitude-Preserving 1D Convolution layer (adapted from 2D version).

    Key differences from standard Conv1d:
    1. No bias term (removed for magnitude preservation)
    2. Weight normalization applied at every forward pass
    3. Weights scaled by 1/sqrt(fan_in) for magnitude preservation
    4. Optional diagonal initialization for identity-like starting behavior
    5. Optional concatenation of ones channel to preserve expressivity

    The 1D adaptation processes sequential data (batch, channels, time) instead
    of spatial data. Critical for building the 1D UNet encoder and decoder blocks.

    Args:
        dim_in: Number of input channels
        dim_out: Number of output channels
        kernel_size: Size of the convolutional kernel (along time dimension)
        eps: Epsilon for weight normalization (default: 1e-4)
        init_dirac: If True, initialize with Dirac delta (identity-like)
        concat_ones_to_input: If True, concatenate a channel of ones to input
                             (used in input block to protect against loss of
                             expressivity from bias removal)
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        init_dirac = False,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size)
        self.weight = nn.Parameter(weight)

        if init_dirac:
            nn.init.dirac_(self.weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        """
        Apply magnitude-preserving 1D convolution.

        During training, weights are normalized in-place for efficiency.
        The normalized weights are then scaled by 1/sqrt(fan_in) and applied.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Convolved tensor of shape (batch, dim_out, time)
        """
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 1, 0), value = 1.)

        return F.conv1d(x, weight, padding = 'same')

class Linear(Module):
    """
    Magnitude-Preserving Linear (fully-connected) layer.

    Similar to Conv1d, this implements a bias-free linear layer with:
    1. No bias term for magnitude preservation
    2. Weight normalization at every forward pass
    3. Scaling by 1/sqrt(fan_in)

    Used primarily for:
    - Time embedding transformations
    - Class embedding transformations
    - Attention projections

    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        eps: Epsilon for weight normalization (default: 1e-4)
    """
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        """
        Apply magnitude-preserving linear transformation.

        Args:
            x: Input tensor of shape (..., dim_in)

        Returns:
            Transformed tensor of shape (..., dim_out)
        """
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

# Magnitude-Preserving Fourier Embeddings

class MPFourierEmbedding(Module):
    """
    Magnitude-Preserving Fourier Feature Embedding for time conditioning.

    Converts scalar timestep values into high-dimensional positional embeddings
    using random Fourier features. The embedding is scaled by sqrt(2) to maintain
    magnitude preservation properties.

    Process:
    1. Random frequencies sampled once at initialization (frozen)
    2. Timestep multiplied by frequencies and 2π
    3. Concatenate sin and cos of the result
    4. Scale by sqrt(2) for magnitude preservation

    This provides the 1D UNet with rich temporal information about the diffusion
    timestep, essential for the denoising process.

    Args:
        dim: Embedding dimension (must be even, as half are sin and half are cos)
    """
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        """
        Embed timestep values into Fourier features.

        Args:
            x: Timestep tensor of shape (batch,)

        Returns:
            Fourier feature embeddings of shape (batch, dim)
        """
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)

# Building Block Modules

class Encoder(Module):
    """
    Encoder block for the 1D UNet architecture.

    The encoder processes sequential data through the downsampling path of the UNet.
    Each encoder block consists of:
    1. Optional downsampling (2x along time dimension via interpolation + conv)
    2. PixelNorm for activation normalization
    3. Two convolutional blocks with MPSiLU activation
    4. Time/class embedding injection via adaptive gain
    5. Magnitude-preserving residual connection
    6. Optional self-attention for long-range dependencies

    1D Adaptation:
    - Downsampling reduces sequence length by factor of 2
    - All convolutions operate on temporal dimension
    - Attention operates over sequence positions

    Args:
        dim: Input channel dimension
        dim_out: Output channel dimension (default: same as dim)
        emb_dim: Dimension of time/class embeddings for conditioning
        dropout: Dropout probability (default: 0.1)
        mp_add_t: Magnitude-preserving add parameter for residual (default: 0.3)
        has_attn: Whether to include self-attention (default: False)
        attn_dim_head: Dimension per attention head (default: 64)
        attn_res_mp_add_t: MP add parameter for attention residual (default: 0.3)
        attn_flash: Whether to use flash attention (default: False)
        downsample: Whether to downsample the sequence (default: False)
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
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv1d(curr_dim, dim_out, 1)
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
            Conv1d(curr_dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv1d(dim_out, dim_out, 3)
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
        Process input through encoder block.

        Args:
            x: Input tensor of shape (batch, channels, time)
            emb: Optional time/class embedding of shape (batch, emb_dim)

        Returns:
            Processed tensor, potentially downsampled in time dimension
        """
        if self.downsample:
            x = interpolate_1d(x, x.shape[-1] // 2, mode = 'bilinear')
            x = self.downsample_conv(x)

        x = self.pixel_norm(x)

        res = x.clone()

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

class Decoder(Module):
    """
    Decoder block for the 1D UNet architecture.

    The decoder processes sequential data through the upsampling path of the UNet.
    Each decoder block consists of:
    1. Optional upsampling (2x along time dimension via interpolation)
    2. Two convolutional blocks with MPSiLU activation
    3. Time/class embedding injection via adaptive gain
    4. Magnitude-preserving residual connection
    5. Optional self-attention for long-range dependencies

    Key difference from Encoder:
    - Skip connections: Decoder receives concatenated skip connections from encoder
      (handled externally via MPCat before passing to decoder)
    - Upsampling increases sequence length by factor of 2

    1D Adaptation:
    - Upsampling doubles sequence length along time dimension
    - Processes concatenated encoder features and decoder features
    - Attention operates over upsampled sequence positions

    Args:
        dim: Input channel dimension (may include concatenated skip channels)
        dim_out: Output channel dimension (default: same as dim)
        emb_dim: Dimension of time/class embeddings for conditioning
        dropout: Dropout probability (default: 0.1)
        mp_add_t: Magnitude-preserving add parameter for residual (default: 0.3)
        has_attn: Whether to include self-attention (default: False)
        attn_dim_head: Dimension per attention head (default: 64)
        attn_res_mp_add_t: MP add parameter for attention residual (default: 0.3)
        attn_flash: Whether to use flash attention (default: False)
        upsample: Whether to upsample the sequence (default: False)
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
            Conv1d(dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv1d(dim_out, dim_out, 3)
        )

        self.res_conv = Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

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
        Process input through decoder block.

        Args:
            x: Input tensor of shape (batch, channels, time)
               If skip connections used, channels dimension already includes
               concatenated encoder features
            emb: Optional time/class embedding of shape (batch, emb_dim)

        Returns:
            Processed tensor, potentially upsampled in time dimension
        """
        if self.upsample:
            x = interpolate_1d(x, x.shape[-1] * 2, mode = 'bilinear')

        res = self.res_conv(x)

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

# Attention

class Attention(Module):
    """
    Magnitude-Preserving Self-Attention for 1D sequences.

    Implements multi-head self-attention adapted for 1D sequential data with:
    1. Query, key, value projections via Conv1d (kernel_size=1)
    2. PixelNorm applied to Q, K, V for magnitude preservation
    3. Learnable memory keys/values for improved capacity
    4. Magnitude-preserving residual connection
    5. Optional flash attention for efficiency

    1D Adaptation:
    - Attention operates over sequence positions (time dimension)
    - Unlike 2D attention which operates over spatial positions (H×W),
      this attends over temporal positions (T)
    - Useful for capturing long-range dependencies in sequential data

    The memory keys/values allow the attention to learn common patterns
    that don't depend on the input, improving expressive power.

    Args:
        dim: Input/output channel dimension
        heads: Number of attention heads (default: 4)
        dim_head: Dimension per head (default: 64)
        num_mem_kv: Number of memory key-value pairs (default: 4)
        flash: Use flash attention for efficiency (default: False)
        mp_add_t: MP add parameter for residual connection (default: 0.3)
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
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim = -1)

        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv1d(dim, hidden_dim * 3, 1)
        self.to_out = Conv1d(hidden_dim, dim, 1)

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        """
        Apply self-attention to 1D sequence.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Attention output with residual, shape (batch, channels, time)
        """
        res, b, c, n = x, *x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h n c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)

        return self.mp_add(out, res)

# UNet proposed by Karras et al.
# Improvised 1D version
# Bias-less, no group-norms, with magnitude preserving operations

class KarrasUnet1D(Module):
    """
    1D Karras UNet - Magnitude-Preserving Denoising Network for Sequential Data

    This is a 1D adaptation of the UNet architecture from "Analyzing and Improving
    the Training Dynamics of Diffusion Models" by Karras et al. (2023).
    Based on Figure 21, Config G from the paper.

    Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │ Input: (batch, channels, seq_len)                           │
    │   ↓                                                          │
    │ Input Conv1d + Concat Ones                                  │
    │   ↓                                                          │
    │ Encoder Blocks (downsample path)                            │
    │   ├─ Multiple encoder blocks at each resolution             │
    │   ├─ Downsample by 2x at each stage                        │
    │   ├─ Channels double (up to dim_max)                       │
    │   └─ Optional attention at specified resolutions            │
    │   ↓                                                          │
    │ Middle Blocks (bottleneck)                                  │
    │   ├─ Deepest resolution processing                         │
    │   └─ Attention at bottleneck                               │
    │   ↓                                                          │
    │ Decoder Blocks (upsample path)                              │
    │   ├─ Skip connections from encoder (via MPCat)             │
    │   ├─ Upsample by 2x at each stage                          │
    │   ├─ Channels halve                                         │
    │   └─ Optional attention at specified resolutions            │
    │   ↓                                                          │
    │ Output Conv1d + Gain                                        │
    │   ↓                                                          │
    │ Output: (batch, channels, seq_len)                          │
    └─────────────────────────────────────────────────────────────┘

    Key 1D Adaptations:
    - Conv1d instead of Conv2d throughout
    - Sequence length (time) replaces spatial dimensions (height, width)
    - Downsampling reduces sequence length: seq_len → seq_len/2
    - Upsampling increases sequence length: seq_len → seq_len*2
    - Attention operates over temporal positions instead of spatial locations
    - Input shape: (batch, channels, seq_len) vs 2D: (batch, channels, H, W)

    Magnitude-Preserving Features:
    - No biases anywhere (replaced with MPAdd, MPCat, etc.)
    - Weight normalization on all Conv1d and Linear layers
    - PixelNorm instead of GroupNorm/BatchNorm
    - MPSiLU activation instead of standard SiLU
    - MPAdd for residual connections
    - MPCat for skip connections

    Conditioning:
    - Time embedding via MPFourierEmbedding
    - Optional class conditioning via one-hot or integer labels
    - Self-conditioning support for iterative refinement

    Args:
        seq_len: Length of input sequence (must be divisible by 2^num_downsamples)
        dim: Base channel dimension (default: 192)
        dim_max: Maximum channel dimension (caps channel doubling) (default: 768)
        num_classes: Number of classes for class-conditional generation (default: None)
        channels: Number of input/output channels (default: 4)
        num_downsamples: Number of downsampling stages (default: 3)
        num_blocks_per_stage: Encoder/decoder blocks per resolution (default: 4)
        attn_res: Resolutions at which to apply attention, e.g., (16, 8) (default: (16, 8))
        fourier_dim: Dimension of Fourier time embedding (default: 16)
        attn_dim_head: Dimension per attention head (default: 64)
        attn_flash: Use flash attention for efficiency (default: False)
        mp_cat_t: MP parameter for skip connection concatenation (default: 0.5)
        mp_add_emb_t: MP parameter for embedding addition (default: 0.5)
        attn_res_mp_add_t: MP parameter for attention residuals (default: 0.3)
        resnet_mp_add_t: MP parameter for conv block residuals (default: 0.3)
        dropout: Dropout probability (default: 0.1)
        self_condition: Enable self-conditioning (default: False)
    """

    def __init__(
        self,
        *,
        seq_len,
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
        self.seq_len = seq_len
        input_channels = channels * (2 if self_condition else 1)

        # input and output blocks

        self.input_block = Conv1d(input_channels, dim, 3, concat_ones_to_input = True)

        self.output_block = nn.Sequential(
            Conv1d(dim, channels, 3),
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
        curr_res = seq_len

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
        Calculate the total downsampling factor of the network.

        For 1D sequences, this determines how much the sequence length is
        reduced at the bottleneck. With num_downsamples=3, the bottleneck
        sequence length is seq_len/8.

        Returns:
            int: Total downsampling factor (2^num_downsamples)
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
        Forward pass through the 1D UNet for denoising.

        Processing flow:
        1. Concatenate self-conditioning if enabled
        2. Embed time via Fourier features
        3. Optionally add class embedding to time embedding
        4. Pass through input block
        5. Encoder path with skip connection storage
        6. Middle blocks at bottleneck
        7. Decoder path with skip connection fusion via MPCat
        8. Output block to produce denoised sequence

        Args:
            x: Noisy input sequence of shape (batch, channels, seq_len)
            time: Diffusion timestep of shape (batch,)
            self_cond: Optional self-conditioning from previous prediction,
                      shape (batch, channels, seq_len). Used when self_condition=True.
            class_labels: Optional class labels for conditional generation.
                         Can be integer labels (batch,) or one-hot (batch, num_classes).
                         Required if num_classes was specified during init.

        Returns:
            Denoised sequence prediction of shape (batch, channels, seq_len)

        Raises:
            AssertionError: If input shape doesn't match expected (channels, seq_len)
            AssertionError: If class_labels provided but model not configured for it
                          (or vice versa)
        """
        # validate image shape

        assert x.shape[1:] == (self.channels, self.seq_len)

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

# Improvised MP Transformer (experimental, uses Conv2d - not adapted to 1D)

class MPFeedForward(Module):
    """
    Magnitude-Preserving Feed-Forward Network.

    A simple feed-forward block with:
    1. PixelNorm for normalization
    2. Expansion via 1x1 convolution
    3. MPSiLU activation
    4. Projection back to original dimension
    5. Magnitude-preserving residual connection

    Note: This currently uses Conv2d and is part of an experimental transformer
    implementation. It has not been adapted to 1D like the main UNet components.

    Args:
        dim: Input/output channel dimension
        mult: Expansion multiplier for hidden dimension (default: 4)
        mp_add_t: MP parameter for residual connection (default: 0.3)
    """
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        mp_add_t = 0.3
    ):
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
        Apply feed-forward transformation with residual.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor with magnitude-preserving residual
        """
        res = x
        out = self.net(x)
        return self.mp_add(out, res)

class MPImageTransformer(Module):
    """
    Magnitude-Preserving Transformer (experimental).

    A stack of transformer layers, each consisting of:
    1. Self-attention with magnitude preservation
    2. Feed-forward network with magnitude preservation

    This is an experimental component that could potentially be used as an
    alternative to or in combination with the convolutional UNet architecture.

    Note: The feed-forward component uses Conv2d, so this is not fully adapted
    for 1D sequences. The Attention component works with 1D data.

    Args:
        dim: Model dimension
        depth: Number of transformer layers
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 8)
        num_mem_kv: Number of memory key-value pairs in attention (default: 4)
        ff_mult: Feed-forward expansion multiplier (default: 4)
        attn_flash: Use flash attention (default: False)
        residual_mp_add_t: MP parameter for all residual connections (default: 0.3)
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
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, num_mem_kv = num_mem_kv, flash = attn_flash, mp_add_t = residual_mp_add_t),
                MPFeedForward(dim = dim, mult = ff_mult, mp_add_t = residual_mp_add_t)
            ]))

    def forward(self, x):
        """
        Apply transformer layers sequentially.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor after all layers
        """

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

# Example Usage

if __name__ == '__main__':
    # Create a 1D Karras UNet for sequences of length 64
    # with 4 channels and 1000 class conditioning
    unet = KarrasUnet1D(
        seq_len = 64,          # Input sequence length
        dim = 192,             # Base channel dimension
        dim_max = 768,         # Maximum channels after downsampling
        num_classes = 1000,    # Number of classes for conditional generation
    )

    # Create a batch of 2 noisy sequences
    # Shape: (batch=2, channels=4, seq_len=64)
    images = torch.randn(2, 4, 64)

    # Denoise the sequences with time and class conditioning
    denoised_images = unet(
        images,
        time = torch.ones(2,),                    # Timestep for each batch item
        class_labels = torch.randint(0, 1000, (2,))  # Class label for each batch item
    )

    # Verify output shape matches input shape
    assert denoised_images.shape == images.shape
