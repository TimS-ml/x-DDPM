"""
1D Denoising Diffusion Probabilistic Models (DDPM)

This module implements diffusion models for 1D sequential data such as time series,
audio waveforms, sensor data, and other continuous sequential signals.

Key Differences from 2D Image Diffusion:
--------------------------------------------
1. **Data Structure**: Operates on 1D sequences (batch, channels, length) instead of
   2D images (batch, channels, height, width). This makes it suitable for temporal
   or sequential data rather than spatial data.

2. **Architecture**: Uses Conv1d operations instead of Conv2d throughout the U-Net,
   processing data along a single spatial dimension (sequence length) rather than
   two dimensions (height and width).

3. **Applications**: Designed for:
   - Time series generation (stock prices, weather data, etc.)
   - Audio waveform synthesis
   - Sensor signal generation
   - Any sequential continuous data

4. **Sequence Length**: Works with variable-length sequences but requires fixed
   length during training (similar to how 2D diffusion requires fixed image size).

Core Components:
----------------
- Dataset1D: Simple dataset wrapper for 1D tensor data
- Unet1D: 1D U-Net architecture with temporal convolutions and attention
- GaussianDiffusion1D: Implements the forward and reverse diffusion processes
- Trainer1D: Training loop with gradient accumulation, EMA, and checkpointing

Diffusion Process Overview:
----------------------------
1. Forward Process: Gradually adds Gaussian noise to data over T timesteps
2. Reverse Process: Neural network learns to denoise and reconstruct original data
3. Sampling: Start from pure noise and iteratively denoise to generate new samples
"""

import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from denoising_diffusion_pytorch.version import __version__

# constants

# Named tuple to store model predictions during the diffusion process
# pred_noise: predicted noise at timestep t
# pred_x_start: predicted clean data (x_0) from noisy input
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    """
    Check if a value exists (is not None).

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
    Used as a no-op placeholder in conditional logic.

    Args:
        t: Input tensor or value
        *args, **kwargs: Ignored arguments

    Returns:
        t: The input unchanged
    """
    return t

def cycle(dl):
    """
    Infinitely cycle through a dataloader.
    When the dataloader is exhausted, restart from the beginning.

    Args:
        dl: PyTorch DataLoader to cycle through

    Yields:
        data: Batches from the dataloader, infinitely
    """
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    """
    Check if a number has an integer square root.
    Used to validate that sample grids can be arranged in a square.

    Args:
        num: Number to check

    Returns:
        bool: True if sqrt(num) is an integer
    """
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    """
    Divide a number into groups of a given size.
    Used for batching when the total doesn't divide evenly.

    Args:
        num: Total number to divide
        divisor: Size of each group

    Returns:
        list: List of group sizes, e.g., num_to_groups(10, 3) -> [3, 3, 3, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    """
    Convert an image to a specified type/mode.
    Legacy function from 2D version, kept for compatibility.

    Args:
        img_type: Target image mode (e.g., 'RGB', 'L')
        image: PIL Image to convert

    Returns:
        Converted image
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalize data from [0, 1] range to [-1, 1] range.
    This is the standard normalization for diffusion models.

    Args:
        img: Tensor with values in [0, 1]

    Returns:
        Tensor with values in [-1, 1]
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize data from [-1, 1] range back to [0, 1] range.
    Used after sampling to get data in original range.

    Args:
        t: Tensor with values in [-1, 1]

    Returns:
        Tensor with values in [0, 1]
    """
    return (t + 1) * 0.5

# data

class Dataset1D(Dataset):
    """
    Simple dataset wrapper for 1D sequential data.

    This class wraps a tensor of 1D sequences for use with PyTorch DataLoader.
    The tensor should have shape (num_samples, channels, sequence_length) or
    (num_samples, sequence_length, channels) depending on the data format.

    Args:
        tensor: Tensor containing all training sequences
                Shape: (num_samples, channels, seq_len) or (num_samples, seq_len, channels)

    Example:
        >>> data = torch.randn(1000, 3, 128)  # 1000 samples, 3 channels, length 128
        >>> dataset = Dataset1D(data)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    def __init__(self, tensor: Tensor):
        super().__init__()
        # Clone the tensor to avoid unwanted modifications to original data
        self.tensor = tensor.clone()

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.tensor)

    def __getitem__(self, idx):
        """
        Get a single sequence by index.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            Cloned tensor of the sequence at index idx
        """
        return self.tensor[idx].clone()

# small helper modules

class Residual(Module):
    """
    Residual connection wrapper that adds the input to the output.
    Implements: output = fn(x) + x

    This helps with gradient flow and allows networks to learn identity mappings.

    Args:
        fn: Any neural network module to wrap with a residual connection
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """Apply function and add input (residual connection)."""
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    """
    Upsample 1D sequences by factor of 2 using nearest neighbor interpolation.

    This doubles the sequence length. For example, a sequence of length 64
    becomes length 128. Used in the decoder path of the U-Net.

    Args:
        dim: Input number of channels
        dim_out: Output number of channels (default: same as dim)

    Returns:
        Sequential module that upsamples then applies convolution
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),  # Double the sequence length
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)  # Refine with convolution
    )

def Downsample(dim, dim_out = None):
    """
    Downsample 1D sequences by factor of 2 using strided convolution.

    This halves the sequence length. For example, a sequence of length 128
    becomes length 64. Used in the encoder path of the U-Net.

    Args:
        dim: Input number of channels
        dim_out: Output number of channels (default: same as dim)

    Returns:
        Conv1d module with stride 2 for downsampling
    """
    # Kernel size 4, stride 2, padding 1 -> halves the sequence length
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization for 1D sequences.

    RMSNorm is a simpler alternative to LayerNorm that normalizes using only
    the root mean square statistic, without centering. It's computationally
    efficient and works well for diffusion models.

    Args:
        dim: Number of channels to normalize
    """
    def __init__(self, dim):
        super().__init__()
        # Learnable scale parameter for each channel
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        """
        Normalize input using RMS and scale by learnable parameter.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Normalized and scaled tensor
        """
        # Normalize across channel dimension, then scale
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    """
    Wrapper that applies normalization before a function.

    This implements the "pre-norm" architecture where normalization comes
    before the main operation (e.g., attention), which can improve training
    stability compared to post-norm.

    Args:
        dim: Number of channels
        fn: Function/module to apply after normalization
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """Normalize then apply function."""
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    """
    Sinusoidal positional embeddings for encoding timesteps.

    Converts scalar timestep values into high-dimensional embeddings using
    sin and cos functions at different frequencies. This allows the model
    to understand the relative position/magnitude of timesteps.

    Based on the positional encoding from "Attention is All You Need".

    Args:
        dim: Dimension of the embedding (must be even)
        theta: Base for the geometric progression of frequencies (default: 10000)
    """
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """
        Compute sinusoidal embeddings for timestep values.

        Args:
            x: Timestep values, shape (batch_size,)

        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        # Create geometric progression of frequencies
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Multiply timesteps by frequencies
        emb = x[:, None] * emb[None, :]
        # Concatenate sin and cos components
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """
    Random or learned Fourier features for timestep embedding.

    An alternative to fixed sinusoidal embeddings that can either use random
    fixed frequencies or learnable frequencies. Can provide better performance
    than standard sinusoidal embeddings in some cases.

    Following @crowsonkb's implementation:
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8

    Args:
        dim: Dimension of the embedding (must be even)
        is_random: If True, frequencies are fixed random values.
                   If False, frequencies are learnable parameters.
    """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        # Random or learnable frequency weights
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        """
        Compute Fourier features for timestep values.

        Args:
            x: Timestep values, shape (batch_size,)

        Returns:
            Embeddings of shape (batch_size, dim + 1)
            (includes original value concatenated with sin/cos features)
        """
        x = rearrange(x, 'b -> b 1')
        # Compute frequencies scaled by learned/random weights
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # Apply sin and cos
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # Concatenate original value with Fourier features
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    """
    Basic convolutional block with normalization and activation.

    This is a fundamental building block used in the ResNet-style architecture.
    Applies: Conv1d -> Norm -> (optional scale/shift) -> SiLU -> Dropout

    Args:
        dim: Input number of channels
        dim_out: Output number of channels
        dropout: Dropout probability (default: 0.)
    """
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()  # Smooth activation function (Swish)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        """
        Forward pass with optional adaptive normalization.

        Args:
            x: Input tensor (batch, dim, sequence_length)
            scale_shift: Optional tuple of (scale, shift) tensors for
                        adaptive normalization based on timestep embedding

        Returns:
            Processed tensor (batch, dim_out, sequence_length)
        """
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
    Residual block with timestep conditioning for the U-Net.

    This block processes sequential data with two convolutional blocks and
    incorporates timestep information through adaptive normalization (similar
    to AdaGN/AdaLN). The timestep embedding modulates the features to help
    the model understand what noise level to remove.

    Structure: x -> Block1 (with time conditioning) -> Block2 -> + residual -> output

    Args:
        dim: Input number of channels
        dim_out: Output number of channels
        time_emb_dim: Dimension of timestep embedding (if None, no time conditioning)
        dropout: Dropout probability for first block
    """
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        # MLP to project timestep embedding to scale and shift parameters
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)  # *2 for scale and shift
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        # Projection for residual connection when dimensions don't match
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        """
        Forward pass with optional timestep conditioning.

        Args:
            x: Input tensor (batch, dim, sequence_length)
            time_emb: Timestep embedding (batch, time_emb_dim)

        Returns:
            Output tensor (batch, dim_out, sequence_length) with residual connection
        """

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # Project timestep embedding to scale and shift parameters
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')  # Add sequence dimension
            scale_shift = time_emb.chunk(2, dim = 1)  # Split into scale and shift

        # First block with timestep-adaptive normalization
        h = self.block1(x, scale_shift = scale_shift)

        # Second block without adaptive normalization
        h = self.block2(h)

        # Add residual connection
        return h + self.res_conv(x)

class LinearAttention(Module):
    """
    Linear attention mechanism with O(n) complexity instead of O(n^2).

    This is a more efficient approximation of standard attention that scales
    linearly with sequence length. It uses softmax normalization in the feature
    dimension instead of computing full pairwise attention scores.

    Used in earlier layers of the U-Net where efficiency is more important
    than the full expressiveness of standard attention.

    Args:
        dim: Input channel dimension
        heads: Number of attention heads (default: 4)
        dim_head: Dimension per attention head (default: 32)
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        Apply linear attention to input sequence.

        Args:
            x: Input tensor (batch, channels, sequence_length)

        Returns:
            Output tensor (batch, channels, sequence_length)
        """
        b, c, n = x.shape
        # Generate query, key, value projections
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        # Normalize q and k with softmax (linear attention trick)
        q = q.softmax(dim = -2)  # Softmax over channel dimension
        k = k.softmax(dim = -1)  # Softmax over sequence dimension

        q = q * self.scale

        # Compute context: k^T @ v (results in d x e matrix)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Apply context to queries
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    """
    Full multi-head self-attention mechanism with O(n^2) complexity.

    This is the standard scaled dot-product attention from "Attention is All
    You Need". It computes pairwise attention scores between all positions
    in the sequence, allowing each position to attend to all others.

    Used in the bottleneck of the U-Net where the sequence is shortest and
    we can afford the quadratic complexity for maximum expressiveness.

    Args:
        dim: Input channel dimension
        heads: Number of attention heads (default: 4)
        dim_head: Dimension per attention head (default: 32)
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5  # Scaling factor for dot product
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Apply full self-attention to input sequence.

        Args:
            x: Input tensor (batch, channels, sequence_length)

        Returns:
            Output tensor (batch, channels, sequence_length)
        """
        b, c, n = x.shape
        # Generate query, key, value projections
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        # Scale queries for stable gradients
        q = q * self.scale

        # Compute attention scores: Q @ K^T
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)  # Normalize attention weights

        # Apply attention to values: Attention @ V
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # Concatenate heads and project to output
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(Module):
    """
    1D U-Net architecture for denoising diffusion models.

    This is the core neural network that learns to denoise 1D sequences.
    It uses a U-Net architecture with:
    - Encoder path: Progressively downsamples and increases channels
    - Bottleneck: Processes the most compressed representation with attention
    - Decoder path: Progressively upsamples and decreases channels
    - Skip connections: Concatenate encoder features to decoder

    Key differences from 2D U-Net:
    - Uses Conv1d instead of Conv2d (operates on sequences, not images)
    - Downsampling/upsampling happens along the sequence dimension only
    - Attention mechanisms work across sequence positions (temporal attention)

    Args:
        dim: Base channel dimension (will be multiplied by dim_mults)
        init_dim: Initial channel dimension after first conv (default: same as dim)
        out_dim: Output channel dimension (default: channels or channels*2 if learned_variance)
        dim_mults: Channel multipliers for each resolution level (default: (1,2,4,8))
        channels: Number of input/output channels (e.g., 1 for mono audio, 3 for RGB-like data)
        dropout: Dropout probability in residual blocks
        self_condition: If True, enables self-conditioning (concatenates previous prediction)
        learned_variance: If True, model predicts both mean and variance
        learned_sinusoidal_cond: Use learned sinusoidal embeddings for timesteps
        random_fourier_features: Use random Fourier features for timesteps
        learned_sinusoidal_dim: Dimension for learned sinusoidal embeddings
        sinusoidal_pos_emb_theta: Base frequency for sinusoidal embeddings
        attn_dim_head: Dimension per attention head
        attn_heads: Number of attention heads in bottleneck
    """
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        # Double input channels if using self-conditioning (concatenate previous prediction)
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # Initial convolution to project input to model dimension
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        # Calculate channel dimensions for each resolution level
        # E.g., if dim=64 and dim_mults=(1,2,4,8): dims=[64, 64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))  # Pairs of (input_dim, output_dim)

        # time embeddings

        time_dim = dim * 4  # Timestep embedding dimension

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # Choose timestep embedding type
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        # MLP to process timestep embeddings
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Create partial function for ResnetBlock with fixed time_emb_dim and dropout
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])  # Encoder path
        self.ups = ModuleList([])    # Decoder path
        num_resolutions = len(in_out)

        # Build encoder (downsampling path)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # Each resolution level has: 2 ResNet blocks + attention + downsample
            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),  # First ResNet block
                resnet_block(dim_in, dim_in),  # Second ResNet block
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),  # Linear attention
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        # Bottleneck (middle of U-Net at lowest resolution)
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        # Use full attention in bottleneck for maximum expressiveness
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # Build decoder (upsampling path)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            # Each resolution level has: 2 ResNet blocks + attention + upsample
            # Note: dim_out + dim_in for first two blocks due to skip connections
            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),  # First block (receives skip connection)
                resnet_block(dim_out + dim_in, dim_out),  # Second block (receives skip connection)
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),  # Linear attention
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        # Output projection
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # Final layers to produce output
        self.final_res_block = resnet_block(init_dim * 2, init_dim)  # *2 for final skip connection
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        """
        Forward pass through the 1D U-Net.

        Args:
            x: Noisy input sequence (batch, channels, sequence_length)
            time: Timestep values (batch,) indicating noise level
            x_self_cond: Optional previous prediction for self-conditioning
                        (batch, channels, sequence_length)

        Returns:
            Denoised output (batch, out_dim, sequence_length)
            If learned_variance=True, out_dim = channels*2 (mean and variance)
            Otherwise, out_dim = channels
        """
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

        # Encoder path: downsample and extract features
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)  # Save for skip connection

            x = block2(x, t)
            x = attn(x)
            h.append(x)  # Save for skip connection

            x = downsample(x)  # Reduce sequence length

        # Bottleneck: process at lowest resolution with full attention
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)  # Full attention for maximum expressiveness
        x = self.mid_block2(x, t)

        # Decoder path: upsample and combine with skip connections
        for block1, block2, attn, upsample in self.ups:
            # Concatenate with skip connection from encoder
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            # Concatenate with another skip connection
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)  # Increase sequence length

        # Final skip connection with initial features
        x = torch.cat((x, r), dim = 1)

        # Final processing and projection to output channels
        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """
    Extract values from a tensor 'a' at indices 't' and reshape for broadcasting.

    This is a helper function to gather schedule values (like alphas, betas) for
    specific timesteps and reshape them to broadcast with data tensors.

    Args:
        a: 1D tensor of schedule values (e.g., alphas_cumprod)
        t: Batch of timestep indices
        x_shape: Shape of the data tensor to broadcast to

    Returns:
        Tensor of shape (batch_size, 1, 1, ...) for broadcasting
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    # Reshape to (batch, 1, 1, ...) for broadcasting with (batch, channels, seq_length)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    Create a linear schedule for beta values (noise variance).

    This is the original schedule from the DDPM paper. Beta increases linearly
    from a small value to a larger value over the diffusion process.

    Args:
        timesteps: Number of diffusion steps

    Returns:
        Tensor of beta values, shape (timesteps,)
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    Create a cosine schedule for beta values (noise variance).

    Proposed in "Improved Denoising Diffusion Probabilistic Models"
    (https://openreview.net/forum?id=-NEXDKk8gZ)

    The cosine schedule provides more stable training and better sample quality
    than the linear schedule. It starts with smaller noise increments and
    increases more gradually.

    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta from being too small at t=0

    Returns:
        Tensor of beta values, shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    # Compute alpha_cumprod using cosine function
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # Derive betas from alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(Module):
    """
    Gaussian Diffusion process for 1D data.

    This class implements the core diffusion process including:
    - Forward diffusion: Gradually adding noise to data
    - Reverse diffusion: Denoising using the trained neural network
    - Sampling: Generating new data from noise

    The diffusion process is mathematically equivalent to 2D diffusion but operates
    on 1D sequences. The key difference is in the data shape and how we interpret
    the sequence dimension.

    Args:
        model: The denoising model (typically Unet1D)
        seq_length: Length of the 1D sequences
        timesteps: Number of diffusion steps (default: 1000)
        sampling_timesteps: Number of steps for sampling (default: same as timesteps)
                           If less than timesteps, uses DDIM sampling
        objective: Training objective - 'pred_noise', 'pred_x0', or 'pred_v'
                  - pred_noise: Predict the noise added at each step (original DDPM)
                  - pred_x0: Predict the clean data directly
                  - pred_v: Predict velocity (used in some recent models)
        beta_schedule: Noise schedule - 'linear' or 'cosine'
        ddim_sampling_eta: Stochasticity in DDIM sampling (0=deterministic, 1=DDPM)
        auto_normalize: Automatically normalize data to [-1, 1]
        channels: Number of channels in the data
        self_condition: Enable self-conditioning (predicts x0, then uses it as input)
        channel_first: If True, data shape is (batch, channels, seq_len)
                      If False, data shape is (batch, seq_len, channels)
    """
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        channels = None,
        self_condition = None,
        channel_first = True
    ):
        super().__init__()
        self.model = model
        self.channels = default(channels, lambda: self.model.channels)
        self.self_condition = default(self_condition, lambda: self.model.self_condition)

        # Handle different data layouts: (B, C, L) vs (B, L, C)
        self.channel_first = channel_first
        self.seq_index = -2 if not channel_first else -1  # Index of sequence dimension

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # Create noise schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # Pre-compute values used in diffusion equations
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # Product of all alphas up to t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)  # Shifted version

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        # Allow faster sampling with fewer steps using DDIM
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps  # Use DDIM if sampling with fewer steps
        self.ddim_sampling_eta = ddim_sampling_eta  # Controls stochasticity (0=deterministic)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # Register all noise schedule values as buffers (saved with model but not trained)
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # These coefficients are used in the forward diffusion process

        # Coefficients for adding noise: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1-alpha_cumprod) * noise
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))

        # Coefficients for predicting x_0 from x_t and noise
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # These are used in the reverse diffusion process

        # Variance of the posterior distribution
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))

        # Coefficients for computing posterior mean: mu = coef1 * x_0 + coef2 * x_t
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight
        # Different objectives benefit from different loss weighting based on SNR

        snr = alphas_cumprod / (1 - alphas_cumprod)  # Signal-to-noise ratio

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)  # Uniform weighting for noise prediction
        elif objective == 'pred_x0':
            loss_weight = snr  # Weight by SNR for x0 prediction
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)  # Balanced weighting for v-prediction

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        # Normalize data to [-1, 1] range (standard for diffusion models)
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict clean data x_0 from noisy data x_t and predicted noise.

        Uses the reparameterization: x_0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            noise: Predicted or actual noise

        Returns:
            Predicted clean data x_0
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        Predict noise from noisy data x_t and clean data x_0.

        Inverse of predict_start_from_noise.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            x0: Clean data

        Returns:
            Predicted noise
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """
        Predict v-parameterization from clean data and noise.

        v-parameterization is defined as: v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * x_0
        Used in progressive distillation and some recent diffusion models.

        Args:
            x_start: Clean data x_0
            t: Timestep
            noise: Noise

        Returns:
            v-parameterization value
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict clean data x_0 from noisy data x_t and v-parameterization.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            v: v-parameterization value

        Returns:
            Predicted clean data x_0
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the posterior distribution q(x_{t-1} | x_t, x_0).

        This is the distribution we sample from during reverse diffusion when
        we know the clean data x_0. During training/sampling, we use the model's
        prediction of x_0.

        Args:
            x_start: Clean data x_0 (or prediction of it)
            x_t: Noisy data at timestep t
            t: Current timestep

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

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, model_forward_kwargs: dict = dict()):
        """
        Get model predictions and convert to both noise and x_0 predictions.

        This method handles different training objectives (pred_noise, pred_x0, pred_v)
        and converts the model output to a standard format containing both predicted
        noise and predicted clean data.

        Args:
            x: Noisy input data
            t: Timestep
            x_self_cond: Self-conditioning input (previous prediction)
            clip_x_start: Whether to clip predicted x_0 to [-1, 1]
            rederive_pred_noise: If True and clip_x_start=True, recompute noise from clipped x_0
            model_forward_kwargs: Additional kwargs for model forward pass

        Returns:
            ModelPrediction namedtuple with pred_noise and pred_x_start
        """

        if exists(x_self_cond):
            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        model_output = self.model(x, t, **model_forward_kwargs)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        # Convert model output to standard format based on objective
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            # Optionally rederive noise from clipped x_0 for consistency
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

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True, model_forward_kwargs: dict = dict()):
        """
        Compute mean and variance for the reverse diffusion step p(x_{t-1} | x_t).

        Args:
            x: Noisy data at timestep t
            t: Current timestep
            x_self_cond: Self-conditioning input
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            model_forward_kwargs: Additional kwargs for model

        Returns:
            Tuple of (model_mean, posterior_variance, posterior_log_variance, x_start)
        """

        if exists(x_self_cond):
            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        preds = self.model_predictions(x, t, **model_forward_kwargs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # Compute posterior distribution parameters
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True, model_forward_kwargs: dict = dict()):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) using the model.

        This is one step of the reverse diffusion process (DDPM sampling).

        Args:
            x: Noisy data at timestep t
            t: Current timestep (scalar, not batched)
            x_self_cond: Self-conditioning input
            clip_denoised: Whether to clip predictions
            model_forward_kwargs: Additional kwargs for model

        Returns:
            Tuple of (pred_img, x_start) where pred_img is x_{t-1}
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised, model_forward_kwargs = model_forward_kwargs)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0 (deterministic final step)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_noise = False, model_forward_kwargs: dict = dict()):
        """
        Generate samples using the full DDPM sampling loop.

        Starts from pure noise and iteratively denoises over all timesteps.
        This is the standard DDPM sampling procedure.

        Args:
            shape: Shape of samples to generate (batch, channels, seq_length)
            return_noise: If True, return both samples and initial noise
            model_forward_kwargs: Additional kwargs for model

        Returns:
            Generated samples, or (samples, noise) if return_noise=True
        """
        batch, device = shape[0], self.betas.device

        # Start from pure Gaussian noise
        noise = torch.randn(shape, device=device)
        img = noise

        x_start = None

        # Iteratively denoise from t=T to t=0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None  # Use previous prediction for self-conditioning
            img, x_start = self.p_sample(img, t, self_cond, model_forward_kwargs = model_forward_kwargs)

        # Unnormalize from [-1, 1] back to original range
        img = self.unnormalize(img)

        if not return_noise:
            return img

        return img, noise

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True, model_forward_kwargs: dict = dict(), return_noise = False):
        """
        Generate samples using DDIM (Denoising Diffusion Implicit Models) sampling.

        DDIM allows faster sampling by skipping timesteps. When sampling_timesteps < num_timesteps,
        this provides a deterministic (eta=0) or semi-stochastic (eta>0) sampling procedure
        that's much faster than full DDPM sampling.

        Reference: "Denoising Diffusion Implicit Models" (Song et al., 2021)

        Args:
            shape: Shape of samples to generate (batch, channels, seq_length)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            model_forward_kwargs: Additional kwargs for model
            return_noise: If True, return both samples and initial noise

        Returns:
            Generated samples, or (samples, noise) if return_noise=True
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # Create evenly spaced timesteps for sampling (can skip steps)
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Start from pure noise
        noise = torch.randn(shape, device = device)
        img = noise

        x_start = None

        # DDIM sampling loop
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised, model_forward_kwargs = model_forward_kwargs)

            # Final step: return predicted x_0
            if time_next < 0:
                img = x_start
                continue

            # DDIM update equation
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # Variance term (eta=0 for deterministic, eta=1 for DDPM-like)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            # Update: combine predicted x_0, noise direction, and optional stochasticity
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # Unnormalize from [-1, 1] back to original range
        img = self.unnormalize(img)

        if not return_noise:
            return img

        return img, noise

    @torch.no_grad()
    def sample(self, batch_size = 16, return_noise = False, model_forward_kwargs: dict = dict()):
        """
        Generate samples from the diffusion model.

        Automatically chooses between DDPM and DDIM sampling based on sampling_timesteps.

        Args:
            batch_size: Number of samples to generate
            return_noise: If True, return both samples and initial noise
            model_forward_kwargs: Additional kwargs for model

        Returns:
            Generated samples of shape (batch_size, channels, seq_length)
            or (samples, noise) if return_noise=True
        """
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        # Create shape based on channel_first setting
        shape = (batch_size, channels, seq_length) if self.channel_first else (batch_size, seq_length, channels)
        return sample_fn(shape, return_noise = return_noise, model_forward_kwargs = model_forward_kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        """
        Interpolate between two sequences in latent space.

        Adds noise to both sequences, interpolates in noisy space, then denoises.
        This creates smooth transitions between sequences.

        Args:
            x1: First sequence
            x2: Second sequence (must have same shape as x1)
            t: Timestep to add noise to (default: num_timesteps-1, maximum noise)
            lam: Interpolation weight (0 = all x1, 1 = all x2, 0.5 = equal mix)

        Returns:
            Interpolated sequence
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        # Add noise to both sequences
        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        # Interpolate in noisy space
        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        # Denoise from timestep t back to 0
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process q(x_t | x_0).

        Add noise to clean data according to the noise schedule at timestep t.
        This is the forward diffusion process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise

        Args:
            x_start: Clean data x_0
            t: Timestep(s) to sample at
            noise: Optional pre-sampled noise (default: sample from N(0,I))

        Returns:
            Noisy data x_t
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, model_forward_kwargs: dict = dict(), return_reduced_loss = True):
        """
        Compute training loss for the diffusion model.

        This is the main training objective. It adds noise to clean data, then
        trains the model to predict the noise (or x_0, or v depending on objective).

        Args:
            x_start: Clean data x_0
            t: Timesteps for this batch
            noise: Optional pre-sampled noise
            model_forward_kwargs: Additional kwargs for model
            return_reduced_loss: If True, return mean loss. If False, return per-sample loss.

        Returns:
            Scalar loss value (if return_reduced_loss=True) or per-sample losses
        """
        b = x_start.shape[0]
        n = x_start.shape[self.seq_index]

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample - create noisy version of data

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # Get prediction for self-conditioning (don't backprop through this)
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        # predict and take gradient step

        model_out = self.model(x, t, **model_forward_kwargs)

        # Determine target based on training objective
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # Compute MSE loss between prediction and target
        loss = F.mse_loss(model_out, target, reduction = 'none')

        # Return unreduced loss if requested (useful for custom weighting)
        if not return_reduced_loss:
            return loss * extract(self.loss_weight, t, loss.shape)

        # Reduce to per-sample loss
        loss = reduce(loss, 'b ... -> b', 'mean')

        # Apply loss weighting based on timestep (SNR-based weighting)
        loss = loss * extract(self.loss_weight, t, loss.shape)

        # Return mean loss over batch
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Forward pass for training.

        Randomly samples timesteps, adds noise, and computes loss.

        Args:
            img: Clean data
            *args, **kwargs: Additional arguments for p_losses

        Returns:
            Training loss
        """
        b, n, device, seq_length, = img.shape[0], img.shape[self.seq_index], img.device, self.seq_length

        assert n == seq_length, f'seq length must be {seq_length}'
        # Sample random timesteps for each item in batch
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Normalize to [-1, 1] range
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# trainer class

class Trainer1D(object):
    """
    Training loop for 1D diffusion models.

    This class handles:
    - Training loop with gradient accumulation
    - Exponential Moving Average (EMA) of model weights
    - Periodic sampling and checkpointing
    - Multi-GPU training via Accelerate
    - Mixed precision training (optional)

    Args:
        diffusion_model: GaussianDiffusion1D model to train
        dataset: PyTorch Dataset containing training sequences
        train_batch_size: Batch size for training (default: 16)
        gradient_accumulate_every: Number of steps to accumulate gradients (default: 1)
        train_lr: Learning rate (default: 1e-4)
        train_num_steps: Total number of training steps (default: 100000)
        ema_update_every: Update EMA every N steps (default: 10)
        ema_decay: EMA decay rate (default: 0.995)
        adam_betas: Beta parameters for Adam optimizer (default: (0.9, 0.99))
        save_and_sample_every: Save checkpoint and generate samples every N steps (default: 1000)
        num_samples: Number of samples to generate (must have integer sqrt) (default: 25)
        results_folder: Folder to save checkpoints and samples (default: './results')
        amp: Enable automatic mixed precision (default: False)
        mixed_precision_type: Type of mixed precision - 'fp16' or 'bf16' (default: 'fp16')
        split_batches: Split batches across GPUs in distributed training (default: True)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
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
        max_grad_norm = 1.
    ):
        super().__init__()

        # accelerator - handles multi-GPU, mixed precision, etc.

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)  # Create infinite iterator over dataloader

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # EMA (Exponential Moving Average) of model weights for better sample quality
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        """Get the device being used for training (CPU or GPU)."""
        return self.accelerator.device

    def save(self, milestone):
        """
        Save a checkpoint of the model, optimizer, and training state.

        Only saves on the main process in distributed training.

        Args:
            milestone: Identifier for this checkpoint (typically step // save_every)
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
        Load a checkpoint and restore model, optimizer, and training state.

        Args:
            milestone: Identifier of the checkpoint to load
        """
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        # Unwrap model from accelerator wrapper before loading state
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        # Restore training state
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # Restore gradient scaler if using mixed precision
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        """
        Main training loop.

        Trains the diffusion model with:
        - Gradient accumulation for effective larger batch sizes
        - Gradient clipping for stability
        - EMA of model weights for better sample quality
        - Periodic sampling and checkpointing
        - Multi-GPU support via Accelerate

        The loop continues until reaching train_num_steps.
        """
        accelerator = self.accelerator
        device = accelerator.device

        # Progress bar (only show on main process in distributed training)
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                # Gradient accumulation loop
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    # Forward pass with automatic mixed precision if enabled
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every  # Scale loss for accumulation
                        total_loss += loss.item()

                    # Backward pass (accumulates gradients)
                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                # Synchronize across all processes
                accelerator.wait_for_everyone()
                # Clip gradients to prevent exploding gradients
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.opt.step()
                self.opt.zero_grad()

                # Synchronize again before moving to next step
                accelerator.wait_for_everyone()

                self.step += 1

                # Update EMA and handle sampling/checkpointing (main process only)
                if accelerator.is_main_process:
                    self.ema.update()  # Update exponential moving average

                    # Periodic sampling and saving
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        # Generate samples using EMA model (better quality)
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            # Generate samples in batches to avoid OOM
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        # Save samples and checkpoint
                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
