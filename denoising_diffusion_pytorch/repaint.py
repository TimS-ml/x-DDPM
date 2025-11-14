"""
RePaint: Inpainting using Denoising Diffusion Probabilistic Models

This module implements the RePaint algorithm for image inpainting using diffusion models.
RePaint is a powerful technique that enables conditional image generation by leveraging
pre-trained unconditional diffusion models for the task of image inpainting.

What is RePaint?
----------------
RePaint (Resampling Paint) is an algorithm proposed in "RePaint: Inpainting using Denoising
Diffusion Probabilistic Models" (https://arxiv.org/abs/2201.09865) that performs image
inpainting without requiring any additional training. It works by conditioning a pre-trained
diffusion model on known pixels (masked regions) during the reverse diffusion process.

How RePaint Works:
-----------------
1. **Masked Conditioning**: During sampling, RePaint conditions on the known (unmasked)
   pixels by replacing them with noisy versions of the ground truth at each timestep.

2. **Resampling Strategy**: To improve coherence between known and unknown regions,
   RePaint introduces a resampling mechanism that jumps back a few timesteps and
   re-denoises multiple times. This helps the generated content better harmonize
   with the known pixels.

3. **Forward-Backward Jumps**: At regular intervals during reverse diffusion, the
   algorithm performs multiple forward-backward jumps (adding noise then denoising)
   to refine the boundary between known and unknown regions.

Key Parameters:
--------------
- mask: Binary mask where 1 indicates known pixels, 0 indicates pixels to inpaint
- gt: Ground truth image containing the known pixel values
- resample: Whether to enable the resampling mechanism
- resample_iter: Number of resampling iterations at each resampling step
- resample_jump: Number of timesteps to jump back during resampling
- resample_every: Frequency of resampling (every N timesteps)

The algorithm enables high-quality inpainting by ensuring that:
1. Known regions remain faithful to the ground truth
2. Unknown regions are generated coherently with the known context
3. Boundaries between regions are seamless and natural
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

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

# constants

# Named tuple to store model predictions during diffusion sampling
# pred_noise: predicted noise at timestep t
# pred_x_start: predicted clean image x_0
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    """Check if a value is not None."""
    return x is not None

def default(val, d):
    """
    Return val if it exists, otherwise return d.
    If d is callable, call it to get the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    """
    Cast input to a tuple of specified length.
    If already a tuple, return as is. Otherwise, repeat the value.
    """
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    """Check if numer is evenly divisible by denom."""
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    """Identity function that returns input unchanged."""
    return t

def cycle(dl):
    """
    Infinitely cycle through a dataloader.
    Useful for training loops that need continuous data.
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
    Returns a list where most elements are divisor, with remainder at the end.
    Example: num_to_groups(10, 3) -> [3, 3, 3, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    """Convert PIL image to specified type (e.g., 'RGB', 'L') if not already."""
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalize image from [0, 1] to [-1, 1].
    This is the standard normalization for diffusion models.
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize image from [-1, 1] back to [0, 1].
    Used to convert model outputs back to valid image range.
    """
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    """
    2x upsampling block using nearest neighbor interpolation followed by convolution.
    Used in the decoder path of the U-Net.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    """
    2x downsampling block using space-to-depth rearrangement followed by 1x1 convolution.
    Rearranges (h, w) spatial dimensions into channels: (h/2, w/2, 4*c).
    Used in the encoder path of the U-Net.
    """
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.
    Normalizes using RMS instead of mean and variance, which is more efficient.
    Commonly used in modern transformer and diffusion architectures.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    """
    Sinusoidal positional embeddings for encoding timesteps.
    Maps continuous timestep values to high-dimensional embeddings using sin/cos functions.
    This allows the model to understand the relative position in the diffusion process.
    """
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta  # Base for the sinusoidal frequencies

    def forward(self, x):
        """
        Args:
            x: Timestep values, shape (batch_size,)
        Returns:
            Positional embeddings, shape (batch_size, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        # Create exponentially spaced frequencies
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Apply frequencies to timesteps
        emb = x[:, None] * emb[None, :]
        # Concatenate sin and cos embeddings
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """
    Random or learned sinusoidal positional embeddings.
    Following @crowsonkb's approach with random (optionally learned) sinusoidal embeddings.
    Reference: https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8

    Can use either random fixed frequencies or learned frequencies for more flexibility.
    """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        # Initialize random weights; freeze if is_random=True, otherwise learnable
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        """
        Args:
            x: Input timesteps, shape (batch_size,)
        Returns:
            Fourier features concatenated with original input, shape (batch_size, dim+1)
        """
        x = rearrange(x, 'b -> b 1')
        # Apply learned/random frequencies
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # Create Fourier features with sin and cos
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # Concatenate with original input for additional context
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    """
    Basic convolutional block with normalization and activation.
    Supports adaptive normalization via scale_shift for time conditioning.
    """
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()  # Smooth activation function (also called Swish)

    def forward(self, x, scale_shift = None):
        """
        Args:
            x: Input tensor
            scale_shift: Optional tuple of (scale, shift) for adaptive normalization
        """
        x = self.proj(x)
        x = self.norm(x)

        # Apply adaptive normalization if time conditioning is provided
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(Module):
    """
    Residual block with time embedding conditioning.
    Uses two convolution blocks with a residual connection.
    Time embeddings are injected via adaptive normalization (scale and shift).
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
        # Residual connection - adjust channels if needed
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        """
        Args:
            x: Input feature map
            time_emb: Time step embeddings for conditioning
        Returns:
            Output with residual connection
        """
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # Process time embedding to get scale and shift parameters
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # Split into scale and shift for adaptive normalization
            scale_shift = time_emb.chunk(2, dim = 1)

        # First block with time conditioning
        h = self.block1(x, scale_shift = scale_shift)

        # Second block (no time conditioning here)
        h = self.block2(h)

        # Add residual connection
        return h + self.res_conv(x)

class LinearAttention(Module):
    """
    Linear attention mechanism with O(n) complexity instead of O(n^2).
    Uses kernel trick to avoid explicit attention matrix computation.
    More efficient for processing spatial features in diffusion models.
    Includes learnable memory key-value pairs for enhanced expressiveness.
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

        # Learnable memory key-value pairs (persistent context)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input feature map, shape (batch, channels, height, width)
        Returns:
            Attention output with same shape as input
        """
        b, c, h, w = x.shape

        x = self.norm(x)

        # Generate queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # Add learnable memory key-value pairs
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        # Apply softmax to make it linear attention
        q = q.softmax(dim = -2)  # Normalize over feature dimension
        k = k.softmax(dim = -1)  # Normalize over spatial dimension

        q = q * self.scale

        # Compute context matrix (key-value interaction)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Apply context to queries
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    """
    Standard multi-head self-attention mechanism.
    Uses O(n^2) complexity but provides full attention between all positions.
    Supports flash attention for improved efficiency when available.
    Includes learnable memory key-value pairs for enhanced context.
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
        self.attend = Attend(flash = flash)  # Handles attention computation

        # Learnable memory tokens for persistent context across all samples
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input feature map, shape (batch, channels, height, width)
        Returns:
            Attention output with same shape as input
        """
        b, c, h, w = x.shape

        x = self.norm(x)

        # Generate queries, keys, values from input
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        # Prepend learnable memory key-value pairs
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # Compute attention (scaled dot-product or flash attention)
        out = self.attend(q, k, v)

        # Reshape back to spatial format
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    """
    U-Net architecture for diffusion models.

    The U-Net is the core denoising network in diffusion models. It takes a noisy image
    and a timestep as input, and predicts either the noise, the clean image, or the
    velocity (v-parameterization) depending on the objective.

    Architecture:
    - Encoder path: Progressively downsamples the image while increasing channels
    - Bottleneck: Processes at the lowest resolution with full attention
    - Decoder path: Progressively upsamples while decreasing channels
    - Skip connections: Concatenates encoder features to decoder for detail preservation
    - Time conditioning: Timestep embeddings are injected into residual blocks

    For RePaint inpainting, this U-Net processes the noisy image at each timestep,
    predicting how to denoise it while the masked regions are separately conditioned
    on the ground truth.
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
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """
    Extract values from array a at indices t, and reshape for broadcasting.

    This is a helper function for gathering precomputed diffusion parameters
    (like alphas, betas) at specific timesteps and reshaping them to broadcast
    with batched tensors.

    Args:
        a: 1D array of precomputed values (e.g., alphas, betas)
        t: Timestep indices, shape (batch,)
        x_shape: Target shape for broadcasting (e.g., image shape)

    Returns:
        Extracted values reshaped to (batch, 1, 1, 1, ...) for broadcasting
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

class GaussianDiffusion(Module):
    """
    Gaussian Diffusion Probabilistic Model with RePaint Inpainting Support.

    This class implements the diffusion process for training and sampling, including
    the RePaint algorithm for image inpainting. It handles:

    1. **Forward Process (Training)**: Gradually adds Gaussian noise to images
    2. **Reverse Process (Sampling)**: Iteratively denoises to generate images
    3. **RePaint Inpainting**: Conditions on known pixels during reverse diffusion

    RePaint Implementation:
    ----------------------
    The RePaint algorithm is implemented in the p_sample() and p_sample_loop() methods.
    At each reverse diffusion step:
    - Known pixels (mask=1) are replaced with noisy versions of ground truth
    - Unknown pixels (mask=0) are denoised by the model
    - Resampling performs forward-backward jumps to harmonize boundaries

    Key Components:
    - model: U-Net that predicts noise/x0/v at each timestep
    - Noise schedules: Control the noise level at each timestep (linear/cosine/sigmoid)
    - Sampling: DDPM or DDIM sampling strategies
    - Inpainting: Mask-based conditioning with optional resampling
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
        min_snr_gamma = 5
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

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

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

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

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
        Predict clean image x_0 from noisy image x_t and predicted noise.
        Used when objective is 'pred_noise'.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        Predict noise from noisy image x_t and clean image x_0.
        Inverse of predict_start_from_noise.
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """
        Compute velocity v for v-parameterization.
        v = sqrt(alpha_t) * noise - sqrt(1-alpha_t) * x_0
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict clean image x_0 from noisy image x_t and predicted velocity v.
        Used when objective is 'pred_v'.
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute posterior q(x_{t-1} | x_t, x_0).
        Returns mean, variance, and log variance of the posterior distribution.
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
        Get model predictions and convert to noise and x_start predictions.

        The model can predict different targets based on the objective:
        - 'pred_noise': Directly predict the noise
        - 'pred_x0': Directly predict the clean image
        - 'pred_v': Predict velocity (v-parameterization)

        This method converts any prediction to both noise and x_start.
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
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None, gt=None, mask=None):
        """
        Single denoising step with optional RePaint inpainting conditioning.

        This method implements Algorithm 1 from the RePaint paper (lines 5-7).
        It performs one reverse diffusion step, optionally conditioning on known pixels.

        RePaint Conditioning Process:
        ----------------------------
        1. If mask is provided, replace known pixels with noisy ground truth
        2. Denoise the full image (both known and unknown regions)
        3. At t=0, paste back the clean ground truth in known regions

        The key insight: By replacing known pixels with appropriately noisy versions
        of the ground truth at each timestep, we ensure the model generates content
        that is coherent with the context.

        Args:
            x: Current noisy image at timestep t, shape (batch, channels, H, W)
            t: Current timestep (int), ranges from num_timesteps-1 to 0
            x_self_cond: Self-conditioning input (previous prediction)
            gt: Ground truth image for inpainting, shape (batch, channels, H, W)
            mask: Binary mask, 1=known pixels, 0=pixels to inpaint, shape (batch, 1, H, W)

        Returns:
            pred_img: Denoised image at timestep t-1
            x_start: Predicted clean image (x_0)

        Reference: https://arxiv.org/abs/2201.09865
        """

        # ===== RePaint Step 1: Condition on Known Pixels =====
        # Replace known pixels with appropriately noisy versions of the ground truth
        if mask is not None:
            mask = mask.to(x.device)
            gt = normalize_to_neg_one_to_one(gt)
            # Get noise level for current timestep
            alpha_cumnprod_t = self.alphas_cumprod[t]

            # Create noisy version of ground truth at timestep t
            # Formula: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise
            gt_weight = torch.sqrt(alpha_cumnprod_t).to(x.device)
            gt_part = gt_weight * gt
            noise_weight = torch.sqrt(1 - alpha_cumnprod_t).to(x.device)
            noise_part = noise_weight * torch.randn_like(x,device=x.device)
            weighed_gt = gt_part + noise_part

            # Replace known pixels (mask=1) with noisy ground truth
            # Keep unknown pixels (mask=0) from current denoising state
            x = (mask * weighed_gt) + ((1 - mask) * x)

        # ===== RePaint Step 2: Perform Standard Denoising =====
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)

        # Get model predictions for mean and variance of p(x_{t-1} | x_t)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )

        # Sample from predicted distribution (no noise at final step)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        # ===== RePaint Step 3: Final Paste at t=0 =====
        if t==0 and mask is not None:
            # if t == 0, we use the ground-truth image if in-painting
            pred_img = (mask * gt) +  ((1 - mask) * pred_img)

        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape,
        return_all_timesteps=False,
        gt=None,
        mask=None,
        resample=True,
        resample_iter=10,
        resample_jump=3,
        resample_every=50,
    ):
        """
        Full reverse diffusion sampling loop with optional RePaint resampling.

        This method implements the complete RePaint Algorithm 1 from https://arxiv.org/abs/2201.09865
        It performs reverse diffusion from pure noise to a clean image, with optional
        mask-based conditioning and resampling for inpainting tasks.

        RePaint Resampling Strategy:
        ---------------------------
        The resampling mechanism (lines 9-13 of Algorithm 1) helps harmonize the boundary
        between known and unknown regions:

        1. **Main Loop**: Denoise from t=T to t=0, conditioning on known pixels
        2. **Resampling Trigger**: At regular intervals (every resample_every steps)
        3. **Forward Jump**: Add noise to jump forward by resample_jump timesteps
        4. **Reverse Denoise**: Denoise back those resample_jump steps
        5. **Iterate**: Repeat forward-backward jumps resample_iter times

        This forward-backward process acts like annealing, allowing the unknown regions
        to better adapt to the known context by repeatedly refining the boundary.

        Args:
            shape: Output image shape (batch, channels, height, width)
            return_all_timesteps: If True, return all intermediate steps
            gt: Ground truth image for inpainting (known pixels)
            mask: Binary mask, 1=known pixels, 0=pixels to inpaint
            resample: Whether to enable resampling (recommended for inpainting)
            resample_iter: Number of forward-backward iterations per resample
            resample_jump: Number of timesteps to jump in resampling
            resample_every: Frequency of resampling (every N timesteps)

        Returns:
            Generated/inpainted image(s), unnormalized to [0, 1]
        """
        batch, device = shape[0], self.device

        # Start from pure Gaussian noise
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # Main reverse diffusion loop: from t=T-1 to t=0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # Self-conditioning: use previous prediction as additional input
            self_cond = x_start if self.self_condition else None

            # Single denoising step (with mask conditioning if provided)
            img, x_start = self.p_sample(x=img, t=t, x_self_cond=self_cond, gt=gt, mask=mask)
            imgs.append(img)

            # Resampling loop: line 9 of Algorithm 1 in https://arxiv.org/pdf/2201.09865
            if resample is True and (t > 0) and (t % resample_every == 0 or t == 1) and mask is not None:
                # Perform multiple forward-backward jumps to refine the result
                for iter in tqdm(range(resample_iter), desc = 'resample loop', total = resample_iter):
                    # Forward jump: add noise to move forward by resample_jump timesteps
                    t = resample_jump
                    beta = self.betas[t]
                    # Add noise according to forward diffusion process
                    img = torch.sqrt(1 - beta) * img + torch.sqrt(beta) * torch.randn_like(img)

                    # Backward jump: denoise back those resample_jump timesteps
                    for j in reversed(range(0, resample_jump)):
                        img, x_start = self.p_sample(x=img, t=t, gt=gt, mask=mask)
                imgs.append(img)

        # Return final result (or all timesteps if requested)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)  # Convert from [-1, 1] back to [0, 1]
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

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

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(
        self,
        batch_size=16,
        return_all_timesteps=False,
        gt=None,
        mask=None,
        resample=True,
        resample_iter=10,
        resample_jump=10,
        resample_every=50,
    ):
        """
        High-level sampling interface with RePaint inpainting support.

        This is the main entry point for both unconditional generation and
        inpainting. When mask and gt are provided, it performs RePaint inpainting.

        Usage Examples:
        --------------
        # Unconditional generation:
        images = diffusion.sample(batch_size=4)

        # Image inpainting with RePaint:
        images = diffusion.sample(
            gt=ground_truth_image,      # Original image with known pixels
            mask=binary_mask,             # 1=keep, 0=inpaint
            resample=True,                # Enable RePaint resampling
            resample_iter=10,             # Iterations per resample
            resample_jump=10,             # Timesteps to jump
            resample_every=50             # Resample every N steps
        )

        Args:
            batch_size: Number of images to generate (ignored if mask is provided)
            return_all_timesteps: If True, return all denoising steps
            gt: Ground truth image for inpainting
            mask: Binary mask (1=known, 0=inpaint)
            resample: Enable RePaint resampling for better boundaries
            resample_iter: Number of forward-backward iterations per resample
            resample_jump: Number of timesteps to jump during resampling
            resample_every: Frequency of resampling triggers

        Returns:
            Generated or inpainted images in range [0, 1]
        """
        (h, w), channels = self.image_size, self.channels
        batch_size = mask.shape[0] if mask is not None else batch_size
        return self.p_sample_loop(
            shape=(batch_size, channels, h, w),
            return_all_timesteps=return_all_timesteps,
            gt=gt,
            mask=mask,
            resample=resample,
            resample_iter=resample_iter,
            resample_jump=resample_jump,
            resample_every=resample_every,
        )

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        """
        Forward diffusion process: add noise to clean images.

        Samples from q(x_t | x_0), the forward diffusion distribution.
        Formula: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Args:
            x_start: Clean images (x_0)
            t: Timesteps to sample at
            noise: Optional noise (generated if not provided)

        Returns:
            Noisy images at timestep t
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        """
        Compute training loss for diffusion model.

        The training objective depends on the model's prediction target:
        - 'pred_noise': Predict noise, minimize ||noise - predicted_noise||^2
        - 'pred_x0': Predict clean image, minimize ||x_0 - predicted_x0||^2
        - 'pred_v': Predict velocity, minimize ||v - predicted_v||^2

        Loss is weighted by SNR (signal-to-noise ratio) for better training stability.

        Args:
            x_start: Clean training images
            t: Random timesteps for training
            noise: Optional noise to add (generated if not provided)
            offset_noise_strength: Strength of offset noise (for darker/brighter images)

        Returns:
            Weighted MSE loss
        """
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    """
    Image dataset for diffusion model training.

    Loads images from a folder, applies transformations, and returns
    tensors ready for training. Supports various image formats and
    optional data augmentation.

    Args:
        folder: Path to folder containing images
        image_size: Target size for images (int or tuple)
        exts: List of image file extensions to load
        augment_horizontal_flip: Whether to randomly flip images horizontally
        convert_image_to: Image mode to convert to ('RGB', 'L', etc.)
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
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
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
    Trainer for diffusion models with support for distributed training.

    Handles the complete training loop including:
    - Data loading and augmentation
    - Model optimization with gradient accumulation
    - Exponential moving average (EMA) of model weights
    - Periodic sampling and checkpoint saving
    - FID score evaluation
    - Mixed precision training via accelerate

    The trained model can then be used for both unconditional generation
    and RePaint inpainting tasks.

    Args:
        diffusion_model: GaussianDiffusion model to train
        folder: Path to training images folder
        train_batch_size: Batch size per device
        gradient_accumulate_every: Gradient accumulation steps
        augment_horizontal_flip: Enable horizontal flip augmentation
        train_lr: Learning rate
        train_num_steps: Total training steps
        ema_update_every: Update EMA every N steps
        ema_decay: EMA decay rate
        adam_betas: Adam optimizer betas
        save_and_sample_every: Save checkpoint and sample every N steps
        num_samples: Number of samples to generate for visualization
        results_folder: Folder to save results and checkpoints
        amp: Enable automatic mixed precision
        mixed_precision_type: Precision type ('fp16' or 'bf16')
        split_batches: Split batches across devices
        convert_image_to: Image format ('RGB', 'L', etc.)
        calculate_fid: Compute FID scores during training
        inception_block_idx: Inception layer for FID computation
        max_grad_norm: Maximum gradient norm for clipping
        num_fid_samples: Number of samples for FID computation
        save_best_and_latest_only: Only save best and latest checkpoints
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

        # accelerator

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
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
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
        return self.accelerator.device

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
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

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

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
