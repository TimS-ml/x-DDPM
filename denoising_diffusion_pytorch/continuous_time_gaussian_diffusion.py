"""
Continuous-Time Gaussian Diffusion Models

This module implements a continuous-time formulation of Gaussian diffusion models, which
differs from the discrete-timestep approach in several fundamental ways:

Continuous vs. Discrete Timesteps:
---------------------------------
Traditional diffusion models (DDPM) operate with discrete timesteps t ∈ {1, 2, ..., T}
and use discrete noise schedules. In contrast, continuous-time diffusion:
  - Uses continuous time t ∈ [0, 1] where t=0 is clean data and t=1 is pure noise
  - Parameterizes the diffusion process using continuous functions
  - Allows for arbitrary sampling steps during inference (not tied to training schedule)
  - Provides a more principled mathematical framework based on stochastic differential equations (SDEs)

Mathematical Framework:
----------------------
The continuous diffusion process is characterized by the signal-to-noise ratio (SNR):
  SNR(t) = α²(t) / σ²(t)

where α(t) and σ(t) define the forward diffusion process:
  q(x_t | x_0) = N(x_t; α(t)x_0, σ²(t)I)

Instead of learning separate α and σ schedules, we learn log(SNR) directly, which:
  - Ensures monotonic decrease of signal over time (noise increases)
  - Provides numerical stability
  - Simplifies the mathematics of score matching

Score Matching and Denoising:
----------------------------
The model learns to predict the noise ε added to the data, which is equivalent to
learning the score function ∇log p(x_t). The loss is a continuous-time version
of the denoising score matching objective, weighted by the SNR.

Key Advantages:
--------------
1. Flexible sampling: Can use any number of steps at inference (not tied to training)
2. Better theoretical foundation via SDE/ODE formulations
3. Smoother noise schedules with no discretization artifacts
4. Natural connection to score-based generative models
5. Optional learned noise schedules that adapt to the data

References:
----------
- Variational Diffusion Models (Kingma et al., 2021): https://openreview.net/forum?id=2LdBqxc1Yv
- Score-Based Generative Modeling (Song et al., 2021)
- Katherine Crowson's implementation: https://github.com/crowsonkb/v-diffusion-jax
"""

import math
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    """
    Check if a value is not None.

    Args:
        val: Any value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None

def default(val, d):
    """
    Return val if it exists, otherwise return default value d.

    Args:
        val: The primary value to return if it exists
        d: Default value or callable that returns default value

    Returns:
        val if it exists, otherwise d (or d() if d is callable)
    """
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalize image from [0, 1] range to [-1, 1] range.

    This is commonly used to preprocess images before feeding them to the diffusion model,
    as many models work better with zero-centered data.

    Args:
        img: Tensor with values in [0, 1]

    Returns:
        Tensor with values in [-1, 1]
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize tensor from [-1, 1] range back to [0, 1] range.

    This is used to convert model outputs back to standard image range [0, 1].

    Args:
        t: Tensor with values in [-1, 1]

    Returns:
        Tensor with values in [0, 1]
    """
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    """
    Right-pad the dimensions of tensor t to match the number of dimensions in x.

    This is crucial for broadcasting operations in diffusion models where we need to
    apply scalar values (like time or log_snr) to multi-dimensional tensors (like images).

    For example, if x has shape (batch, channels, height, width) and t has shape (batch,),
    this will reshape t to (batch, 1, 1, 1) so it can be broadcast with x.

    Args:
        x: Reference tensor whose dimensionality we want to match
        t: Tensor to be padded with dimensions

    Returns:
        Tensor t with additional singleton dimensions added on the right
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# neural net helpers

class Residual(nn.Module):
    """
    Residual wrapper that adds the input to the output of a function.

    This implements a residual connection: output = x + fn(x)
    Residual connections help with gradient flow and are fundamental to deep networks.

    Args:
        fn: A neural network module or function to wrap
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            x + fn(x)
        """
        return x + self.fn(x)

class MonotonicLinear(nn.Module):
    """
    A linear layer constrained to be monotonically increasing.

    This is achieved by taking the absolute value of both weights and biases,
    ensuring all parameters are positive. This is crucial for learned noise schedules
    where we need to guarantee monotonic behavior (log_snr must decrease with time).

    The forward pass computes: y = |W|x + |b|

    Args:
        *args: Arguments passed to nn.Linear (e.g., in_features, out_features)
        **kwargs: Keyword arguments passed to nn.Linear
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        """
        Forward pass with positive-constrained weights and biases.

        Args:
            x: Input tensor

        Returns:
            Linear transformation with absolute value of weights and biases
        """
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())

# continuous schedules

# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    """
    Numerically stable logarithm with clamping to avoid log(0).

    Args:
        t: Input tensor
        eps: Minimum value to clamp to before taking log (default: 1e-20)

    Returns:
        log(max(t, eps))
    """
    return torch.log(t.clamp(min = eps))

def beta_linear_log_snr(t):
    """
    Compute log(SNR) for a linear beta schedule in continuous time.

    This approximates the original DDPM linear schedule in continuous form.
    The schedule uses: β(t) = 1e-4 + 10t²

    The log-SNR is derived from the cumulative noise added up to time t:
    log(SNR(t)) = -log(exp(∫β(s)ds) - 1)

    Args:
        t: Continuous time in [0, 1], where 0 is no noise and 1 is maximum noise

    Returns:
        log(SNR) at time t, a negative value that decreases as t increases
    """
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):
    """
    Compute log(SNR) for a cosine schedule in continuous time.

    The cosine schedule is defined as:
    α(t) = cos((t + s)/(1 + s) * π/2)

    This provides a smoother noise schedule compared to linear, with more gentle
    transitions at the start and end of the diffusion process.

    Args:
        t: Continuous time in [0, 1]
        s: Small offset to prevent singularity at t=0 (default: 0.008)

    Returns:
        log(SNR) at time t, computed as -log(cos((t+s)/(1+s)*π/2)^-2 - 1)
    """
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

class learned_noise_schedule(nn.Module):
    """
    Learnable noise schedule parameterized by a monotonic neural network.

    Instead of using fixed noise schedules (linear or cosine), this learns an optimal
    schedule from the data. The network is constrained to be monotonic (using MonotonicLinear
    layers) to ensure log_snr decreases with time.

    Mathematical Details:
    --------------------
    The network learns a function f(t) that is then normalized to produce log_snr values
    in the range [log_snr_max, log_snr_min]:

    1. Compute f(t), f(0), and f(1) using the monotonic network
    2. Normalize: g(t) = (f(t) - f(0)) / (f(1) - f(0))  # Now in [0, 1]
    3. Scale: log_snr(t) = log_snr_min * g(t) + log_snr_max * (1 - g(t))

    This ensures:
    - log_snr(0) ≈ log_snr_max (clean data, high SNR)
    - log_snr(1) ≈ log_snr_min (pure noise, low SNR)
    - Monotonic decrease from 0 to 1

    Architecture:
    ------------
    - MonotonicLinear layers ensure f(t) is monotonic
    - Residual connection allows learning deviations from linear schedule
    - Sigmoid activation keeps values bounded

    Gradient Fractioning:
    --------------------
    The frac_gradient parameter controls how much of the gradient flows back to the
    noise schedule network. Values < 1 slow down learning of the schedule relative
    to the denoising model, which can improve stability.

    Reference: Variational Diffusion Models, Sections H and I.2
    """

    def __init__(
        self,
        *,
        log_snr_max,
        log_snr_min,
        hidden_dim = 1024,
        frac_gradient = 1.
    ):
        """
        Initialize the learned noise schedule.

        Args:
            log_snr_max: Maximum log(SNR) at t=0 (clean data)
            log_snr_min: Minimum log(SNR) at t=1 (pure noise)
            hidden_dim: Hidden dimension of the MLP (default: 1024)
            frac_gradient: Fraction of gradient to pass through (0 to 1).
                          Use < 1 to slow down schedule learning (default: 1.0)
        """
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        # Monotonic network: ensures output increases monotonically with input
        # This is critical so log_snr decreases monotonically with time
        self.net = nn.Sequential(
            Rearrange('... -> ... 1'),
            MonotonicLinear(1, 1),
            Residual(nn.Sequential(
                MonotonicLinear(1, hidden_dim),
                nn.Sigmoid(),
                MonotonicLinear(hidden_dim, 1)
            )),
            Rearrange('... 1 -> ...'),
        )

        self.frac_gradient = frac_gradient

    def forward(self, x):
        """
        Compute log(SNR) for the given time values.

        The network output is normalized so that:
        - f(0) maps to log_snr_max
        - f(1) maps to log_snr_min
        - Interpolation is monotonic

        Args:
            x: Time values in [0, 1], shape (batch,) or (batch, ...)

        Returns:
            log(SNR) values at the given times, same shape as input
        """
        frac_gradient = self.frac_gradient
        device = x.device

        # Evaluate network at boundaries to normalize
        out_zero = self.net(torch.zeros_like(x))
        out_one =  self.net(torch.ones_like(x))

        # Evaluate network at input times
        x = self.net(x)

        # Normalize to [0, 1] range, then scale to [log_snr_max, log_snr_min]
        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept

        # Apply gradient fractioning: only frac_gradient flows through, rest is detached
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)

class ContinuousTimeGaussianDiffusion(nn.Module):
    """
    Continuous-time Gaussian diffusion model for image generation.

    This class implements the continuous-time formulation of diffusion models where time
    is treated as a continuous variable t ∈ [0, 1] rather than discrete steps. The key
    innovation is parameterizing the diffusion process via log(SNR) which provides:

    1. Unified Framework: Both forward and reverse processes defined by continuous functions
    2. Flexible Sampling: Use any number of steps during generation (not tied to training)
    3. Better Theoretical Foundation: Direct connection to SDEs and score matching
    4. Improved Stability: Monotonic log(SNR) schedule ensures well-defined diffusion

    Forward Process (Adding Noise):
    ------------------------------
    Given clean data x_0, the forward process at continuous time t is:
        q(x_t | x_0) = N(x_t; α(t)x_0, σ²(t)I)

    where α(t) and σ(t) are derived from log(SNR):
        α²(t) = sigmoid(log_snr(t))
        σ²(t) = sigmoid(-log_snr(t))

    Reverse Process (Denoising):
    ---------------------------
    The model learns to predict the noise ε that was added:
        x̂_0 = (x_t - σ(t)ε_θ(x_t, t)) / α(t)

    The reverse step from time t to t_next < t uses:
        p(x_{t_next} | x_t) = N(x_{t_next}; μ_θ(x_t, t, t_next), Σ(t, t_next))

    Training Objective:
    ------------------
    Simple MSE loss between predicted and actual noise:
        L = E[||ε - ε_θ(x_t, t)||²]

    Optional min-SNR weighting can be applied to balance learning across timesteps.

    Key Differences from DDPM:
    -------------------------
    - DDPM: Discrete t ∈ {1,...,T}, fixed schedule, T sampling steps required
    - Continuous: t ∈ [0,1], learned schedules possible, arbitrary sampling steps

    Args:
        model: The neural network (U-Net) that predicts noise
        image_size: Size of square images (e.g., 64 for 64x64)
        channels: Number of image channels (default: 3 for RGB)
        noise_schedule: Type of noise schedule - 'linear', 'cosine', or 'learned' (default: 'linear')
        num_sample_steps: Number of steps to use during sampling/generation (default: 500)
        clip_sample_denoised: Whether to clamp predicted x_0 to [-1, 1] (default: True)
        learned_schedule_net_hidden_dim: Hidden dimension for learned schedule network (default: 1024)
        learned_noise_schedule_frac_gradient: Gradient fraction for learned schedule (default: 1.0)
        min_snr_loss_weight: Whether to use min-SNR loss weighting (default: False)
        min_snr_gamma: Gamma parameter for min-SNR weighting (default: 5)
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        noise_schedule = 'linear',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,   # between 0 and 1, determines what percentage of gradients go back, so one can update the learned noise schedule more slowly
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        super().__init__()
        # Model must support continuous time conditioning via sinusoidal embeddings
        assert model.random_or_learned_sinusoidal_cond
        assert not model.self_condition, 'not supported yet'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # continuous noise schedule related stuff
        # The log_snr function maps time t ∈ [0,1] to log(signal/noise ratio)

        if noise_schedule == 'linear':
            # Approximates DDPM's linear beta schedule in continuous time
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            # Smoother schedule with cosine interpolation
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            # Learn optimal schedule from data via monotonic neural network
            # Initialize bounds from linear schedule's endpoints
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # min-SNR loss weighting proposed in https://arxiv.org/abs/2303.09556
        # Helps balance loss across different noise levels

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        """
        Get the device (CPU/GPU) where the model parameters are stored.

        Returns:
            torch.device: Device of the model
        """
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        """
        Compute the mean and variance of the reverse diffusion step p(x_{t_next} | x_t).

        This implements the reverse process of the continuous diffusion, transitioning from
        a noisier state at time t to a less noisy state at time_next (where time_next < time).

        Mathematical Details:
        --------------------
        Given x_t, we:
        1. Predict the noise: ε_θ(x_t, t)
        2. Estimate x_0: x̂_0 = (x_t - σ(t)ε_θ) / α(t)
        3. Compute posterior mean using the corrected equation from reviewer feedback:
           μ(x_t, t, t_next) = α(t_next) * [x_t * (1-c)/α(t) + c * x̂_0]

        where c = -expm1(log_snr(t) - log_snr(t_next)) controls the interpolation.

        The variance is: Σ(t, t_next) = σ²(t_next) * c

        Note: The original paper had an error (missing sigma term), which was corrected
        in the review discussion: https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        Args:
            x: Noisy image at time t, shape (batch, channels, height, width)
            time: Current time t (scalar or tensor)
            time_next: Next time step t_next < t (scalar or tensor)

        Returns:
            tuple: (model_mean, posterior_variance)
                - model_mean: Mean of p(x_{t_next} | x_t), shape same as x
                - posterior_variance: Variance of the posterior, scalar or shape (batch,)
        """
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        # Compute log(SNR) at current and next times
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)

        # Interpolation coefficient: c = 1 - exp(log_snr - log_snr_next)
        c = -expm1(log_snr - log_snr_next)

        # Convert log(SNR) to α² and σ² using sigmoid
        # α²(t) = sigmoid(log_snr) = 1 / (1 + exp(-log_snr))
        # σ²(t) = sigmoid(-log_snr) = 1 / (1 + exp(log_snr))
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        # Take square roots to get α, σ
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        # Predict noise at current time
        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred_noise = self.model(x, batch_log_snr)

        if self.clip_sample_denoised:
            # Estimate clean image x_0 from noisy x_t and predicted noise
            x_start = (x - sigma * pred_noise) / alpha

            # Clamp to valid range [-1, 1]
            # In Imagen paper, this was changed to dynamic thresholding for better quality
            x_start.clamp_(-1., 1.)

            # Compute mean using estimated x_0
            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            # Alternative formulation without explicit x_0 estimation
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        # Posterior variance
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        """
        Perform a single reverse diffusion step from time t to time_next.

        This samples from p(x_{t_next} | x_t) by:
        1. Computing mean and variance using p_mean_variance
        2. Adding Gaussian noise scaled by the variance (except at the final step)

        The stochasticity comes from the added noise, which is necessary to match
        the reverse diffusion process. At the final step (time_next=0), we return
        just the mean to get a deterministic final output.

        Args:
            x: Noisy image at time t, shape (batch, channels, height, width)
            time: Current time t
            time_next: Next time t_next < t (or 0 for final step)

        Returns:
            Denoised image at time_next, shape same as x
        """
        batch, *_, device = *x.shape, x.device

        # Get mean and variance of reverse step
        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        # At final step (time_next = 0), return mean without noise
        if time_next == 0:
            return model_mean

        # Sample from Gaussian: x_{t_next} ~ N(mean, variance)
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        Generate samples by iteratively denoising from pure noise to clean images.

        This is the main sampling loop that:
        1. Starts with pure Gaussian noise at t=1
        2. Iteratively denoises through num_sample_steps steps
        3. Returns clean images at t=0

        The key advantage of continuous-time formulation is that num_sample_steps
        can be chosen independently at inference time - it doesn't need to match
        any training hyperparameter.

        Args:
            shape: Shape of samples to generate, (batch, channels, height, width)

        Returns:
            Generated images in [0, 1] range, shape same as input shape
        """
        batch = shape[0]

        # Start from pure noise at t=1
        img = torch.randn(shape, device = self.device)

        # Create linearly spaced timesteps from t=1 (noise) to t=0 (clean)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        # Iteratively denoise
        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        # Final cleanup: clamp to valid range and convert to [0, 1]
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        """
        Generate a batch of images from random noise.

        This is a convenience wrapper around p_sample_loop that sets up the
        shape based on the configured image size and channels.

        Args:
            batch_size: Number of images to generate (default: 16)

        Returns:
            Generated images, shape (batch_size, channels, image_size, image_size)
            Values are in [0, 1] range
        """
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # training related functions - noise prediction

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, times, noise = None):
        """
        Sample from the forward diffusion process q(x_t | x_0).

        This implements the forward diffusion at continuous time t, which adds Gaussian
        noise to clean images according to the noise schedule. The noising follows:

            x_t = α(t) * x_0 + σ(t) * ε

        where ε ~ N(0, I) is standard Gaussian noise, and α(t), σ(t) are derived
        from the log(SNR) schedule.

        Mathematical Details:
        --------------------
        From log(SNR) = log(α²/σ²), we get:
            α²(t) = sigmoid(log_snr(t))
            σ²(t) = sigmoid(-log_snr(t)) = 1 - α²(t)

        This ensures α²(t) + σ²(t) = 1, which is a common parameterization
        (though not strictly necessary for all diffusion formulations).

        Args:
            x_start: Clean images, shape (batch, channels, height, width)
            times: Diffusion times in [0, 1], shape (batch,)
            noise: Optional pre-sampled noise. If None, samples N(0, I)

        Returns:
            tuple: (x_noised, log_snr)
                - x_noised: Noised images at time t, shape same as x_start
                - log_snr: log(SNR) values at the given times, shape (batch,)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Get log(SNR) at the specified times
        log_snr = self.log_snr(times)

        # Compute α(t) and σ(t) from log(SNR)
        # Need to broadcast to image dimensions for element-wise multiplication
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())

        # Forward diffusion: x_t = α(t) * x_0 + σ(t) * ε
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    def random_times(self, batch_size):
        """
        Sample random continuous times uniformly from [0, 1].

        In continuous-time diffusion, times are sampled uniformly during training,
        unlike discrete diffusion where we sample from {1, ..., T}.

        Args:
            batch_size: Number of time samples to generate

        Returns:
            Tensor of shape (batch_size,) with values uniformly distributed in [0, 1]
        """
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise = None):
        """
        Compute the training loss for noise prediction.

        This implements the denoising score matching objective, which trains the model
        to predict the noise ε that was added during the forward process.

        Loss Formulation:
        ----------------
        Basic loss: L = E[||ε - ε_θ(x_t, t)||²]

        where:
        - ε is the actual noise added
        - ε_θ(x_t, t) is the model's noise prediction
        - x_t = α(t)x_0 + σ(t)ε is the noised image

        Optional Min-SNR Weighting:
        --------------------------
        If min_snr_loss_weight is enabled, we apply SNR-based weighting:
            w(t) = min(SNR(t), gamma) / SNR(t)

        This helps balance the loss across different noise levels, preventing the model
        from focusing too much on high-noise timesteps. See: https://arxiv.org/abs/2303.09556

        Args:
            x_start: Clean images in [-1, 1] range, shape (batch, channels, height, width)
            times: Diffusion times in [0, 1], shape (batch,)
            noise: Optional pre-sampled noise

        Returns:
            Scalar loss value (mean over batch)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Apply forward diffusion to get noised images
        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)

        # Predict the noise
        model_out = self.model(x, log_snr)

        # Compute MSE loss between predicted and actual noise
        losses = F.mse_loss(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')  # Average over spatial dimensions

        # Apply optional min-SNR loss weighting
        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            # weight = min(SNR, gamma) / SNR
            # min-SNR (arxiv.org/abs/2303.09556): clip SNR to a MAXIMUM of gamma so
            # high-SNR (low-noise) timesteps get down-weighted. Upstream fix: use
            # clamp(max=...) here — clamp(min=...) inverts the intended weighting.
            loss_weight = snr.clamp(max = self.min_snr_gamma) / snr
            losses = losses * loss_weight

        return losses.mean()

    def forward(self, img, *args, **kwargs):
        """
        Forward pass for training.

        This is called during training to compute the loss for a batch of images.
        It:
        1. Validates image dimensions
        2. Normalizes images to [-1, 1]
        3. Samples random timesteps
        4. Computes the denoising loss

        Args:
            img: Batch of images in [0, 1] range, shape (batch, channels, height, width)
            *args: Additional arguments passed to p_losses
            **kwargs: Additional keyword arguments passed to p_losses

        Returns:
            Scalar loss value for the batch
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # Sample random continuous times for each image in the batch
        times = self.random_times(b)

        # Normalize images from [0, 1] to [-1, 1]
        img = normalize_to_neg_one_to_one(img)

        # Compute and return loss
        return self.p_losses(img, times, *args, **kwargs)
