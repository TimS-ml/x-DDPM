"""
V-Parameterization for Continuous Time Gaussian Diffusion Models

This module implements the v-parameterization approach for diffusion models, which offers
significant advantages over traditional noise prediction (epsilon) and data prediction (x0)
parameterizations.

What is V-Parameterization?
----------------------------
V-parameterization is an alternative objective for training diffusion models introduced in
"Progressive Distillation for Fast Sampling of Diffusion Models" (Salimans & Ho, 2022).
Instead of predicting noise (ε) or the original data (x₀), the model predicts a velocity-like
quantity v defined as:

    v = α_t * ε - σ_t * x₀

where:
    - α_t is the signal coefficient at time t
    - σ_t is the noise coefficient at time t
    - ε is the noise added to the data
    - x₀ is the original clean data

Relationship to Other Predictions:
----------------------------------
Given the noised data x_t = α_t * x₀ + σ_t * ε, we can derive:
    - x₀ prediction: x₀ = α_t * x_t - σ_t * v
    - Noise prediction: ε = α_t * v + σ_t * x_t

This means v-prediction is a linear combination of the noise and data predictions,
effectively blending both objectives.

Benefits of V-Parameterization:
-------------------------------
1. **Improved Training Stability**: The v-parameterization balances the scale of predictions
   across different noise levels, preventing the numerical instabilities that can occur when
   α_t or σ_t become very small.

2. **Better Convergence**: By operating in a more stable numerical range, models trained with
   v-parameterization often converge faster and more reliably.

3. **Enhanced Distillation**: The original paper showed that v-parameterization is crucial for
   progressive distillation, enabling faster sampling by distilling multi-step generation into
   fewer steps.

4. **Color Artifact Reduction**: As noted in the Imagen Video paper, v-parameterization
   significantly reduces color shifting artifacts in upsampling networks, making it especially
   valuable for super-resolution tasks.

5. **Continuous Time Formulation**: This implementation uses continuous time steps (t ∈ [0, 1])
   rather than discrete steps, providing smoother noise schedules and more flexible sampling.

References:
-----------
- Progressive Distillation: https://arxiv.org/abs/2202.00512
  "Progressive Distillation for Fast Sampling of Diffusion Models" (Salimans & Ho, 2022)

- Imagen Video: https://arxiv.org/abs/2210.02303
  "Imagen Video: High Definition Video Generation with Diffusion Models" (Ho et al., 2022)
  Section discussing v-parameterization benefits for video upsampling.

- Original DDPM: https://arxiv.org/abs/2006.11239
  "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

Implementation Details:
-----------------------
This implementation uses:
- Continuous time formulation with t ∈ [0, 1]
- Log-SNR (Signal-to-Noise Ratio) scheduling via cosine schedule
- Conversion between v-prediction and x₀ for sampling
- Variance-preserving diffusion process
"""

import math
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.amp import autocast

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
        val: Value to check
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

    Diffusion models typically work better with data normalized to [-1, 1]
    as it centers the data around zero.

    Args:
        img: Image tensor in [0, 1] range

    Returns:
        Image tensor in [-1, 1] range
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize tensor from [-1, 1] range back to [0, 1] range.

    Used to convert generated samples back to standard image range.

    Args:
        t: Tensor in [-1, 1] range

    Returns:
        Tensor in [0, 1] range
    """
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    """
    Pad dimensions of tensor t to match the number of dimensions in tensor x.

    This is used to broadcast scalar or low-dimensional tensors (like timesteps)
    to match the dimensions of image tensors for element-wise operations.

    For example, if x has shape [B, C, H, W] and t has shape [B], this will
    reshape t to [B, 1, 1, 1] so it can be broadcast with x.

    Args:
        x: Reference tensor with target number of dimensions
        t: Tensor to pad with singleton dimensions

    Returns:
        Tensor t with added singleton dimensions to match x.ndim
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# continuous schedules
# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    """
    Compute logarithm with numerical stability.

    Clamps input to a minimum value to avoid log(0) = -inf.

    Args:
        t: Input tensor
        eps: Minimum value for clamping (default: 1e-20)

    Returns:
        Logarithm of clamped input
    """
    return torch.log(t.clamp(min = eps))

def alpha_cosine_log_snr(t, s = 0.008):
    """
    Compute log signal-to-noise ratio (SNR) using cosine schedule.

    The cosine schedule provides a smooth noise schedule that approximates the
    original linear schedule from DDPM but in continuous time. It's defined as:

        log(SNR) = -log((cos((t + s) / (1 + s) * π/2))^(-2) - 1)

    This schedule ensures:
    - At t=0 (start of generation): high SNR (mostly signal, little noise)
    - At t=1 (end of forward process): low SNR (mostly noise, little signal)
    - Smooth transition between timesteps

    The offset s=0.008 prevents the SNR from becoming too extreme at the boundaries.

    Args:
        t: Time values in [0, 1], where 0 is clean and 1 is pure noise
        s: Small offset to prevent boundary issues (default: 0.008)

    Returns:
        Log SNR values for the given timesteps

    Reference:
        This schedule is based on the improved noise schedule from:
        "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
    """
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

class VParamContinuousTimeGaussianDiffusion(nn.Module):
    """
    V-Parameterization Diffusion Model with Continuous Time.

    This class implements diffusion models using v-parameterization in continuous time,
    as proposed in "Progressive Distillation for Fast Sampling of Diffusion Models"
    (https://arxiv.org/abs/2202.00512).

    Key Features:
    -------------
    1. **V-Parameterization**: Instead of predicting noise (ε) or clean data (x₀), the model
       predicts v = α_t * ε - σ_t * x₀, which offers:
       - Improved distillation over noise prediction objective
       - Better training stability across different noise levels
       - Reduced color shifting artifacts in upsampling tasks (noted in Imagen Video)

    2. **Continuous Time**: Uses continuous timesteps t ∈ [0, 1] instead of discrete steps,
       allowing for flexible sampling schedules and smoother transitions.

    3. **Log-SNR Scheduling**: Parameterizes the diffusion process using log signal-to-noise
       ratio (log-SNR), which provides better numerical stability than traditional α, β
       parameterizations.

    Mathematical Background:
    ------------------------
    Forward process (adding noise):
        x_t = α_t * x₀ + σ_t * ε

    V-prediction target:
        v = α_t * ε - σ_t * x₀

    Conversion from v to x₀ (used during sampling):
        x₀ = α_t * x_t - σ_t * v

    Where:
        - α_t = sqrt(sigmoid(log_snr_t)) is the signal coefficient
        - σ_t = sqrt(sigmoid(-log_snr_t)) is the noise coefficient
        - These ensure α_t² + σ_t² = 1 (variance preserving)

    References:
    -----------
    - Progressive Distillation: https://arxiv.org/abs/2202.00512
    - Imagen Video: https://arxiv.org/abs/2210.02303 (Section on v-parameterization benefits)
    """

    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 500,
        clip_sample_denoised = True,
    ):
        """
        Initialize the V-Parameterization Continuous Time Diffusion model.

        Args:
            model: The neural network model (U-Net) that predicts v.
                   Must have random_or_learned_sinusoidal_cond=True for continuous time.
                   Self-conditioning is not yet supported.
            image_size: Size of square images (height = width = image_size)
            channels: Number of image channels (default: 3 for RGB)
            num_sample_steps: Number of denoising steps during sampling (default: 500).
                             More steps generally produce higher quality but slower generation.
            clip_sample_denoised: Whether to clamp predicted x₀ to [-1, 1] range (default: True).
                                 Helps prevent artifacts but may reduce sample diversity.

        Raises:
            AssertionError: If model doesn't have random_or_learned_sinusoidal_cond enabled
            AssertionError: If model has self_condition enabled (not yet supported)
        """
        super().__init__()
        # V-parameterization requires continuous time conditioning
        assert model.random_or_learned_sinusoidal_cond
        # Self-conditioning not yet implemented for v-parameterization
        assert not model.self_condition, 'not supported yet'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # continuous noise schedule related stuff
        # Using cosine log-SNR schedule for smooth noise progression

        self.log_snr = alpha_cosine_log_snr

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised        

    @property
    def device(self):
        """
        Get the device (CPU/GPU) where the model parameters are located.

        Returns:
            torch.device: Device of the model parameters
        """
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        """
        Compute the mean and variance for the reverse diffusion process p(x_{t-1} | x_t).

        This implements the denoising step that takes a noisy image at timestep t and
        predicts the distribution for the slightly less noisy image at timestep t-1.

        The implementation follows the corrected equations from the Progressive Distillation
        paper. Note: A reviewer found an error in the original paper equation (missing sigma),
        which is corrected here following:
        https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        Mathematical Process:
        --------------------
        1. Get log-SNR values for current and next timesteps
        2. Compute coefficients c = -expm1(log_snr - log_snr_next)
        3. Calculate α and σ from log-SNR: α² = sigmoid(log_snr), σ² = sigmoid(-log_snr)
        4. Predict v using the model
        5. Convert v-prediction to x₀: x₀ = α_t * x_t - σ_t * v (from Appendix D)
        6. Compute posterior mean: μ = α_{t-1} * (x_t * (1-c)/α_t + c * x₀)
        7. Compute posterior variance: σ² = σ²_{t-1} * c

        Args:
            x: Noisy image at current timestep, shape (batch, channels, height, width)
            time: Current timestep value(s) in [0, 1]
            time_next: Next (less noisy) timestep value(s) in [0, 1]

        Returns:
            tuple: (model_mean, posterior_variance)
                - model_mean: Mean of p(x_{t-1} | x_t), shape same as x
                - posterior_variance: Variance of p(x_{t-1} | x_t), scalar or per-sample

        Note:
            The relationship between v, noise (ε), and clean data (x₀):
            - v = α_t * ε - σ_t * x₀  (training target)
            - x₀ = α_t * x_t - σ_t * v  (reconstruction during sampling)
            - ε = α_t * v + σ_t * x_t  (noise prediction if needed)
        """
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        # Compute log-SNR for current and next timesteps
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)

        # c is a coefficient used in the mean and variance calculations
        # expm1(x) = exp(x) - 1, more numerically stable than exp(x) - 1 for small x
        c = -expm1(log_snr - log_snr_next)

        # Convert log-SNR to squared alpha and sigma values
        # sigmoid(log_snr) = α², sigmoid(-log_snr) = σ²
        # This ensures α² + σ² = 1 (variance preserving)
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        # Take square roots to get actual alpha and sigma values
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        # Prepare log-SNR for model input (expand to batch dimension)
        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])

        # Get v-prediction from the model
        pred_v = self.model(x, batch_log_snr)

        # Convert v-prediction to x₀ prediction (shown in Appendix D in the paper)
        # Derivation: given x_t = α_t * x₀ + σ_t * ε and v = α_t * ε - σ_t * x₀
        # We can solve for x₀: x₀ = α_t * x_t - σ_t * v
        x_start = alpha * x - sigma * pred_v

        # Optionally clamp x₀ prediction to valid range to prevent artifacts
        if self.clip_sample_denoised:
            x_start.clamp_(-1., 1.)

        # Compute the mean of the posterior distribution p(x_{t-1} | x_t, x₀)
        # This interpolates between the current noisy image and predicted clean image
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        # Compute the variance of the posterior distribution
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        """
        Perform a single denoising step in the reverse diffusion process.

        This samples from p(x_{t-1} | x_t) by:
        1. Computing the mean and variance using p_mean_variance()
        2. If at the final step (time_next=0), return the mean directly
        3. Otherwise, sample by adding scaled Gaussian noise to the mean

        Args:
            x: Current noisy image, shape (batch, channels, height, width)
            time: Current timestep in [0, 1], where 1 is pure noise
            time_next: Next timestep in [0, 1], where 0 is clean image

        Returns:
            torch.Tensor: Denoised image at time_next, same shape as x

        Note:
            Uses @torch.no_grad() decorator for efficiency during sampling.
            At the final step (time_next=0), we return the mean without adding noise
            to get a deterministic clean image.
        """
        batch, *_, device = *x.shape, x.device

        # Get mean and variance for this denoising step
        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        # At the final step, return deterministic prediction (no noise)
        if time_next == 0:
            return model_mean

        # For other steps, sample from Gaussian distribution
        # x_{t-1} ~ N(model_mean, model_variance)
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        Generate samples by iteratively denoising from pure noise to clean images.

        This implements the full reverse diffusion process:
        1. Start with pure Gaussian noise
        2. Iteratively denoise for num_sample_steps steps
        3. Each step moves from t to t-1, gradually reducing noise
        4. Finally, clamp and unnormalize to get valid images

        The timesteps follow a linear schedule from 1.0 (pure noise) to 0.0 (clean).

        Args:
            shape: Desired output shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Generated images in [0, 1] range, shape matching input

        Process:
            t=1.0 (pure noise) -> t=0.8 -> t=0.6 -> ... -> t=0.0 (clean image)
        """
        batch = shape[0]

        # Start from pure Gaussian noise (t=1.0)
        img = torch.randn(shape, device = self.device)

        # Create linear timestep schedule from 1.0 to 0.0
        # +1 because we need num_sample_steps intervals (num_sample_steps + 1 points)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        # Iteratively denoise
        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]          # Current timestep
            times_next = steps[i + 1] # Next (less noisy) timestep
            img = self.p_sample(img, times, times_next)

        # Final cleanup: clamp to valid range and unnormalize
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        """
        Generate a batch of images from random noise.

        This is the main user-facing sampling function. It generates images by
        running the full reverse diffusion process starting from Gaussian noise.

        Args:
            batch_size: Number of images to generate (default: 16)

        Returns:
            torch.Tensor: Generated images in [0, 1] range,
                         shape (batch_size, channels, image_size, image_size)

        Example:
            >>> diffusion = VParamContinuousTimeGaussianDiffusion(...)
            >>> generated_images = diffusion.sample(batch_size=4)
            >>> # generated_images has shape (4, 3, 64, 64) for 64x64 RGB images
        """
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # training related functions - noise prediction

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, times, noise = None):
        """
        Sample from the forward diffusion process q(x_t | x_0).

        This implements the forward process that adds noise to clean images.
        Given a clean image x_0 and a timestep t, it produces a noisy version x_t.

        The forward process is defined as:
            x_t = α_t * x_0 + σ_t * ε

        where:
            - α_t = sqrt(sigmoid(log_snr_t)) controls the signal strength
            - σ_t = sqrt(sigmoid(-log_snr_t)) controls the noise strength
            - ε ~ N(0, I) is Gaussian noise
            - α_t² + σ_t² = 1 (variance preserving)

        Args:
            x_start: Clean images, shape (batch, channels, height, width)
            times: Timesteps in [0, 1], shape (batch,) or scalar
            noise: Optional pre-sampled noise. If None, samples fresh noise.

        Returns:
            tuple: (x_noised, log_snr, alpha, sigma)
                - x_noised: Noised images at timestep t
                - log_snr: Log signal-to-noise ratio at timestep t
                - alpha: Signal coefficient α_t
                - sigma: Noise coefficient σ_t

        Note:
            Uses @autocast(enabled=False) to prevent mixed precision issues
            with the noise sampling process.
        """
        # Use provided noise or sample fresh noise
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Get log-SNR for the given timesteps
        log_snr = self.log_snr(times)

        # Pad log_snr dimensions to match x_start for broadcasting
        # e.g., [B] -> [B, 1, 1, 1] for images
        log_snr_padded = right_pad_dims_to(x_start, log_snr)

        # Compute alpha and sigma from log-SNR
        # sigmoid(log_snr) = α², sigmoid(-log_snr) = σ²
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())

        # Apply forward diffusion: x_t = α_t * x_0 + σ_t * ε
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr, alpha, sigma

    def random_times(self, batch_size):
        """
        Sample random timesteps uniformly from [0, 1] for training.

        In continuous time diffusion, we use continuous timesteps rather than
        discrete steps. During training, we randomly sample timesteps to ensure
        the model learns to denoise at all noise levels.

        Args:
            batch_size: Number of random timesteps to generate

        Returns:
            torch.Tensor: Random timesteps in [0, 1], shape (batch_size,)
                         0.0 = clean image, 1.0 = pure noise
        """
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise = None):
        """
        Compute the v-parameterization training loss.

        This is the core training objective for v-parameterization diffusion models.
        The model learns to predict v = α_t * ε - σ_t * x_0, which is a linear
        combination of the noise and the clean image.

        Training Process:
        -----------------
        1. Add noise to clean images: x_t = α_t * x_0 + σ_t * ε
        2. Compute ground truth v: v = α_t * ε - σ_t * x_0
        3. Model predicts v from x_t and log_snr
        4. Loss = MSE(predicted_v, true_v)

        Why V-Parameterization?
        -----------------------
        - Balances prediction difficulty across noise levels
        - More stable than predicting ε alone (unstable at low noise)
        - More stable than predicting x_0 alone (unstable at high noise)
        - Critical for progressive distillation (Salimans & Ho, 2022)

        The v-prediction objective is described in Section 4 of the Progressive
        Distillation paper, with mathematical derivation in Appendix D.

        Args:
            x_start: Clean images in [-1, 1] range, shape (batch, channels, height, width)
            times: Timesteps in [0, 1], shape (batch,)
            noise: Optional pre-sampled noise. If None, samples fresh noise.

        Returns:
            torch.Tensor: Mean squared error loss (scalar)

        Reference:
            Progressive Distillation paper (Section 4, Appendix D):
            https://arxiv.org/abs/2202.00512
        """
        # Use provided noise or sample fresh noise
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Forward diffusion: add noise to get x_t and coefficients
        x, log_snr, alpha, sigma = self.q_sample(x_start = x_start, times = times, noise = noise)

        # Compute ground truth v-target
        # v = α_t * ε - σ_t * x_0
        # (described in section 4 as the prediction objective, with derivation in Appendix D)
        v = alpha * noise - sigma * x_start

        # Get model prediction
        model_out = self.model(x, log_snr)

        # Compute MSE loss between predicted and true v
        return F.mse_loss(model_out, v)

    def forward(self, img, *args, **kwargs):
        """
        Forward pass for training the diffusion model.

        This is called during training (e.g., loss = model(images)). It:
        1. Validates image dimensions
        2. Normalizes images to [-1, 1]
        3. Samples random timesteps
        4. Computes v-parameterization loss

        Args:
            img: Batch of images in [0, 1] range, shape (batch, channels, height, width)
            *args: Additional arguments passed to p_losses
            **kwargs: Additional keyword arguments passed to p_losses

        Returns:
            torch.Tensor: Training loss (scalar)

        Raises:
            AssertionError: If image dimensions don't match expected image_size

        Example:
            >>> diffusion = VParamContinuousTimeGaussianDiffusion(...)
            >>> images = torch.rand(32, 3, 64, 64)  # 32 images, 64x64 RGB
            >>> loss = diffusion(images)
            >>> loss.backward()
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # Sample random timesteps for each image in the batch
        times = self.random_times(b)

        # Normalize images from [0, 1] to [-1, 1]
        img = normalize_to_neg_one_to_one(img)

        # Compute and return training loss
        return self.p_losses(img, times, *args, **kwargs)
