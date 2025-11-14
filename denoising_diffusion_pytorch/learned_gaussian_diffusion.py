"""
Learned Variance Gaussian Diffusion Models

This module implements learned variance diffusion models based on the paper:
"Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
https://arxiv.org/abs/2102.09672

Key Concepts:
-------------

1. LEARNED VARIANCE vs FIXED VARIANCE:
   - Fixed Variance: Original DDPM uses predetermined variance schedules (either β_t or
     β̃_t = (1-ᾱ_t-1)/(1-ᾱ_t) * β_t) for the reverse diffusion process p(x_t-1|x_t).
   - Learned Variance: This implementation learns to interpolate between these two bounds,
     allowing the model to adaptively choose the optimal variance at each timestep.

2. VARIANCE INTERPOLATION:
   The model predicts an interpolation parameter v ∈ ℝ which is converted to [0,1] range
   to interpolate between minimum and maximum variance bounds:

   log σ²_t = v * log β_t + (1-v) * log β̃_t

   where:
   - β_t is the forward process variance (upper bound)
   - β̃_t is the posterior variance (lower bound)
   - v is the learned interpolation fraction

3. HYBRID LOSS FUNCTION:
   The total loss combines:
   - Simple Loss: MSE on noise prediction (like standard DDPM)
   - VB Loss: Variational lower bound term for learning variance

   L = L_simple + λ * L_VB

   where λ (vb_loss_weight) is typically 0.001 as per the paper.

4. VB LOSS COMPONENTS:
   - For t > 0: KL divergence KL(q(x_t-1|x_t,x_0) || p_θ(x_t-1|x_t))
   - For t = 0: Negative log-likelihood of a discretized Gaussian (decoder NLL)

5. MODEL OUTPUT:
   The U-Net outputs twice the number of channels:
   - First half: noise prediction (or x_0 prediction depending on objective)
   - Second half: variance interpolation parameter v

References:
-----------
- Improved DDPM: https://arxiv.org/abs/2102.09672
- Original DDPM: https://arxiv.org/abs/2006.11239
- Used in GLIDE and DALL-E 2 cascade models
"""

import torch
from collections import namedtuple
from math import pi, sqrt, log as ln
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, extract, unnormalize_to_zero_to_one

# constants

# NAT converts from nats to bits for measuring information-theoretic quantities
# (1 nat = 1/ln(2) bits ≈ 1.44 bits)
NAT = 1. / ln(2)

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_variance'])
"""Named tuple for storing model predictions.

Fields:
    pred_noise: Predicted noise ε_θ(x_t, t)
    pred_x_start: Predicted clean image x_0
    pred_variance: Predicted variance interpolation parameter
"""

# helper functions

def exists(x):
    """Check if a value is not None.

    Args:
        x: Value to check

    Returns:
        bool: True if x is not None, False otherwise
    """
    return x is not None

def default(val, d):
    """Return val if it exists, otherwise return d (or d() if d is a function).

    Args:
        val: Primary value to return
        d: Default value or function returning default value

    Returns:
        val if it exists, otherwise d or d()
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

# tensor helpers

def log(t, eps = 1e-15):
    """Numerically stable logarithm that clamps input to avoid log(0).

    Args:
        t: Input tensor
        eps: Minimum value to clamp to (default: 1e-15)

    Returns:
        torch.Tensor: Logarithm of clamped input
    """
    return torch.log(t.clamp(min = eps))

def meanflat(x):
    """Compute mean over all dimensions except the batch dimension.

    This is used to reduce spatial dimensions while preserving the batch structure,
    which is important for computing per-sample losses.

    Args:
        x: Input tensor of shape (batch, ...)

    Returns:
        torch.Tensor: Mean over all non-batch dimensions, shape (batch,)
    """
    return x.mean(dim = tuple(range(1, len(x.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    """Compute KL divergence between two Gaussian distributions.

    Calculates KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2))).

    This is used in the VB loss to measure the difference between:
    - The true posterior q(x_t-1|x_t, x_0) (mean1, logvar1)
    - The learned reverse process p_θ(x_t-1|x_t) (mean2, logvar2)

    Formula:
        KL(p||q) = 0.5 * [-1 + log(σ²_q/σ²_p) + σ²_p/σ²_q + (μ_p - μ_q)²/σ²_q]

    Args:
        mean1: Mean of first distribution (μ_p)
        logvar1: Log variance of first distribution (log σ²_p)
        mean2: Mean of second distribution (μ_q)
        logvar2: Log variance of second distribution (log σ²_q)

    Returns:
        torch.Tensor: KL divergence (not reduced, same shape as inputs)
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    """Fast approximation of the standard normal cumulative distribution function.

    Uses a tanh-based approximation that is computationally efficient.
    This approximation is from Glow (https://arxiv.org/abs/1807.03039).

    Args:
        x: Input tensor

    Returns:
        torch.Tensor: Approximate CDF values Φ(x)
    """
    return 0.5 * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * (x ** 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres = 0.999):
    """Compute log-likelihood of data under a discretized Gaussian distribution.

    This is used for the decoder NLL at timestep t=0. Instead of treating pixels as
    continuous values, this accounts for the fact that images are discrete (quantized
    to 256 levels). Each pixel value represents a bin, and we integrate the Gaussian
    probability density over that bin.

    For pixel value x, the bin is [x - 1/255, x + 1/255], and we compute:
        log P(pixel = x) = log [Φ((x + 1/255 - μ)/σ) - Φ((x - 1/255 - μ)/σ)]

    Edge cases are handled specially:
    - For very negative values (x < -thres): use log Φ((x + 1/255 - μ)/σ)
    - For very positive values (x > thres): use log(1 - Φ((x - 1/255 - μ)/σ))

    This follows the approach from "Improved DDPM" and PixelCNN++.

    Args:
        x: Target data (typically images scaled to [-1, 1])
        means: Mean of the Gaussian distribution
        log_scales: Log standard deviation of the Gaussian distribution
        thres: Threshold for edge case handling (default: 0.999)

    Returns:
        torch.Tensor: Log probabilities for each element (not reduced)
    """
    assert x.shape == means.shape == log_scales.shape

    # Center the data around the predicted mean
    centered_x = x - means
    # Inverse standard deviation for normalization
    inv_stdv = torch.exp(-log_scales)
    # Upper edge of the bin: x + 1/255 (accounting for 8-bit quantization)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    # Lower edge of the bin: x - 1/255
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1. - cdf_min)
    # Probability mass in the bin
    cdf_delta = cdf_plus - cdf_min

    # Handle edge cases to avoid numerical issues
    log_probs = torch.where(x < -thres,
        log_cdf_plus,  # For very negative values, use CDF at upper edge
        torch.where(x > thres,
            log_one_minus_cdf_min,  # For very positive values, use 1 - CDF at lower edge
            log(cdf_delta)))  # Normal case: log of probability mass in bin

    return log_probs

# https://arxiv.org/abs/2102.09672

# Author's note: Results may be questionable if focusing only on FID metrics,
# but this implementation is included as it's used in GLIDE and DALL-E 2 cascade models.
# This is a Gaussian diffusion implementation with learned variance using the
# hybrid loss approach (simple epsilon loss + variational bound loss).

class LearnedGaussianDiffusion(GaussianDiffusion):
    """Gaussian Diffusion with Learned Variance.

    Extends the standard GaussianDiffusion to learn the variance of the reverse
    process instead of using a fixed schedule. This follows the "Improved DDPM" paper.

    Key differences from standard DDPM:
    1. Model outputs 2x channels: [noise_prediction, variance_interpolation]
    2. Variance is learned by interpolating between theoretical bounds
    3. Loss combines simple MSE loss with variational bound (VB) loss
    4. VB loss helps learn optimal variance at each timestep

    The learned variance typically improves log-likelihood metrics but may not
    always improve perceptual quality (FID). However, it's used in state-of-the-art
    models like GLIDE and DALL-E 2.

    Args:
        model: U-Net model that outputs 2x channels (must have out_dim = channels * 2)
        vb_loss_weight: Weight for the VB loss term (λ in the paper). Default 0.001
                       as recommended in "Improved DDPM". This balances the simple
                       loss and VB loss components.
        *args: Additional arguments passed to parent GaussianDiffusion
        **kwargs: Additional keyword arguments passed to parent GaussianDiffusion

    Raises:
        AssertionError: If model output dimension is not 2x the number of channels
        AssertionError: If model uses self-conditioning (not yet supported)
    """
    def __init__(
        self,
        model,
        vb_loss_weight = 0.001,  # lambda was 0.001 in the paper
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        # Ensure model outputs double channels for noise + variance predictions
        assert model.out_dim == (model.channels * 2), 'dimension out of unet must be twice the number of channels for learned variance - you can also set the `learned_variance` keyword argument on the Unet to be `True`'
        # Self-conditioning not yet implemented for learned variance
        assert not model.self_condition, 'not supported yet'

        # Weight for the variational bound loss (recommended: 0.001 from paper)
        self.vb_loss_weight = vb_loss_weight

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        """Extract predictions from the model output.

        The model outputs 2x channels which are split into:
        1. Main prediction (noise or x_0, depending on objective)
        2. Variance interpolation parameter

        Args:
            x: Noisy input tensor x_t at timestep t
            t: Current timestep
            x_self_cond: Self-conditioning input (not used, for compatibility)
            clip_x_start: Whether to clip predicted x_0 to [-1, 1] range
            rederive_pred_noise: Whether to rederive noise prediction (not used here)

        Returns:
            ModelPrediction: Named tuple containing:
                - pred_noise: Predicted noise ε_θ(x_t, t)
                - pred_x_start: Predicted clean image x_0
                - pred_variance: Predicted variance interpolation parameter v
        """
        # Get model output: [noise/x_0, variance] concatenated along channel dim
        model_output = self.model(x, t)
        # Split into main prediction and variance prediction
        model_output, pred_variance = model_output.chunk(2, dim = 1)

        # Set up clipping function based on clip_x_start flag
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        # Extract predictions based on the training objective
        if self.objective == 'pred_noise':
            # Model predicts noise ε, derive x_0 from it
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            # Model predicts x_0 directly, derive noise from it
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        # Apply optional clipping to x_0 prediction
        x_start = maybe_clip(x_start)

        return ModelPrediction(pred_noise, x_start, pred_variance)

    def p_mean_variance(self, *, x, t, clip_denoised, model_output = None, **kwargs):
        """Compute the mean and variance of the reverse process distribution p_θ(x_t-1|x_t).

        This method implements the key innovation of learned variance. Instead of using
        a fixed variance schedule, the model learns to interpolate between theoretical
        variance bounds.

        VARIANCE INTERPOLATION EXPLAINED:
        ---------------------------------
        The reverse process variance σ²_t can theoretically range between two bounds:

        1. Lower bound (β̃_t): Posterior variance from the forward process
           β̃_t = (1 - ᾱ_t-1) / (1 - ᾱ_t) * β_t
           This is the variance when we know both x_t and x_0

        2. Upper bound (β_t): Forward process variance
           This is the variance of the forward diffusion

        The model outputs v ∈ ℝ which is converted to [0,1] and used to interpolate:

        log σ²_t = v * log(β_t) + (1-v) * log(β̃_t)

        This allows the model to adaptively choose the variance at each timestep,
        which improves the variational lower bound (ELBO).

        Args:
            x: Current noisy sample x_t
            t: Current timestep
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            model_output: Pre-computed model output (optional, computed if not provided)
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (model_mean, model_variance, model_log_variance, x_start)
                - model_mean: Mean μ_θ(x_t, t) of p_θ(x_t-1|x_t)
                - model_variance: Variance σ²_θ(x_t, t)
                - model_log_variance: Log variance log(σ²_θ(x_t, t))
                - x_start: Predicted clean image x_0
        """
        # Get or compute model output
        model_output = default(model_output, lambda: self.model(x, t))
        # Split into noise prediction and variance interpolation parameter
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim = 1)

        # Get variance bounds for interpolation
        # min_log = log(β̃_t): posterior variance (lower bound)
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        # max_log = log(β_t): forward process variance (upper bound)
        max_log = extract(torch.log(self.betas), t, x.shape)
        # Convert model output to [0, 1] range for interpolation
        var_interp_frac = unnormalize_to_zero_to_one(var_interp_frac_unnormalized)

        # LEARNED VARIANCE INTERPOLATION:
        # Interpolate between min and max variance in log space
        # log(σ²) = v * log(β_t) + (1-v) * log(β̃_t)
        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        # Convert back to variance: σ² = exp(log(σ²))
        model_variance = model_log_variance.exp()

        # Predict x_0 from noise prediction
        x_start = self.predict_start_from_noise(x, t, pred_noise)

        # Optionally clip x_0 to valid range
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # Compute mean of reverse process using predicted x_0
        # Uses the posterior mean formula: μ_θ(x_t, t) = f(x_t, x_0, t)
        model_mean, _, _ = self.q_posterior(x_start, x, t)

        return model_mean, model_variance, model_log_variance, x_start

    def p_losses(self, x_start, t, noise = None, clip_denoised = False):
        """Compute the hybrid training loss combining simple loss and VB loss.

        This is the core training method that implements the hybrid loss from "Improved DDPM":

        HYBRID LOSS = L_simple + λ * L_VB

        Where:
        1. L_simple: Standard MSE loss on noise prediction (like original DDPM)
           L_simple = ||ε - ε_θ(x_t, t)||²

        2. L_VB: Variational bound loss for learning the variance
           - For t > 0: KL(q(x_t-1|x_t,x_0) || p_θ(x_t-1|x_t))
           - For t = 0: -log p_θ(x_0|x_1) using discretized Gaussian

        3. λ (vb_loss_weight): Balancing weight, typically 0.001

        WHY DETACHED MEAN IN VB LOSS?
        ------------------------------
        The KL loss uses a DETACHED model mean for stability. This means:
        - The mean is not updated by gradients from the VB loss
        - Only the variance parameters receive gradients from VB loss
        - The mean is trained solely by the simple loss

        This is a key design choice from the paper that prevents the VB loss from
        interfering with the noise/x_0 prediction training.

        KL LOSS COMPONENT EXPLAINED:
        ----------------------------
        For t > 0, we minimize KL divergence between:
        - q(x_t-1|x_t, x_0): True posterior (knows clean image x_0)
        - p_θ(x_t-1|x_t): Model's reverse process (doesn't know x_0)

        The KL divergence penalizes the model for choosing incorrect variance.
        The model learns to interpolate between variance bounds to minimize this.

        Args:
            x_start: Clean input images x_0
            t: Timesteps for each sample in the batch
            noise: Optional pre-generated noise (generated if not provided)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            torch.Tensor: Scalar loss value combining simple loss and VB loss
        """
        # Generate noise if not provided
        noise = default(noise, lambda: torch.randn_like(x_start))
        # Create noisy samples x_t using the forward diffusion process
        x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

        # ============================================================
        # MODEL OUTPUT: [noise_prediction, variance_interpolation]
        # ============================================================
        model_output = self.model(x_t, t)

        # ============================================================
        # VB LOSS: Learn optimal variance through KL divergence
        # ============================================================

        # Get TRUE posterior distribution q(x_t-1|x_t, x_0)
        # This is the "ground truth" distribution that knows both x_t and x_0
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start = x_start, x_t = x_t, t = t)

        # Get MODEL's predicted distribution p_θ(x_t-1|x_t) with learned variance
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x = x_t, t = t, clip_denoised = clip_denoised, model_output = model_output)

        # IMPORTANT: Detach model mean for stability
        # Only the variance receives gradients from VB loss
        # The mean is trained by the simple loss only
        detached_model_mean = model_mean.detach()

        # Compute KL divergence: KL(q(x_t-1|x_t,x_0) || p_θ(x_t-1|x_t))
        # This measures how well the model's learned variance matches the true posterior
        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT  # Reduce spatially and convert to bits

        # For t=0, use discretized Gaussian NLL instead of KL
        # This accounts for the discrete nature of pixel values
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means = detached_model_mean, log_scales = 0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT

        # Choose between decoder NLL (t=0) and KL divergence (t>0)
        # At t=0, we're generating the final image, so use likelihood
        # At t>0, we're denoising, so use KL between posteriors
        vb_losses = torch.where(t == 0, decoder_nll, kl)

        # ============================================================
        # SIMPLE LOSS: Standard noise prediction MSE
        # ============================================================

        # Extract noise prediction (ignore variance prediction for simple loss)
        pred_noise, _ = model_output.chunk(2, dim = 1)

        # Compute MSE between predicted and true noise
        # This is the same loss as original DDPM
        simple_losses = F.mse_loss(pred_noise, noise)

        # ============================================================
        # HYBRID LOSS: Combine simple loss and VB loss
        # ============================================================
        # L_total = L_simple + λ * L_VB
        # where λ = vb_loss_weight (typically 0.001)
        return simple_losses + vb_losses.mean() * self.vb_loss_weight
