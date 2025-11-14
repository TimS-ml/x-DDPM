"""
Weighted Objective Gaussian Diffusion

This module implements a variant of Gaussian Diffusion with weighted objectives for improved
training dynamics and generation quality.

Weighted Objectives in Diffusion Models
----------------------------------------
Traditional diffusion models typically predict either the noise or the original image (x_start)
from a noisy image. This implementation takes a hybrid approach where the model learns to:

1. Predict both the noise and the original image (x_start) simultaneously
2. Learn time-dependent weights to combine these predictions optimally
3. Train with multiple loss terms to balance different objectives

Why Weighted Objectives Are Useful
-----------------------------------
Weighted objectives provide several advantages:

- **Flexibility**: The model can adaptively choose the best prediction strategy at each timestep
- **Stability**: By combining multiple prediction modes, training becomes more robust
- **Better Convergence**: Different weighting schemes can emphasize different aspects of generation
- **Adaptive Learning**: The model learns which prediction type works best at different noise levels

Weighting Schemes
-----------------
This implementation uses three types of weighting:

1. **Prediction Loss Weights**: Fixed weights (set at initialization) that control how much
   the model should care about predicting noise vs. predicting x_start directly.
   - pred_noise_loss_weight: Penalizes errors in noise prediction
   - pred_x_start_loss_weight: Penalizes errors in direct x_start prediction

2. **Learned Dynamic Weights**: The model outputs softmax-normalized weights that determine
   how to combine x_start derived from noise prediction with direct x_start prediction.
   These weights are learned and can vary by timestep and spatial location.

3. **Combined Loss**: The final loss is a weighted sum of:
   - Noise prediction loss (scaled by pred_noise_loss_weight)
   - Direct x_start prediction loss (scaled by pred_x_start_loss_weight)
   - Weighted x_start combination loss (primary objective)

Effects on Training
-------------------
- Lower pred_noise_loss_weight: Model focuses less on accurate noise prediction
- Lower pred_x_start_loss_weight: Model focuses less on direct x_start prediction
- The weighted combination loss ensures the model learns useful weights for combining predictions
- The softmax-normalized weights allow smooth interpolation between prediction modes
"""

import torch
from inspect import isfunction
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion

# helper functions

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
        d: Default value or function that returns default value

    Returns:
        val if val exists, otherwise d() if d is a function, else d
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

# some improvisation on my end
# where i have the model learn to both predict noise and x0
# and learn the weighted sum for each depending on time step

class WeightedObjectiveGaussianDiffusion(GaussianDiffusion):
    """
    Gaussian Diffusion with Weighted Objectives.

    This class extends the standard GaussianDiffusion to implement a dual-prediction approach
    where the model learns to predict both noise and the original image (x_start), along with
    learned weights to optimally combine these predictions.

    Key Differences from Standard Diffusion
    ----------------------------------------
    1. **Multi-Head Output**: The model outputs (channels * 2 + 2) values:
       - channels for predicted noise
       - channels for predicted x_start
       - 2 weight values (softmax normalized) for combining predictions

    2. **Hybrid Loss Function**: Trains with three loss components:
       - Noise prediction loss
       - Direct x_start prediction loss
       - Weighted combination loss (primary objective)

    3. **Adaptive Weighting**: The model learns to weight predictions differently at different
       timesteps and spatial locations, allowing it to choose the best strategy dynamically.

    Weighting Strategy
    ------------------
    The model learns 2 weights (w1, w2) that are softmax-normalized. These weights determine:
    - w1: How much to trust x_start derived from noise prediction
    - w2: How much to trust direct x_start prediction

    The final prediction is: x_start_final = w1 * x_start_from_noise + w2 * x_start_direct

    Loss Weighting Effects
    -----------------------
    - **pred_noise_loss_weight** (default: 0.1): Controls auxiliary loss for noise prediction.
      Lower values make the model care less about accurate noise prediction as a standalone task.

    - **pred_x_start_loss_weight** (default: 0.1): Controls auxiliary loss for direct x_start.
      Lower values make the model care less about direct x_start prediction as a standalone task.

    - The weighted combination loss (unweighted, primary objective) ensures the model learns
      to effectively combine both prediction modes for the best final result.

    Attributes:
        split_dims (tuple): Dimensions for splitting model output into (noise, x_start, weights)
        pred_noise_loss_weight (float): Weight for noise prediction loss term
        pred_x_start_loss_weight (float): Weight for x_start prediction loss term
    """

    def __init__(
        self,
        model,
        *args,
        pred_noise_loss_weight = 0.1,
        pred_x_start_loss_weight = 0.1,
        **kwargs
    ):
        """
        Initialize WeightedObjectiveGaussianDiffusion.

        Args:
            model: UNet model for diffusion. Must have out_dim = (channels * 2 + 2) to output
                   noise predictions, x_start predictions, and 2 weights for combining them.
            *args: Additional positional arguments passed to GaussianDiffusion parent class
            pred_noise_loss_weight (float, optional): Weight for the noise prediction loss term.
                Lower values (e.g., 0.1) treat noise prediction as an auxiliary task. Higher
                values (e.g., 1.0) give it equal importance to the main objective. Default: 0.1
            pred_x_start_loss_weight (float, optional): Weight for the direct x_start prediction
                loss term. Lower values treat it as auxiliary. Default: 0.1
            **kwargs: Additional keyword arguments passed to GaussianDiffusion parent class

        Raises:
            AssertionError: If model.out_dim != (channels * 2 + 2)
            AssertionError: If model.self_condition is enabled (not yet supported)
            AssertionError: If DDIM sampling is enabled (not compatible with this approach)

        Note:
            For a 3-channel RGB image, the model should output 8 channels:
            - 3 for predicted noise
            - 3 for predicted x_start
            - 2 for combination weights
        """
        super().__init__(model, *args, **kwargs)
        channels = model.channels

        # Ensure model outputs the correct number of channels for dual prediction + weights
        assert model.out_dim == (channels * 2 + 2), 'dimension out (out_dim) of unet must be twice the number of channels + 2 (for the softmax weighted sum) - for channels of 3, this should be (3 * 2) + 2 = 8'

        # Self-conditioning and DDIM are not yet compatible with weighted objectives
        assert not model.self_condition, 'not supported yet'
        assert not self.is_ddim_sampling, 'ddim sampling cannot be used'

        # Define how to split model output: (noise, x_start, weights)
        self.split_dims = (channels, channels, 2)

        # Store loss weights for the auxiliary prediction tasks
        self.pred_noise_loss_weight = pred_noise_loss_weight
        self.pred_x_start_loss_weight = pred_x_start_loss_weight

    def p_mean_variance(self, *, x, t, clip_denoised, model_output = None):
        """
        Compute the mean and variance of the posterior distribution p(x_{t-1} | x_t) for sampling.

        This method is used during the reverse diffusion process (sampling/generation). It computes
        the distribution from which we sample x_{t-1} given x_t.

        Weighted Prediction Process
        ----------------------------
        1. Model outputs: noise prediction, x_start prediction, and 2 weight values
        2. Weights are softmax-normalized to ensure they sum to 1
        3. Derive x_start from predicted noise using the reverse diffusion equation
        4. Compute weighted combination: final_x_start = w1 * x_start_from_noise + w2 * x_start_direct
        5. Use the weighted x_start to compute the posterior distribution parameters

        This approach allows the model to dynamically choose the best prediction strategy at each
        timestep and spatial location. For example:
        - At high noise levels (early timesteps), noise prediction might be more reliable
        - At low noise levels (late timesteps), direct x_start prediction might work better
        - The learned weights adapt to these conditions

        Args:
            x (Tensor): Current noisy image at timestep t, shape (batch, channels, height, width)
            t (Tensor): Current timestep, shape (batch,)
            clip_denoised (bool): Whether to clip the predicted x_start to [-1, 1] range
            model_output (Tensor, optional): Pre-computed model output (not used, model is called
                directly). This parameter exists for API compatibility.

        Returns:
            tuple: (model_mean, model_variance, model_log_variance)
                - model_mean (Tensor): Mean of p(x_{t-1} | x_t), shape (batch, channels, height, width)
                - model_variance (Tensor): Variance of p(x_{t-1} | x_t), shape matching input
                - model_log_variance (Tensor): Log variance for numerical stability

        Note:
            The weighted combination happens in the "height, width" spatial dimensions, allowing
            the model to use different weighting strategies for different regions of the image.
        """
        # Get model predictions: noise, x_start, and weights for combining them
        model_output = self.model(x, t)

        # Split the output into its three components
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim = 1)

        # Normalize weights using softmax so they sum to 1 (proper probability distribution)
        normalized_weights = weights.softmax(dim = 1)

        # Derive x_start from the predicted noise using the diffusion process equations
        x_start_from_noise = self.predict_start_from_noise(x, t = t, noise = pred_noise)

        # Stack the two x_start predictions along a new dimension for weighted combination
        # Shape: (batch, 2, channels, height, width)
        x_starts = torch.stack((x_start_from_noise, pred_x_start), dim = 1)

        # Compute weighted sum using learned weights (via Einstein summation)
        # 'b j h w, b j c h w -> b c h w' means:
        #   b=batch, j=2 weights, c=channels, h=height, w=width
        # This applies potentially different weights at each spatial location
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', normalized_weights, x_starts)

        # Optionally clip the predicted x_start to valid range [-1, 1]
        if clip_denoised:
            weighted_x_start.clamp_(-1., 1.)

        # Compute the posterior distribution q(x_{t-1} | x_t, x_0) using the weighted x_start
        model_mean, model_variance, model_log_variance = self.q_posterior(weighted_x_start, x, t)

        return model_mean, model_variance, model_log_variance

    def p_losses(self, x_start, t, noise = None, clip_denoised = False):
        """
        Compute the training loss with weighted objectives.

        This method implements a multi-objective loss function that trains the model to:
        1. Predict noise accurately (auxiliary objective)
        2. Predict x_start directly (auxiliary objective)
        3. Learn optimal weights to combine these predictions (primary objective)

        Loss Components and Weighting Effects
        --------------------------------------
        The total loss is a sum of three terms:

        1. **Noise Prediction Loss** (weighted by pred_noise_loss_weight, default 0.1):
           - MSE between true noise and predicted noise
           - Lower weight (e.g., 0.1) treats this as an auxiliary task
           - Higher weight (e.g., 1.0) emphasizes learning good noise predictions
           - Effect: Helps the model learn noise patterns, which aids in deriving x_start

        2. **Direct x_start Prediction Loss** (weighted by pred_x_start_loss_weight, default 0.1):
           - MSE between true x_start and predicted x_start
           - Lower weight treats this as an auxiliary task
           - Higher weight emphasizes direct image reconstruction
           - Effect: Helps the model learn to directly predict the clean image

        3. **Weighted Combination Loss** (unweighted, weight = 1.0, primary objective):
           - MSE between true x_start and the weighted combination of predictions
           - This is the main training signal
           - Effect: Teaches the model to learn optimal weights for combining predictions
           - The model learns when to rely on noise prediction vs. direct prediction

        Weighting Scheme Rationale
        ---------------------------
        The default weights (0.1, 0.1, 1.0) reflect a design choice:
        - Primary goal: Learn to optimally combine predictions (weight = 1.0)
        - Secondary goals: Maintain reasonable noise and x_start predictions (weight = 0.1)
        - This allows the model to focus on the combination while ensuring both prediction
          modes remain useful

        Training Dynamics
        ------------------
        - Early training: Model learns basic noise and image prediction
        - Mid training: Model starts learning useful combination weights
        - Late training: Weights become finely tuned to different timesteps and regions
        - The softmax ensures weights always sum to 1, preventing degenerate solutions

        Args:
            x_start (Tensor): Original clean image, shape (batch, channels, height, width)
            t (Tensor): Timestep for each batch element, shape (batch,)
            noise (Tensor, optional): Pre-sampled noise. If None, random noise is generated.
                Shape (batch, channels, height, width)
            clip_denoised (bool, optional): Whether to clip predictions (not used in loss computation
                but kept for API compatibility). Default: False

        Returns:
            Tensor: Total loss (scalar) = weighted_x_start_loss + x_start_loss + noise_loss

        Note:
            The x_start derived from noise is clamped to [-2, 2] to prevent extreme values
            during training, which improves stability. This is wider than the sampling range
            [-1, 1] to allow the model some flexibility during optimization.
        """
        # Generate random noise if not provided
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Forward diffusion: add noise to clean image to get x_t
        x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

        # Get model predictions: noise, x_start, and combination weights
        model_output = self.model(x_t, t)
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim = 1)

        # Auxiliary Loss 1: Noise prediction loss
        # Penalizes inaccurate noise prediction, scaled by pred_noise_loss_weight
        # Lower weight (default 0.1) makes this a weak auxiliary objective
        noise_loss = F.mse_loss(noise, pred_noise) * self.pred_noise_loss_weight

        # Auxiliary Loss 2: Direct x_start prediction loss
        # Penalizes inaccurate direct image prediction, scaled by pred_x_start_loss_weight
        # Lower weight (default 0.1) makes this a weak auxiliary objective
        x_start_loss = F.mse_loss(x_start, pred_x_start) * self.pred_x_start_loss_weight

        # Derive x_start from predicted noise using the reverse diffusion equation
        x_start_from_pred_noise = self.predict_start_from_noise(x_t, t, pred_noise)

        # Clamp to prevent extreme values during training (stability measure)
        # Range [-2, 2] is wider than sampling range [-1, 1] to allow optimization flexibility
        x_start_from_pred_noise = x_start_from_pred_noise.clamp(-2., 2.)

        # Compute weighted combination of the two x_start predictions
        # Weights are softmax-normalized, so they sum to 1 (proper convex combination)
        # Einstein summation combines: w1 * x_start_from_noise + w2 * x_start_direct
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', weights.softmax(dim = 1), torch.stack((x_start_from_pred_noise, pred_x_start), dim = 1))

        # Primary Loss: Weighted combination loss (unweighted, implicitly weight = 1.0)
        # This is the main training objective - teaches the model to learn optimal weights
        weighted_x_start_loss = F.mse_loss(x_start, weighted_x_start)

        # Total loss: sum of all three components
        # Default: weighted_x_start_loss + 0.1 * x_start_loss + 0.1 * noise_loss
        return weighted_x_start_loss + x_start_loss + noise_loss
