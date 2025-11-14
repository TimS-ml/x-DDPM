"""
Elucidated Diffusion Models

Implementation of the improved diffusion model framework from the paper:
"Elucidating the Design Space of Diffusion-Based Generative Models" by Karras et al. (2022)
Paper: https://arxiv.org/abs/2206.00364

Key Innovations:
-----------------
1. Preconditioning Network Architecture:
   - Separates the denoising network into skip connection (c_skip), output scaling (c_out),
     input scaling (c_in), and noise conditioning (c_noise) components
   - Improves training stability and sample quality by properly scaling inputs/outputs
   - Based on optimal Bayesian denoiser derivation (Equation 7 in paper)

2. Improved Noise Schedule:
   - Uses a power function schedule with rho parameter (typically 7) instead of linear/cosine
   - Better distribution of sampling steps across noise levels (Equation 5 in paper)
   - More time steps at critical intermediate noise levels

3. Optimal Loss Weighting:
   - Loss weights derived from signal-to-noise ratio considerations
   - Prevents over/under-emphasizing certain noise levels during training
   - Leads to more balanced learning across the entire noise range

4. Stochastic Sampling (Churn):
   - Adds controlled noise (gamma parameter) during sampling for stochasticity
   - Can transition between deterministic ODE sampling and stochastic SDE sampling
   - Improves sample diversity while maintaining quality

5. Second-Order ODE Solver:
   - Uses Heun's method (predictor-corrector) for more accurate ODE integration
   - Reduces discretization error compared to first-order Euler method
   - Achieves better quality with fewer sampling steps

6. Flexible Sampling Methods:
   - Supports standard second-order sampler with stochasticity
   - Compatible with advanced samplers like DPM++ (Diffusion Probabilistic Models Solver++)
   - Allows trading off between speed and quality

Training Procedure:
-------------------
- Noise levels sampled from log-normal distribution (P_mean, P_std)
- Network trained to predict denoised images (not noise)
- Self-conditioning: optionally condition on previous denoising prediction
"""

from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce

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
        val if it exists, otherwise d() if d is callable, else d
    """
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

def log(t, eps = 1e-20):
    """
    Safe logarithm that clamps input to avoid log(0).

    Args:
        t: Input tensor
        eps: Minimum value to clamp to (default: 1e-20)

    Returns:
        torch.Tensor: log(max(t, eps))
    """
    return torch.log(t.clamp(min = eps))

# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalize image from [0, 1] range to [-1, 1] range.

    Args:
        img: Image tensor in [0, 1] range

    Returns:
        torch.Tensor: Image scaled to [-1, 1] range
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Unnormalize tensor from [-1, 1] range back to [0, 1] range.

    Args:
        t: Tensor in [-1, 1] range

    Returns:
        torch.Tensor: Tensor scaled to [0, 1] range
    """
    return (t + 1) * 0.5

# main class

class ElucidatedDiffusion(nn.Module):
    """
    Elucidated Diffusion Model implementing the framework from Karras et al. (2022).

    This class wraps a denoising network with improved preconditioning, noise scheduling,
    and sampling techniques. Unlike traditional DDPM/DDIM, this approach:
    - Uses variance-preserving (VP) formulation with sigma parameterization
    - Trains the network to predict denoised images directly (not noise)
    - Applies optimal input/output scaling (preconditioning) to the network
    - Uses second-order ODE solvers for higher quality samples

    The framework is based on analyzing diffusion models as ODEs/SDEs and deriving
    optimal design choices through first principles.
    """

    def __init__(
        self,
        net,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        """
        Initialize the Elucidated Diffusion model.

        Args:
            net: The underlying denoising U-Net network. Must have random_or_learned_sinusoidal_cond=True
                 for proper noise level conditioning.
            image_size: Size of square images (e.g., 64 for 64x64 images)
            channels: Number of image channels (3 for RGB, 1 for grayscale)
            num_sample_steps: Number of discretization steps for sampling (N in paper).
                             More steps = higher quality but slower. Typical: 32-256
            sigma_min: Minimum noise level (default: 0.002). Lower bound of noise schedule.
            sigma_max: Maximum noise level (default: 80). Upper bound of noise schedule.
                       Start sampling from pure noise at this level.
            sigma_data: Estimated standard deviation of the data distribution (default: 0.5).
                       Used in preconditioning formulas. For images normalized to [-1, 1], 0.5 is typical.
            rho: Controls the distribution of sampling steps (default: 7).
                 Higher rho = more steps at lower noise levels. See Equation 5 in paper.
            P_mean: Mean of log-normal distribution for sampling training noise levels (default: -1.2).
            P_std: Std of log-normal distribution for sampling training noise levels (default: 1.2).
                   Together with P_mean, determines which noise levels are emphasized during training.
            S_churn: Controls amount of stochasticity in sampling (default: 80).
                    Higher = more stochastic/diverse samples. Set to 0 for deterministic ODE sampling.
                    Dataset-dependent, see Table 5 in paper.
            S_tmin: Minimum sigma value where stochasticity is applied (default: 0.05)
            S_tmax: Maximum sigma value where stochasticity is applied (default: 50)
            S_noise: Noise scale multiplier for stochastic sampling (default: 1.003)
        """
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition

        self.net = net

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        """
        Get the device (CPU/GPU) where the model parameters are stored.

        Returns:
            torch.device: Device of the neural network
        """
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        """
        Compute the skip connection weight for preconditioning (Table 1 in paper).

        This determines how much of the noisy input to pass directly to the output.
        At high noise (sigma >> sigma_data), c_skip ≈ 0 (rely on network prediction).
        At low noise (sigma << sigma_data), c_skip ≈ 1 (keep most of input).

        Formula: c_skip(σ) = σ_data² / (σ² + σ_data²)

        Args:
            sigma: Noise level (can be float or tensor)

        Returns:
            float or torch.Tensor: Skip connection weight in [0, 1]
        """
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        """
        Compute the output scaling weight for preconditioning (Table 1 in paper).

        This scales the network's output prediction before adding to skip connection.
        Ensures the output has appropriate magnitude across different noise levels.

        Formula: c_out(σ) = σ · σ_data · (σ² + σ_data²)^(-1/2)

        Args:
            sigma: Noise level (can be float or tensor)

        Returns:
            float or torch.Tensor: Output scaling weight
        """
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        """
        Compute the input scaling weight for preconditioning (Table 1 in paper).

        This normalizes the noisy input to have roughly unit variance before
        feeding to the network, improving training stability.

        Formula: c_in(σ) = (σ² + σ_data²)^(-1/2)

        Args:
            sigma: Noise level (can be float or tensor)

        Returns:
            float or torch.Tensor: Input scaling weight
        """
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        """
        Compute the noise level conditioning for the network (Table 1 in paper).

        Transforms sigma into the format expected by the network's time embedding.
        Uses log-space and scaling factor of 0.25 for better conditioning.

        Formula: c_noise(σ) = (1/4) · log(σ)

        Args:
            sigma: Noise level (can be float or tensor)

        Returns:
            float or torch.Tensor: Transformed noise level for conditioning
        """
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma, self_cond = None, clamp = False):
        """
        Forward pass through the preconditioned denoising network (Equation 7 in paper).

        This is the core of the elucidated diffusion framework. Instead of directly using
        the network output, we apply optimal input/output scaling (preconditioning) that
        improves training and sampling:

        D(x; σ) = c_skip(σ) · x + c_out(σ) · F(c_in(σ) · x; c_noise(σ))

        where:
        - x is the noisy input image
        - F is the underlying neural network
        - c_in scales the input
        - c_noise conditions the network on noise level
        - c_out scales the network output
        - c_skip adds a skip connection from input

        This formulation ensures the denoiser has the right inductive bias across
        all noise levels, derived from optimal Bayesian denoising theory.

        Args:
            noised_images: Noisy images tensor of shape (B, C, H, W)
            sigma: Noise level, either a float or tensor of shape (B,)
            self_cond: Optional self-conditioning tensor from previous prediction
            clamp: Whether to clamp output to [-1, 1] range

        Returns:
            torch.Tensor: Denoised image prediction of shape (B, C, H, W)
        """
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            self_cond
        )

        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        """
        Generate the noise level schedule for sampling (Equation 5 in paper).

        Creates a sequence of noise levels from sigma_max down to 0. Uses a power
        function schedule controlled by rho parameter, which allocates more steps
        at lower noise levels where fine details are generated.

        Formula: σ_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ

        The schedule:
        - Starts at sigma_max (pure noise)
        - Ends at 0 (clean image)
        - Rho=7 (default) concentrates steps at intermediate noise levels
        - Higher rho → more steps at low noise (fine details)
        - Lower rho → more uniform distribution

        Args:
            num_sample_steps: Number of steps (default: self.num_sample_steps)

        Returns:
            torch.Tensor: Noise levels of shape (num_sample_steps + 1,) including final 0
        """
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, batch_size = 16, num_sample_steps = None, clamp = True):
        """
        Generate samples using stochastic second-order sampler (Algorithm 2 in paper).

        This is the main sampling method combining several innovations:

        1. Stochastic Sampling (Churn):
           - Adds controlled noise via gamma parameter before each denoising step
           - gamma > 0: Stochastic SDE sampling (more diverse)
           - gamma = 0: Deterministic ODE sampling (more consistent)
           - Applied only in the range [S_tmin, S_tmax]

        2. Second-Order Integration (Heun's Method):
           - First evaluates denoiser at current step (predictor)
           - Then takes tentative step to next noise level
           - Evaluates denoiser again (corrector)
           - Final step uses average of both gradients
           - More accurate than first-order Euler, fewer steps needed

        3. Self-Conditioning:
           - If enabled, feeds previous denoising prediction back to network
           - Helps network make more consistent predictions
           - Can improve sample quality

        The sampling process:
        - Start with pure noise at sigma_max
        - Gradually decrease noise level following schedule
        - At each step: add stochasticity → denoise → second-order correction
        - End at sigma=0 with clean sample

        Args:
            batch_size: Number of images to generate
            num_sample_steps: Number of denoising steps (default: self.num_sample_steps)
                             More steps = higher quality but slower
            clamp: Whether to clamp intermediate predictions to [-1, 1]

        Returns:
            torch.Tensor: Generated images of shape (batch_size, channels, H, W) in [0, 1] range
        """
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device = self.device)

        # for self conditioning

        x_start = None

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            # Add stochasticity: increase noise level from sigma to sigma_hat
            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            self_cond = x_start if self.self_condition else None

            # First-order step (predictor): evaluate denoiser at current noise level
            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                # Second-order step (corrector): evaluate at next noise level and average
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    @torch.no_grad()
    def sample_using_dpmpp(self, batch_size = 16, num_sample_steps = None):
        """
        Generate samples using DPM-Solver++ (advanced deterministic sampler).

        DPM-Solver++ is an improved ODE solver for diffusion models from:
        "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
        by Lu et al. (https://arxiv.org/abs/2211.01095)

        Key advantages over standard sampler:
        - Deterministic (no stochasticity, reproducible with same seed)
        - Higher-order solver with better accuracy
        - Can achieve similar quality with fewer steps
        - Uses exponential integrator formulation in log-space
        - Second-order method with linear multistep correction

        Implementation details:
        - Converts between sigma and log-SNR (t) parameterizations
        - Uses denoised prediction from current and previous steps
        - Applies Richardson extrapolation-like correction
        - More mathematically principled than heuristic samplers

        Credit: Katherine Crowson (https://github.com/crowsonkb) for implementation

        Args:
            batch_size: Number of images to generate
            num_sample_steps: Number of denoising steps (default: self.num_sample_steps)

        Returns:
            torch.Tensor: Generated images of shape (batch_size, channels, H, W) in [0, 1] range
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        images  = sigmas[0] * torch.randn(shape, device = device)

        # Convert between sigma and log-SNR (t) parameterizations
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            # Get denoised prediction at current noise level
            denoised = self.preconditioned_network_forward(images, sigmas[i].item())
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            # Use first-order method for first step or last step
            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                # Second-order correction using previous denoised prediction
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            # Exponential integrator step in log-space
            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    # training

    def loss_weight(self, sigma):
        """
        Compute the loss weighting for a given noise level (Equation 16 in paper).

        The loss weight ensures balanced training across all noise levels by accounting
        for the signal-to-noise ratio. Without proper weighting, the model might focus
        too much on certain noise levels and neglect others.

        Formula: λ(σ) = (σ² + σ_data²) / (σ · σ_data)²

        Intuition:
        - At high noise (large σ): weight is lower (denoising is easier, less important)
        - At low noise (small σ): weight is higher (denoising fine details is harder, more important)
        - Near σ ≈ σ_data: weight is roughly constant
        - Derived from optimal weighting in denoising score matching

        Args:
            sigma: Noise level(s), tensor of shape (B,)

        Returns:
            torch.Tensor: Loss weights of shape (B,)
        """
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        """
        Sample noise levels from log-normal distribution for training.

        Training noise levels are sampled from: σ ~ exp(N(P_mean, P_std²))

        This distribution:
        - Covers a wide range of noise scales (from ~0.001 to ~100)
        - Log-normal ensures σ > 0 always
        - P_mean = -1.2, P_std = 1.2 works well empirically (Table 1 in paper)
        - Concentrates samples around exp(P_mean) ≈ 0.3

        Different from uniform or fixed schedules in DDPM - here we randomly
        sample noise levels each training step for better coverage.

        Args:
            batch_size: Number of noise levels to sample

        Returns:
            torch.Tensor: Noise levels of shape (batch_size,)
        """
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, images):
        """
        Training forward pass - compute denoising loss.

        Training procedure:
        1. Normalize images to [-1, 1]
        2. Sample random noise levels from log-normal distribution
        3. Add Gaussian noise: noisy_image = image + σ * noise
        4. (Optional) Self-conditioning: predict once, feed back to network
        5. Predict denoised image from noisy image
        6. Compute MSE loss between prediction and original image
        7. Weight loss by σ-dependent factor for balanced training
        8. Return mean loss

        Key differences from DDPM:
        - No fixed timestep schedule (sample σ randomly)
        - Predict denoised image directly (not noise)
        - Use variance-preserving formulation (alpha = 1)
        - Apply learned preconditioning to network
        - Optimal loss weighting based on σ

        Args:
            images: Batch of training images in [0, 1] range, shape (B, C, H, W)

        Returns:
            torch.Tensor: Scalar loss value for backpropagation
        """
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'

        images = normalize_to_neg_one_to_one(images)

        # Sample random noise levels for this batch
        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = torch.randn_like(images)

        # Add noise: x_noisy = x + σ * ε, where ε ~ N(0, I)
        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        self_cond = None

        # Self-conditioning: 50% of the time, condition on previous prediction
        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_images, sigmas)
                self_cond.detach_()

        # Predict denoised image
        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)

        # Compute MSE loss between predicted and true denoised images
        losses = F.mse_loss(denoised, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # Apply noise-level-dependent loss weighting
        losses = losses * self.loss_weight(sigmas)

        return losses.mean()
