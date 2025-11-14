"""
Denoising Diffusion Probabilistic Models (DDPM) PyTorch Implementation

This package provides a comprehensive collection of diffusion model implementations,
including various architectures and training strategies for generative modeling.

The package includes:
- Standard 2D diffusion models with Unet architecture
- 1D diffusion models for sequential data
- 3D diffusion models for volumetric data
- Advanced diffusion variants (learned variance, continuous time, weighted objectives)
- Karras et al. UNet architectures with improved noise scheduling
"""

# Standard 2D Gaussian diffusion model with Unet architecture and training utilities
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer

# Advanced diffusion model variants with different parameterizations and objectives
from denoising_diffusion_pytorch.learned_gaussian_diffusion import LearnedGaussianDiffusion
from denoising_diffusion_pytorch.continuous_time_gaussian_diffusion import ContinuousTimeGaussianDiffusion
from denoising_diffusion_pytorch.weighted_objective_gaussian_diffusion import WeightedObjectiveGaussianDiffusion
from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion
from denoising_diffusion_pytorch.v_param_continuous_time_gaussian_diffusion import VParamContinuousTimeGaussianDiffusion

# 1D diffusion models for sequential data (e.g., time series, audio)
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Unet1D, Trainer1D, Dataset1D

# Karras et al. UNet architecture with inverse square root decay learning rate scheduler
from denoising_diffusion_pytorch.karras_unet import (
    KarrasUnet,
    InvSqrtDecayLRSched
)

# Karras UNet variants for different data dimensions
from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D
from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D
