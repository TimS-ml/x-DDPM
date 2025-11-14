"""
Setup script for the denoising-diffusion-pytorch package.

This script configures the package installation, including dependencies,
metadata, and PyPI classifiers.
"""

from setuptools import setup, find_packages

# Import version from the version module
exec(open('denoising_diffusion_pytorch/version.py').read())

setup(
  name = 'denoising-diffusion-pytorch',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/denoising-diffusion-pytorch',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  # Required dependencies for the package
  install_requires=[
    'accelerate',              # For distributed training and mixed precision
    'einops',                  # Tensor operations with readable syntax
    'ema-pytorch>=0.4.2',     # Exponential moving average for model parameters
    'numpy',                   # Numerical computing
    'pillow',                  # Image processing
    'pytorch-fid',            # Frechet Inception Distance for evaluating generative models
    'scipy',                   # Scientific computing utilities
    'torch>=2.0',             # PyTorch deep learning framework (>=2.0 for Flash Attention)
    'torchvision',            # Computer vision utilities and datasets
    'tqdm'                     # Progress bars for training loops
  ],
  # PyPI classifiers for package categorization
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
