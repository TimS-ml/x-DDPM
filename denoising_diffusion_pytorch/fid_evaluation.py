"""
FID (Fréchet Inception Distance) Evaluation Module

This module implements FID score computation for evaluating the quality of generative models,
particularly diffusion models. FID is a widely-used metric in generative modeling that measures
the similarity between distributions of real and generated images.

What is FID?
------------
The Fréchet Inception Distance (FID) measures the distance between two multivariate Gaussian
distributions fitted to feature representations of real and generated images. Lower FID scores
indicate better quality and diversity of generated samples.

How FID Works:
1. Extract features from real and generated images using a pre-trained Inception-v3 network
2. Model these feature distributions as multivariate Gaussians
3. Compute the Fréchet distance (also known as Wasserstein-2 distance) between the two distributions

Why Use FID?
------------
- Captures both quality and diversity of generated samples
- Robust to small changes in the generative model
- Correlates well with human judgment of image quality
- Standard benchmark in the generative modeling community
- Sensitive to mode dropping (when the model fails to generate diverse samples)

FID Formula:
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2(Σ_real × Σ_fake)^(1/2))

where μ and Σ are the mean and covariance of the Inception features for real and fake images.
"""

import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm


def num_to_groups(num, divisor):
    """
    Divide a number into groups of a specified size.

    This utility function splits a total number of items into batches of approximately
    equal size. It's used to determine batch sizes when processing a large number of
    samples that don't divide evenly by the batch size.

    Args:
        num (int): Total number of items to divide
        divisor (int): Desired size of each group (batch size)

    Returns:
        list[int]: List of group sizes. Most groups will be of size 'divisor',
                   with the last group potentially being smaller if there's a remainder.

    Example:
        >>> num_to_groups(10, 3)
        [3, 3, 3, 1]  # Three full batches of 3, plus 1 remaining item
    """
    groups = num // divisor  # Number of complete groups
    remainder = num % divisor  # Remaining items after forming complete groups
    arr = [divisor] * groups  # Create list of full-sized groups
    if remainder > 0:
        arr.append(remainder)  # Add the smaller final group if there are remaining items
    return arr


class FIDEvaluation:
    """
    FID (Fréchet Inception Distance) Score Evaluator.

    This class handles the computation of FID scores to evaluate the quality of
    generative models. It manages the extraction of Inception features from both
    real and generated images, caching of dataset statistics, and calculation of
    the final FID score.

    The FID score quantifies how similar the distribution of generated images is
    to the distribution of real images by comparing their feature representations
    in a pre-trained Inception-v3 network's feature space.

    Typical workflow:
    1. Initialize with a data loader (for real images) and a sampler (for generated images)
    2. Call fid_score() which will:
       - Load or compute statistics for the real dataset
       - Generate samples and compute their statistics
       - Calculate and return the FID score
    """

    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
    ):
        """
        Initialize the FID evaluation module.

        Args:
            batch_size (int): Number of samples to process in each batch
            dl (DataLoader): DataLoader providing real images from the dataset
            sampler (nn.Module): Generative model that can sample synthetic images
            channels (int, optional): Number of image channels (1 for grayscale, 3 for RGB).
                                     Defaults to 3. Grayscale images are converted to RGB
                                     for Inception processing.
            accelerator (Accelerator, optional): HuggingFace Accelerator for distributed training.
                                                Defaults to None.
            stats_dir (str, optional): Directory to cache precomputed dataset statistics.
                                      Defaults to "./results".
            device (str, optional): Device to run computations on ('cuda' or 'cpu').
                                   Defaults to "cuda".
            num_fid_samples (int, optional): Number of samples to use for FID calculation.
                                            More samples give more reliable scores but take longer.
                                            Defaults to 50000 (standard in literature).
            inception_block_idx (int, optional): Dimensionality of Inception features to extract.
                                                Defaults to 2048 (final pooling layer features).
        """
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl  # DataLoader for real images
        self.sampler = sampler  # Generative model for fake images
        self.stats_dir = stats_dir
        # Use accelerator's print function if available for distributed training
        self.print_fn = print if accelerator is None else accelerator.print

        # Validate and get the Inception block index
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]

        # Initialize Inception-v3 model for feature extraction
        # We only need specific intermediate layers, not the full classifier
        self.inception_v3 = InceptionV3([block_idx]).to(device)

        # Track whether dataset statistics have been computed/loaded
        self.dataset_stats_loaded = False

    def calculate_inception_features(self, samples):
        """
        Extract feature representations from images using Inception-v3.

        This method processes a batch of images through a pre-trained Inception-v3
        network to obtain high-level feature representations. These features capture
        semantic information about the images and are used to compute FID.

        The Inception-v3 model was trained on ImageNet and learns features that
        correspond to object parts, textures, and high-level visual concepts.

        Args:
            samples (torch.Tensor): Batch of images with shape (B, C, H, W)
                                   where B is batch size, C is channels, H and W are height/width

        Returns:
            torch.Tensor: Feature vectors of shape (B, D) where D is the feature dimensionality
                         (typically 2048 for the final pooling layer)
        """
        # Convert grayscale images to RGB by repeating the single channel 3 times
        # Inception-v3 expects 3-channel RGB input
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        # Set to evaluation mode to disable dropout and use running stats for batchnorm
        self.inception_v3.eval()

        # Forward pass through Inception-v3 to extract features
        # Returns a tuple, we take the first element
        features = self.inception_v3(samples)[0]

        # Ensure features are spatially pooled to a single vector per image
        # Some Inception blocks may return spatial feature maps instead of vectors
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))

        # Reshape from (B, D, 1, 1) to (B, D) by removing spatial dimensions
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        """
        Load cached dataset statistics or compute them from the real dataset.

        FID calculation requires statistics (mean and covariance) of Inception features
        from the real dataset. Since these statistics don't change, they can be
        precomputed once and cached to disk for future use, saving significant computation time.

        This method:
        1. Attempts to load pre-cached statistics from disk
        2. If not found, computes statistics by:
           - Processing real images through Inception-v3
           - Extracting features for the specified number of samples
           - Computing mean (μ) and covariance (Σ) of the feature distribution
           - Saving the statistics to disk for future runs

        The statistics are stored as:
        - m2 (μ_real): Mean vector of real image features
        - s2 (Σ_real): Covariance matrix of real image features

        Sets:
            self.m2 (np.ndarray): Mean of real dataset features, shape (D,)
            self.s2 (np.ndarray): Covariance of real dataset features, shape (D, D)
            self.dataset_stats_loaded (bool): Flag indicating stats are loaded
        """
        path = os.path.join(self.stats_dir, "dataset_stats")

        try:
            # Attempt to load precomputed statistics from disk
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            # File not found - need to compute statistics from scratch
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )

            # Process real images in batches to extract features
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(self.dl)
                except StopIteration:
                    # DataLoader exhausted before reaching n_samples
                    break

                # Move images to GPU/CPU and extract Inception features
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)

            # Concatenate all feature batches into a single array
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )

            # Compute statistics: mean and covariance of the feature distribution
            # These define the Gaussian distribution for the real data
            m2 = np.mean(stacked_real_features, axis=0)  # Shape: (D,)
            s2 = np.cov(stacked_real_features, rowvar=False)  # Shape: (D, D)

            # Cache the computed statistics for future use
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2

        # Mark that dataset statistics are now available
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        """
        Compute the FID (Fréchet Inception Distance) score.

        This is the main method that orchestrates the entire FID evaluation process.
        It compares the distribution of generated (fake) images to the distribution
        of real images by computing the Fréchet distance between their feature
        distributions in Inception-v3 feature space.

        Process:
        1. Ensure real dataset statistics (μ_real, Σ_real) are loaded/computed
        2. Generate synthetic images using the sampler
        3. Extract Inception features from generated images
        4. Compute statistics (μ_fake, Σ_fake) of generated image features
        5. Calculate Fréchet distance between the two distributions

        The FID formula is:
        FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2(Σ_real × Σ_fake)^(1/2))

        Lower FID scores indicate:
        - Generated images are more similar to real images
        - Better quality and diversity of generated samples
        - The generative model has learned the data distribution well

        Typical FID score ranges:
        - < 10: Excellent quality, very realistic
        - 10-20: Good quality
        - 20-50: Moderate quality
        - > 50: Poor quality or limited diversity

        Returns:
            float: The FID score. Lower is better.

        Note:
            Uses @torch.inference_mode() decorator to disable gradient computation
            and reduce memory usage during evaluation.
        """
        # Ensure we have statistics for the real dataset
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()

        # Set sampler to evaluation mode (disables dropout, etc.)
        self.sampler.eval()

        # Divide total samples into batches (may have unequal last batch)
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []

        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )

        # Generate fake samples in batches and extract their features
        for batch in tqdm(batches):
            # Generate a batch of synthetic images
            fake_samples = self.sampler.sample(batch_size=batch)

            # Extract Inception features from the generated images
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)

        # Concatenate all batches into a single feature array
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()

        # Compute statistics for the generated (fake) image distribution
        m1 = np.mean(stacked_fake_features, axis=0)  # μ_fake: mean vector, shape (D,)
        s1 = np.cov(stacked_fake_features, rowvar=False)  # Σ_fake: covariance matrix, shape (D, D)

        # Calculate and return the Fréchet distance between real and fake distributions
        # This is the final FID score
        return calculate_frechet_distance(m1, s1, self.m2, self.s2)
