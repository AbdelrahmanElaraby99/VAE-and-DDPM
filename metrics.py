"""
FID and Inception Score, written out from their definitions.

Neither metric is imported from a library. The only thing borrowed is the
pretrained InceptionV3 network, and that is unavoidable: FID and IS are
*defined* as statistics of InceptionV3 activations, so reimplementing them
without that network would mean computing a different quantity. Training an
ImageNet classifier from scratch is not part of this task. The mathematics
below -- the Frechet distance and the KL-based score -- is written out in full.

WHAT THE TWO NUMBERS MEAN
-------------------------
FID (lower is better) summarises each image collection as a single 2048-
dimensional Gaussian and measures the distance between the two Gaussians. It
responds to both realism and diversity: a model that produces one perfect face
every time scores terribly, because its covariance collapses.

IS (higher is better) asks whether each image is confidently classified AND
whether the collection covers many classes. It needs no real images at all,
which is its weakness -- it cannot tell you that your samples look nothing like
the dataset.

A caveat worth saying out loud in any write-up: IS uses ImageNet class
probabilities, and a face is not an ImageNet class. On CelebA, IS is a much
weaker signal than FID. We compute it because the task asks for it, and
interpret it with care.

A second caveat: published FID numbers come from the original TensorFlow
Inception graph, whose weights differ slightly from torchvision's port. Absolute
values here are close to, but not identical with, numbers in papers. Since every
model in this project is scored with the same extractor, comparisons *between*
them remain valid -- and that comparison is the point.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# The statistics InceptionV3 was trained with, and the resolution it expects.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_SIZE = 299


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


class Inception(nn.Module):
    """InceptionV3, exposing both the 2048-d pool features and the 1000 logits.

    FID needs the pooled features; IS needs the class logits. Both come from
    the same forward pass, so we take them together.
    """

    def __init__(self) -> None:
        """Load the pretrained ImageNet weights and freeze them."""
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                aux_logits=True, init_weights=False)
        # Keep the classifier aside and replace it with Identity, so `net(x)`
        # returns the 2048-d feature vector and we can apply `fc` ourselves.
        self.fc = self.net.fc
        self.net.fc = nn.Identity()

        self.eval()
        # This is a fixed measuring instrument; it is never trained.
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features and logits from a batch of images.

        Args:
            x: Images of shape (B, 3, H, W) in view space [0, 1]. Any H and W
                are accepted; they are resized internally.

        Returns:
            `(features, logits)` of shapes (B, 2048) and (B, 1000).
        """
        if x.shape[-1] != INCEPTION_SIZE:
            x = F.interpolate(x, size=(INCEPTION_SIZE, INCEPTION_SIZE),
                              mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        features = self.net(x)          # (B, 2048), because fc is Identity
        return features, self.fc(features)


class PngFolder(Dataset):
    """Reads a flat folder of PNGs -- the output of the `generate` commands."""

    def __init__(self, folder: str | Path, limit: int | None = None) -> None:
        """Index the image files.

        Args:
            folder: Directory holding .png files.
            limit: Optionally cap how many are used.

        Raises:
            FileNotFoundError: If the folder is missing or holds no images.
        """
        folder = Path(folder)
        # sorted() keeps the ordering identical across machines.
        self.paths = sorted(p for p in folder.glob("*.png")) if folder.exists() else []
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No .png files in {folder}")

    def __len__(self) -> int:
        """Number of images found."""
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        """Load one image as a [0, 1] tensor.

        Args:
            i: Index into the sorted file list.

        Returns:
            Tensor of shape (3, H, W) in [0, 1].
        """
        arr = np.array(Image.open(self.paths[i]).convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def extract_features(source, model: Inception, device: torch.device,
                     batch_size: int = 50, max_images: int | None = None,
                     model_space: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Inception over a folder of PNGs or over a Dataset.

    Args:
        source: A path to a PNG folder, or a Dataset yielding image tensors.
        model: The `Inception` extractor, already on `device`.
        device: Device to run on.
        batch_size: Images per forward pass.
        max_images: Stop after this many images.
        model_space: True if the source yields [-1, 1] tensors (as `CelebA`
            does); False for [0, 1] (as a PNG folder does).

    Returns:
        `(features, logits)` of shapes (N, 2048) and (N, 1000), on CPU.
    """
    if isinstance(source, (str, Path)):
        dataset = PngFolder(source, limit=max_images)
        model_space = False
    else:
        dataset = source

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    seen = 0

    total = len(dataset) if max_images is None else min(max_images, len(dataset))
    bar = tqdm(total=total, desc="inception", unit="img", leave=False)

    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        if model_space:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1).to(device)

        if max_images is not None and seen + images.shape[0] > max_images:
            images = images[:max_images - seen]

        f, lg = model(images)
        features.append(f.cpu())
        logits.append(lg.cpu())
        seen += images.shape[0]
        bar.update(images.shape[0])
        if max_images is not None and seen >= max_images:
            break

    bar.close()
    return torch.cat(features), torch.cat(logits)


# --------------------------------------------------------------------------- #
# Frechet Inception Distance
# --------------------------------------------------------------------------- #


def gaussian_statistics(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Summarise a feature set by its mean vector and covariance matrix.

    FID models each collection as one 2048-dimensional Gaussian. That is a
    strong assumption, but it is what makes the distance computable in closed
    form.

    Args:
        features: Tensor of shape (N, D) of Inception activations.

    Returns:
        `(mu, sigma)` with shapes (D,) and (D, D), in float64.

    Raises:
        ValueError: If fewer than two samples are given.
    """
    # float64 throughout: the covariance is badly conditioned and float32
    # rounding shows up in the matrix square root below.
    x = features.double().numpy()
    n_samples, n_features = x.shape
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for a covariance, got {n_samples}")

    # A D x D covariance estimated from N <= D samples has rank at most N-1, so
    # it is singular. FID still returns a number, but that number is dominated
    # by estimation noise and is not comparable with anything.
    if n_samples <= n_features:
        warnings.warn(
            f"FID from only {n_samples} samples with {n_features} feature "
            f"dimensions: the covariance is rank-deficient, so the value will be "
            f"inflated and unreliable. Use 10000 images for a number worth quoting.",
            RuntimeWarning, stacklevel=2)

    # rowvar=False: rows are observations, columns are variables.
    return x.mean(axis=0), np.cov(x, rowvar=False)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """The Frechet (2-Wasserstein) distance between two Gaussians.

        d^2 = ||mu1 - mu2||^2  +  Tr( S1 + S2 - 2 * (S1 @ S2)^(1/2) )

    The first term measures how far apart the two "average images" sit in
    feature space. The second measures how differently the two collections are
    spread out. Zero means the two Gaussians are identical.

    The awkward part is `(S1 @ S2)^(1/2)`: a MATRIX square root, not an
    element-wise one. The product of two symmetric positive-definite matrices
    need not itself be symmetric, so we use SciPy's general `sqrtm` and then
    discard the tiny imaginary component that rounding leaves behind.

    Args:
        mu1: Mean of the first set, shape (D,).
        sigma1: Covariance of the first set, shape (D, D).
        mu2: Mean of the second set, shape (D,).
        sigma2: Covariance of the second set, shape (D, D).
        eps: Ridge added to the diagonals if `sqrtm` returns a singular result.

    Returns:
        The FID as a float.

    Raises:
        RuntimeError: If `sqrtm` leaves a non-negligible imaginary component.
    """
    diff = mu1 - mu2

    with warnings.catch_warnings():
        # We handle the singular case explicitly just below.
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        # Older SciPy versions return (matrix, error_estimate).
        if isinstance(covmean, tuple):
            covmean = covmean[0]
        # A singular product can make sqrtm return NaNs; nudging the diagonals
        # makes it positive-definite and the call well-posed.
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if isinstance(covmean, tuple):
                covmean = covmean[0]

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise RuntimeError(f"sqrtm left an imaginary part of "
                               f"{np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2.0 * np.trace(covmean))


def compute_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """FID between a set of real and a set of generated features.

    Args:
        real_features: Tensor of shape (N_real, 2048).
        fake_features: Tensor of shape (N_fake, 2048).

    Returns:
        The FID; lower is better.
    """
    mu_r, sigma_r = gaussian_statistics(real_features)
    mu_f, sigma_f = gaussian_statistics(fake_features)
    return frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


# --------------------------------------------------------------------------- #
# Inception Score
# --------------------------------------------------------------------------- #


def compute_inception_score(logits: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
    """The Inception Score: are the samples confident AND varied?

        IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )

    Reading the two distributions:
      * p(y|x) is the classifier's output for ONE image. A clear, recognisable
        image produces a peaked, low-entropy distribution.
      * p(y) is the average over all images. A varied collection covers many
        classes, so p(y) is close to uniform and high-entropy.

    Their KL divergence is therefore large exactly when individual samples are
    confident and the collection as a whole is diverse -- which is what we want
    a single number to capture.

    The score is computed on `splits` disjoint chunks and reported as mean and
    standard deviation, as in the original paper, because the value is
    noticeably sensitive to sample size.

    Args:
        logits: Raw classifier outputs of shape (N, 1000).
        splits: Number of chunks to average over.

    Returns:
        `(mean, std)` of the per-split scores.

    Raises:
        ValueError: If there are fewer samples than splits.
    """
    n = logits.shape[0]
    if n < splits:
        raise ValueError(f"Need at least {splits} samples for {splits} splits, got {n}")

    # softmax over the class axis turns logits into p(y|x).
    probs = F.softmax(logits.double(), dim=1)

    scores = []
    size = n // splits
    for i in range(splits):
        chunk = probs[i * size:(i + 1) * size]               # (M, 1000)
        marginal = chunk.mean(dim=0, keepdim=True)           # p(y), (1, 1000)
        # KL per image = sum_y p(y|x) * (log p(y|x) - log p(y)).
        # The 1e-12 floors avoid log(0) for classes with zero probability.
        kl = chunk * (torch.log(chunk + 1e-12) - torch.log(marginal + 1e-12))
        scores.append(torch.exp(kl.sum(dim=1).mean()).item())

    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def evaluate(generated_dir: str | Path, real_dataset, device: torch.device,
             num_images: int = 10000) -> dict:
    """Score a folder of generated images against real CelebA.

    Args:
        generated_dir: Folder of PNGs written by a `generate` command.
        real_dataset: A `CelebA` dataset supplying the reference images.
        device: Device to run Inception on.
        num_images: How many images from each side to use.

    Returns:
        A dict with "fid", "is_mean", "is_std" and "num_images".
    """
    model = Inception().to(device)

    print("extracting features from real images")
    real_features, _ = extract_features(real_dataset, model, device,
                                        max_images=num_images, model_space=True)
    print("extracting features from generated images")
    fake_features, fake_logits = extract_features(generated_dir, model, device,
                                                  max_images=num_images)

    fid = compute_fid(real_features, fake_features)
    is_mean, is_std = compute_inception_score(fake_logits)

    return {"fid": fid, "is_mean": is_mean, "is_std": is_std,
            "num_images": int(min(len(real_features), len(fake_features)))}
