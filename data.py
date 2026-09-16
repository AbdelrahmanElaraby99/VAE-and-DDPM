"""
CelebA loading and image saving. The only file that touches the disk.

Both models in this project are UNCONDITIONAL, so we never need the attribute
file (`list_attr_celeba.txt`). That lets the dataset be as simple as it can be:
list the JPEGs in a folder, and load one when asked.

The preprocessing pipeline, in order:

    1. CenterCrop(178)  CelebA's aligned images are 178x218. The extra height is
                        background above and below the face. Cropping a 178x178
                        square keeps the face and drops the filler.
    2. Resize(64)       Down to the training resolution.
    3. RandomFlip       Training only. Faces are roughly symmetric, so mirroring
                        is a free doubling of the dataset. Never applied to
                        validation or FID reference images, because those must
                        be identical from run to run.
    4. ToTensor         PIL image -> float tensor (3, H, W) in [0, 1].
    5. Normalize        [0, 1] -> [-1, 1].

Step 5 is worth pausing on, because both models depend on it:

  * The VAE decoder ends in `tanh`, whose output range is exactly [-1, 1].
  * The DDPM assumes the data has roughly zero mean and unit variance, because
    its forward process mixes the image with standard Gaussian noise. Data in
    [0, 1] has mean ~0.5, and that offset would show up as a colour cast.

We call [-1, 1] "model space" and [0, 1] "view space" throughout the project.
Anything written to a PNG is converted back to view space first.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

# CelebA's aligned images are 178 wide and 218 tall; 178 is the square side.
CELEBA_CROP = 178


class CelebA(Dataset):
    """The aligned CelebA images as a list of tensors in [-1, 1].

    Attributes:
        paths: Sorted list of JPEG paths included in this (sub)set.
        transform: The preprocessing pipeline applied to each image.
    """

    def __init__(self, root: str | Path, image_size: int = 64,
                 indices: Optional[List[int]] = None, flip: bool = True) -> None:
        """Index the image folder. No pixels are read until `__getitem__`.

        Args:
            root: Folder containing `img_align_celeba/`.
            image_size: Output side length in pixels.
            indices: Optional subset of positions in the sorted file list, used
                to carve out a train/validation split without re-listing files.
                `None` means "use every image".
            flip: Enable random horizontal mirroring (training only).

        Raises:
            FileNotFoundError: If the image folder holds no JPEGs.
        """
        image_dir = Path(root) / "img_align_celeba"
        # Some CelebA downloads nest the folder one level deeper.
        if (image_dir / "img_align_celeba").exists():
            image_dir = image_dir / "img_align_celeba"

        # sorted() makes the file order identical on every machine, which is
        # what makes the train/val split below reproducible.
        all_paths = sorted(image_dir.glob("*.jpg"))
        if not all_paths:
            raise FileNotFoundError(
                f"No .jpg files under {image_dir}. Point --data-root at the "
                f"folder that contains img_align_celeba/."
            )

        self.paths = all_paths if indices is None else [all_paths[i] for i in indices]

        ops = [
            transforms.CenterCrop(CELEBA_CROP),
            transforms.Resize(image_size, antialias=True),
        ]
        if flip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops += [
            transforms.ToTensor(),                                  # -> [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),             # -> [-1, 1]
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        """Number of images in this (sub)set."""
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        """Load and preprocess one image.

        Args:
            i: Position in `self.paths`.

        Returns:
            Float tensor of shape (3, S, S) with values in [-1, 1].
        """
        # convert("RGB") guards against the occasional greyscale JPEG.
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def make_loaders(root: str | Path, image_size: int = 64, batch_size: int = 128,
                 limit: Optional[int] = None, val_fraction: float = 0.02,
                 num_workers: int = 4, seed: int = 0
                 ) -> Tuple[DataLoader, DataLoader]:
    """Build the training and validation DataLoaders.

    The split is a deterministic permutation seeded by `seed`, so the same
    images land in validation on every run and validation losses are comparable
    across runs.

    Args:
        root: Folder containing `img_align_celeba/`.
        image_size: Output side length in pixels.
        batch_size: Images per batch.
        limit: Use only the first `limit` images. Useful for smoke tests.
        val_fraction: Fraction held out for validation.
        num_workers: Background JPEG-decoding processes.
        seed: Seed for the split permutation.

    Returns:
        A 2-tuple `(train_loader, val_loader)`.
    """
    # Count files once, cheaply, to know how big the permutation must be.
    n_total = len(CelebA(root, image_size, flip=False))
    if limit is not None:
        n_total = min(n_total, limit)

    # A local generator, so seeding the split does not disturb the global RNG
    # that the training loop uses for weight init and noise.
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()

    n_val = max(1, int(round(n_total * val_fraction)))
    val_ds = CelebA(root, image_size, perm[:n_val], flip=False)
    train_ds = CelebA(root, image_size, perm[n_val:], flip=True)

    shared = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **shared)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **shared)
    return train_loader, val_loader


def reference_dataset(root: str | Path, image_size: int = 64,
                      n: int = 10000) -> CelebA:
    """The real images FID measures generated samples against.

    Flipping is disabled: FID reference statistics must be deterministic, or the
    same model would score differently on two runs.

    Args:
        root: Folder containing `img_align_celeba/`.
        image_size: Must match the resolution the models were trained at.
        n: How many real images to use.

    Returns:
        A `CelebA` dataset over the first `n` images, unaugmented.
    """
    return CelebA(root, image_size, indices=list(range(n)), flip=False)


# --------------------------------------------------------------------------- #
# Image output
# --------------------------------------------------------------------------- #


def to_view(x: torch.Tensor) -> torch.Tensor:
    """Convert model space [-1, 1] to view space [0, 1] for display or saving.

    Args:
        x: Tensor with values in [-1, 1].

    Returns:
        Tensor of the same shape, clamped into [0, 1].
    """
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> Path:
    """Tile a batch of images into one PNG.

    Args:
        images: Tensor of shape (N, 3, H, W) in model space [-1, 1].
        path: Destination PNG path; parent folders are created.
        nrow: Images per row in the tiled output.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_view(images), path, nrow=nrow)
    return path


def save_pngs(images: torch.Tensor, out_dir: str | Path, start_index: int = 0) -> int:
    """Write each image in a batch as its own PNG, for FID scoring.

    Args:
        images: Tensor of shape (N, 3, H, W) in model space [-1, 1].
        out_dir: Destination folder; created if missing.
        start_index: Number to begin the filenames at, so repeated calls do not
            overwrite each other.

    Returns:
        How many files were written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view = to_view(images)
    for i, img in enumerate(view):
        save_image(img, out_dir / f"{start_index + i:06d}.png")
    return len(view)
