"""
The plain Variational Autoencoder: model, loss, training loop and sampling.

Built only from `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.BatchNorm2d`, `nn.Linear`
and activations. No VAE library is used anywhere.

THE IDEA
--------
A VAE assumes images are produced by a two-step story:

    1. draw a latent code     z ~ N(0, I)           <- a short vector, e.g. 128 numbers
    2. decode it into pixels  x ~ p_theta(x | z)    <- the decoder network

To train the decoder we would like p(z | x): given an image, which codes could
have produced it? That is intractable, so we learn an approximation

    q_phi(z | x) = N( mu(x), diag(sigma(x)^2) )     <- the encoder network

THE LOSS
--------
Maximising log p(x) directly is impossible, but we can maximise a lower bound
on it, the Evidence Lower BOund (ELBO):

    log p(x)  >=  E_{z~q}[ log p(x|z) ]  -  KL( q(z|x) || p(z) )

Training minimises the negative, giving two terms with clear jobs:

    loss = reconstruction_error  +  KL

  * reconstruction_error says "the decoder must be able to rebuild this image".
  * KL says "the codes the encoder produces must look like N(0, I)".

The second term is what separates a VAE from an ordinary autoencoder. Without
it the encoder could scatter codes anywhere it liked, and sampling z ~ N(0, I)
at generation time would land in empty space that the decoder never saw. The KL
pulls the encoder's output distribution onto the prior, so the prior becomes a
valid place to sample from.

WHY VAE SAMPLES ARE BLURRY
--------------------------
A per-pixel squared error is the negative log-likelihood of a Gaussian decoder
with fixed variance. Under that model, when several plausible images fit a code
equally well, the loss is minimised by their AVERAGE, not by picking one. The
average of many sharp faces is a smooth face. This is a property of the
objective, not a bug, and it is the main axis on which the DDPM wins.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import save_grid


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Encoder(nn.Module):
    """Maps an image to the mean and log-variance of q(z | x).

    Each stage is Conv(stride=2) -> BatchNorm -> LeakyReLU, halving the spatial
    size and widening the channels. Two linear heads then read the flattened
    feature map.

    We predict LOG-variance rather than variance because a log is unconstrained:
    the network may output any real number, and `exp()` turns it into a positive
    variance. Predicting the variance directly would need a constraint to stop
    it going negative.
    """

    def __init__(self, image_size: int = 64, latent_dim: int = 128,
                 channels: Sequence[int] = (64, 128, 256, 512)) -> None:
        """Build the convolutional trunk and the two heads.

        Args:
            image_size: Input side length; must be divisible by 2**len(channels).
            latent_dim: Size of the latent code z.
            channels: Output channels of each downsampling stage.

        Raises:
            ValueError: If `image_size` is not divisible by the total stride.
        """
        super().__init__()
        stride = 2 ** len(channels)
        if image_size % stride != 0:
            raise ValueError(f"image_size={image_size} must be divisible by {stride}")

        layers: List[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            layers += [
                # kernel 4, stride 2, padding 1 maps H -> H/2 exactly for even H.
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                # LeakyReLU keeps a gradient for negative inputs, which avoids
                # dead units early in training.
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.final_size = image_size // stride          # e.g. 64 / 16 = 4
        self.flat_dim = channels[-1] * self.final_size ** 2   # e.g. 512*4*4 = 8192

        # Two separate heads reading the same features. Kept separate rather
        # than one Linear with 2*latent_dim outputs purely for readability.
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images into posterior parameters.

        Args:
            x: Images of shape (B, 3, S, S) in [-1, 1].

        Returns:
            `(mu, logvar)`, each of shape (B, latent_dim).
        """
        h = self.conv(x).flatten(start_dim=1)           # (B, flat_dim)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Maps a latent code back to an image.

    The exact mirror of the encoder: a Linear reshapes z into a small feature
    map, then ConvTranspose(stride=2) stages double the resolution until the
    original size is reached.
    """

    def __init__(self, image_size: int = 64, latent_dim: int = 128,
                 channels: Sequence[int] = (64, 128, 256, 512)) -> None:
        """Build the projection and the transposed-convolution trunk.

        Args:
            image_size: Output side length (must match the encoder).
            latent_dim: Size of the latent code z.
            channels: The encoder's channel list; walked backwards here.
        """
        super().__init__()
        stride = 2 ** len(channels)
        self.start_ch = channels[-1]
        self.final_size = image_size // stride
        self.fc = nn.Linear(latent_dim, self.start_ch * self.final_size ** 2)

        reversed_ch = list(reversed(channels))          # e.g. [512, 256, 128, 64]
        layers: List[nn.Module] = []
        for i in range(len(reversed_ch) - 1):
            layers += [
                nn.ConvTranspose2d(reversed_ch[i], reversed_ch[i + 1],
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(reversed_ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        # Final upsample back to full resolution.
        layers += [
            nn.ConvTranspose2d(reversed_ch[-1], reversed_ch[-1],
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(reversed_ch[-1]),
            nn.ReLU(inplace=True),
            # A 3x3 conv at full resolution cleans up the checkerboard artefacts
            # that transposed convolutions tend to leave behind.
            nn.Conv2d(reversed_ch[-1], 3, kernel_size=3, padding=1),
            # tanh puts the output in [-1, 1], exactly the range the data lives in.
            nn.Tanh(),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes into images.

        Args:
            z: Latent codes of shape (B, latent_dim).

        Returns:
            Images of shape (B, 3, S, S) in [-1, 1].
        """
        h = self.fc(z).view(-1, self.start_ch, self.final_size, self.final_size)
        return self.deconv(h)


class VAE(nn.Module):
    """Encoder + reparameterisation + decoder.

    Example:
        >>> model = VAE(image_size=64, latent_dim=128)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> x_hat, mu, logvar = model(x)
        >>> x_hat.shape
        torch.Size([2, 3, 64, 64])
    """

    def __init__(self, image_size: int = 64, latent_dim: int = 128,
                 channels: Sequence[int] = (64, 128, 256, 512)) -> None:
        """Create the encoder and decoder.

        Args:
            image_size: Image side length.
            latent_dim: Size of the latent code z.
            channels: Channel widths of the downsampling stages.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = Encoder(image_size, latent_dim, channels)
        self.decoder = Decoder(image_size, latent_dim, channels)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) so that gradients can flow through it.

        Drawing a sample is not a differentiable operation, so we cannot
        backpropagate through `torch.normal(mu, sigma)`. The reparameterisation
        trick rewrites the sample as a deterministic function of the parameters
        plus noise that does not depend on them:

            z = mu + sigma * eps,      eps ~ N(0, I)

        Now dz/dmu = 1 and dz/dsigma = eps, so the encoder receives gradients.
        This single line is what makes a VAE trainable at all.

        Args:
            mu: Posterior means, shape (B, latent_dim).
            logvar: Posterior log-variances, shape (B, latent_dim).

        Returns:
            A sample z of shape (B, latent_dim).
        """
        # sigma = exp(0.5 * log(sigma^2)) = sqrt(variance), always positive.
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode -> sample -> decode pass, as used in training.

        Args:
            x: Images of shape (B, 3, S, S) in [-1, 1].

        Returns:
            `(x_hat, mu, logvar)`: the reconstruction and the posterior params.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate images by drawing codes from the prior p(z) = N(0, I).

        This is the whole generation procedure: one draw, one forward pass. The
        contrast with the DDPM, which needs hundreds of sequential network
        evaluations, is the headline speed difference between the two models.

        Args:
            n: How many images to generate.
            device: Device to run on.

        Returns:
            Images of shape (n, 3, S, S) in [-1, 1].
        """
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode real images, decoding the mean rather than a sample.

        Using `mu` instead of a random draw removes the posterior noise and
        shows the model's best deterministic reconstruction, which is what you
        want when inspecting quality by eye.

        Args:
            x: Images of shape (B, 3, S, S) in [-1, 1].

        Returns:
            Reconstructions of the same shape.
        """
        mu, _ = self.encoder(x)
        return self.decoder(mu)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def vae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ELBO, split into its reconstruction and KL halves.

    Reconstruction term
    -------------------
    Squared error, SUMMED over the 3*64*64 = 12288 pixels of an image and then
    averaged over the batch. The summing is not a detail: the ELBO adds a
    per-image reconstruction term to a per-image KL term, so both must be "per
    image". Averaging over pixels instead would shrink the reconstruction term
    by a factor of 12288, the KL would dominate completely, and the model would
    collapse to outputting the dataset mean.

    KL term
    -------
    Both q(z|x) and the prior N(0, I) are diagonal Gaussians, so their KL has a
    closed form and needs no sampling. Per dimension:

        KL = 0.5 * ( sigma^2 + mu^2 - 1 - log(sigma^2) )

    Reading it: `mu^2` punishes means far from zero, and `sigma^2 - 1 - logvar`
    punishes variances away from one. The expression is zero exactly when
    mu = 0 and sigma = 1, that is, when the posterior equals the prior.

    Args:
        x_hat: Reconstruction, shape (B, 3, H, W) in [-1, 1].
        x: Target image, shape (B, 3, H, W) in [-1, 1].
        mu: Posterior means, shape (B, latent_dim).
        logvar: Posterior log-variances, shape (B, latent_dim).

    Returns:
        A 3-tuple `(total, recon, kl)` of scalars. `total = recon + kl` and is
        the tensor to call `.backward()` on; the other two are for logging.
    """
    # reduction="none" keeps the full error map, so we can choose how to reduce.
    per_pixel = F.mse_loss(x_hat, x, reduction="none")
    # Sum over channels+height+width -> one number per image, then average.
    recon = per_pixel.flatten(start_dim=1).sum(dim=1).mean()

    per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)   # (B, latent_dim)
    kl = per_dim.sum(dim=1).mean()                              # sum dims, mean batch

    return recon + kl, recon.detach(), kl.detach()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_vae(train_loader, val_loader, device: torch.device, out_dir: str | Path,
              image_size: int = 64, latent_dim: int = 128, epochs: int = 20,
              lr: float = 2e-4, amp: bool = True) -> Path:
    """Train the VAE and write checkpoints, samples and a loss history.

    Args:
        train_loader: DataLoader over training images in [-1, 1].
        val_loader: DataLoader over held-out images.
        device: Device to train on.
        out_dir: Run folder; created if missing.
        image_size: Image side length.
        latent_dim: Size of the latent code.
        epochs: Number of passes over the training set.
        lr: Adam learning rate.
        amp: Use float16 autocast on CUDA. Roughly 2x faster, no quality cost.

    Returns:
        Path to the run folder.
    """
    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    model = VAE(image_size, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAE: {n_params / 1e6:.1f}M parameters, latent_dim={latent_dim}, "
          f"device={device}")

    history: List[Dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(epochs):
        # -- training pass --------------------------------------------------- #
        model.train()
        start = time.time()
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0

        # leave=False erases the bar when the epoch finishes, so only the
        # one-line summary printed below survives in the log.
        bar = tqdm(train_loader, desc=f"epoch {epoch:3d} train", leave=False)
        for x in bar:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                x_hat, mu, logvar = model(x)
            # The loss is computed in float32 on purpose: summing 12288 squared
            # errors overflows float16, whose maximum is about 65504.
            loss, recon, kl = vae_loss(x_hat.float(), x.float(),
                                       mu.float(), logvar.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            sums["loss"] += loss.item()
            sums["recon"] += recon.item()
            sums["kl"] += kl.item()
            n_batches += 1
            # Running averages, not the single-batch value: less jitter to read.
            bar.set_postfix(loss=f"{sums['loss'] / n_batches:.1f}",
                            recon=f"{sums['recon'] / n_batches:.1f}",
                            kl=f"{sums['kl'] / n_batches:.1f}")

        train_metrics = {k: v / max(n_batches, 1) for k, v in sums.items()}

        # -- validation pass -------------------------------------------------- #
        # eval() switches BatchNorm to its running averages, which is essential
        # here because generation may run with a batch of one.
        model.eval()
        val_total, val_batches = 0.0, 0
        with torch.no_grad():
            for x in tqdm(val_loader, desc=f"epoch {epoch:3d} val", leave=False):
                x = x.to(device, non_blocking=True)
                x_hat, mu, logvar = model(x)
                loss, _, _ = vae_loss(x_hat.float(), x.float(),
                                      mu.float(), logvar.float())
                val_total += loss.item()
                val_batches += 1
        val_loss = val_total / max(val_batches, 1)

        elapsed = time.time() - start
        print(f"epoch {epoch:3d} | {elapsed:6.1f}s | train {train_metrics['loss']:8.2f} "
              f"(recon {train_metrics['recon']:7.2f}, kl {train_metrics['kl']:6.2f}) "
              f"| val {val_loss:8.2f}")
        history.append({"epoch": epoch, "seconds": elapsed, "val_loss": val_loss,
                        **train_metrics})

        # -- checkpoints and a visual --------------------------------------- #
        checkpoint = {"model": model.state_dict(),
                      "config": {"image_size": image_size, "latent_dim": latent_dim}}
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")

        save_grid(model.sample(64, device), out_dir / "samples" / f"epoch_{epoch:03d}.png")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    return out_dir


def load_vae(run_dir: str | Path, device: torch.device,
             checkpoint: str = "best.pt") -> VAE:
    """Rebuild a trained VAE from its run folder.

    The checkpoint stores the architecture settings next to the weights, so no
    command-line flags need to be repeated at sampling time.

    Args:
        run_dir: Folder written by `train_vae`.
        device: Device to place the model on.
        checkpoint: "best.pt" or "last.pt".

    Returns:
        The model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    path = Path(run_dir) / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = VAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def generate(run_dir: str | Path, device: torch.device, n: int,
             out_dir: Optional[str | Path] = None, batch_size: int = 256) -> Path:
    """Generate `n` images and write them as individual PNGs for FID.

    Args:
        run_dir: Folder written by `train_vae`.
        device: Device to run on.
        n: Number of images to generate.
        out_dir: Destination folder; defaults to `<run_dir>/generated`.
        batch_size: Images per decoder call.

    Returns:
        The destination folder.
    """
    from data import save_pngs

    model = load_vae(run_dir, device)
    out_dir = Path(out_dir) if out_dir else Path(run_dir) / "generated"
    written = 0
    with tqdm(total=n, desc="vae generate", unit="img") as bar:
        while written < n:
            batch = model.sample(min(batch_size, n - written), device)
            just_written = save_pngs(batch, out_dir, start_index=written)
            written += just_written
            bar.update(just_written)
    print(f"wrote {written} images to {out_dir}")
    return out_dir
