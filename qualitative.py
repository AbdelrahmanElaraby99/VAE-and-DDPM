"""The qualitative figures and the DDIM step sweep quoted in REPORT.md.

    python qualitative.py figures      write every figure
    python qualitative.py sweep        FID/IS for DDIM at 10, 25, 50 steps
    python qualitative.py all          both

Everything here reads the trained checkpoints in `runs/` and writes into
`runs/*/figures/`. Nothing is trained. This is a reporting script, kept apart
from `main.py` so the pipeline there stays the short, obvious one.

The figures answer questions a metric cannot:

    comparison          do the samples look like faces, next to real ones?
    reconstructions     what does the VAE keep and what does it throw away?
    interpolation       is the latent space smooth, or a lookup table?
    ddim_steps          how few sampling steps can we get away with?
    trajectory          what does the reverse process actually do?
    nearest_neighbours  is the model generating, or memorising?
    typical_atypical    what do the failures look like?
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

import ddpm as ddpm_module
import vae as vae_module
from data import CelebA, reference_dataset, to_view

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parent / "CelebA"
RUNS = HERE / "runs"
SHARED_FIGURES = RUNS / "figures"


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #


def panel(ax, images: torch.Tensor, nrow: int, title: str = "",
          pad: int = 2) -> None:
    """Draw one tiled block of images into a matplotlib axis.

    Args:
        ax: The axis to draw on.
        images: Tensor of shape (N, 3, H, W) in model space [-1, 1].
        nrow: Images per row.
        title: Optional heading above the block.
        pad: Pixels of white between tiles.
    """
    grid = make_grid(to_view(images.cpu()), nrow=nrow, padding=pad, pad_value=1.0)
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=6)


def save(fig, path: Path) -> Path:
    """Write a figure and report where it went.

    Args:
        fig: The matplotlib figure.
        path: Destination PNG path; parent folders are created.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation between two Gaussian noise tensors.

    Straight-line interpolation between two samples of N(0, I) passes through
    points with norm well below the typical norm of the distribution, which the
    model has never seen; the midpoint comes out washed out. Interpolating along
    the sphere keeps the norm roughly constant the whole way.

    Args:
        a: Start tensor.
        b: End tensor.
        t: Position in [0, 1].

    Returns:
        The interpolated tensor.
    """
    a_flat, b_flat = a.flatten(), b.flatten()
    cos = (a_flat @ b_flat / (a_flat.norm() * b_flat.norm())).clamp(-1.0, 1.0)
    omega = torch.acos(cos)
    if omega.abs() < 1e-6:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


# --------------------------------------------------------------------------- #
# DDIM with a caller-supplied starting noise
# --------------------------------------------------------------------------- #


@torch.no_grad()
def ddim_from_noise(diffusion, x: torch.Tensor, num_steps: int = 100,
                    capture: Optional[Sequence[int]] = None):
    """Run deterministic DDIM starting from a given `x_T`.

    `Diffusion.ddim_sample` draws its own starting noise, which is what you want
    for generation but not for interpolation or for tracing one trajectory. This
    is the same update with `x_T` passed in, plus an option to record
    intermediate states.

    Args:
        diffusion: A loaded `Diffusion`.
        x: Starting noise of shape (B, 3, S, S).
        num_steps: How many of the T timesteps to visit.
        capture: Step indices (into the descending schedule) to record. When
            given, the return value is `(final, [(t, x_t, x0_hat), ...])`.

    Returns:
        The final images, or `(final, captured)` when `capture` is given.
    """
    times = torch.linspace(0, diffusion.timesteps - 1, num_steps).long().tolist()
    times = list(reversed(times))
    pairs = list(zip(times, times[1:] + [-1]))
    capture_at = set(capture or [])
    recorded = []

    for i, (t_now, t_next) in enumerate(pairs):
        t = torch.full((x.shape[0],), t_now, device=x.device, dtype=torch.long)
        eps = diffusion.model(x, t)
        x_start = diffusion.predict_x_start(x, t, eps).clamp(-1.0, 1.0)

        if i in capture_at:
            recorded.append((t_now, x.clone(), x_start.clone()))

        if t_next < 0:
            x = x_start
            break

        ab = ddpm_module.extract(diffusion.alphas_bar, t, x.shape)
        t_prev = torch.full((x.shape[0],), t_next, device=x.device, dtype=torch.long)
        ab_prev = ddpm_module.extract(diffusion.alphas_bar, t_prev, x.shape)
        x = torch.sqrt(ab_prev) * x_start + torch.sqrt(torch.relu(1 - ab_prev)) * eps

    if capture is not None:
        recorded.append((0, x.clone(), x.clone()))
        return x, recorded
    return x


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_comparison(vae, diffusion, real: torch.Tensor, device, seed: int = 7) -> None:
    """Real images, VAE samples and DDPM samples side by side, same scale."""
    torch.manual_seed(seed)
    with torch.no_grad():
        vae_x = vae.sample(16, device)
    torch.manual_seed(seed)
    ddpm_x = diffusion.ddim_sample((16, 3, 64, 64), device, num_steps=100,
                                   progress=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    panel(axes[0], real[:16], 4, "Real CelebA")
    panel(axes[1], vae_x, 4, "VAE  (1 forward pass)")
    panel(axes[2], ddpm_x, 4, "DDPM  (DDIM, 100 steps)")
    fig.suptitle("Uncurated samples at 64x64, identical display scale",
                 fontsize=12, y=1.0)
    save(fig, SHARED_FIGURES / "comparison.png")


def fig_vae_interpolation(vae, real: torch.Tensor, device, n_pairs: int = 5,
                          n_steps: int = 9) -> None:
    """Walk the VAE latent space in a straight line between two real faces."""
    with torch.no_grad():
        mu, _ = vae.encoder(real.to(device))
    rows = []
    for p in range(n_pairs):
        # Pair the front of the batch with the back, so the two endpoints are
        # as unlike each other as the batch allows.
        a, b = mu[p], mu[-1 - p]
        zs = torch.stack([a + (b - a) * (k / (n_steps - 1)) for k in range(n_steps)])
        with torch.no_grad():
            rows.append(vae.decoder(zs))
    images = torch.cat(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    panel(ax, images, n_steps,
          "VAE latent interpolation: decode(mu_a + t(mu_b - mu_a)), t = 0 .. 1")
    save(fig, RUNS / "vae" / "figures" / "interpolation.png")


def fig_ddim_interpolation(diffusion, device, n_pairs: int = 4,
                           n_steps: int = 9, seed: int = 3) -> None:
    """Walk the DDPM's noise space with deterministic DDIM (eta = 0)."""
    torch.manual_seed(seed)
    rows = []
    for _ in range(n_pairs):
        a = torch.randn(3, 64, 64, device=device)
        b = torch.randn(3, 64, 64, device=device)
        x_t = torch.stack([slerp(a, b, k / (n_steps - 1)) for k in range(n_steps)])
        rows.append(ddim_from_noise(diffusion, x_t, num_steps=100))
    images = torch.cat(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    panel(ax, images, n_steps,
          "DDIM interpolation (eta = 0): spherical walk in x_T, 100 steps each")
    save(fig, RUNS / "ddpm" / "figures" / "interpolation.png")


def fig_trajectory(diffusion, device, n_cols: int = 8, num_steps: int = 100,
                   seed: int = 11) -> None:
    """The reverse process on one seed: the running state and its x0 estimate."""
    torch.manual_seed(seed)
    x_t = torch.randn(1, 3, 64, 64, device=device)
    picks = [round(k * (num_steps - 1) / (n_cols - 1)) for k in range(n_cols - 1)]
    _, recorded = ddim_from_noise(diffusion, x_t, num_steps=num_steps, capture=picks)

    ts = [t for t, _, _ in recorded]
    xt = torch.cat([a for _, a, _ in recorded])
    x0 = torch.cat([b for _, _, b in recorded])

    fig, axes = plt.subplots(2, 1, figsize=(11, 3.4))
    panel(axes[0], xt, len(ts), "")
    panel(axes[1], x0, len(ts), "")
    axes[0].set_ylabel("x_t", fontsize=10)
    axes[1].set_ylabel("x0_hat", fontsize=10)
    axes[0].set_title("t = " + "   ".join(str(t) for t in ts), fontsize=9)
    fig.suptitle("The DDIM reverse process: state (top) and the model's guess at "
                 "the clean image (bottom)", fontsize=11, y=1.12)
    save(fig, RUNS / "ddpm" / "figures" / "trajectory.png")


def fig_ddim_steps(diffusion, device, counts=(5, 10, 25, 50, 100, 250),
                   n_rows: int = 4, seed: int = 5) -> None:
    """The same starting noise decoded with more and more sampling steps."""
    columns = []
    for steps in counts:
        torch.manual_seed(seed)
        x_t = torch.randn(n_rows, 3, 64, 64, device=device)
        columns.append(ddim_from_noise(diffusion, x_t, num_steps=steps).cpu())

    torch.manual_seed(seed)
    ancestral = diffusion.p_sample_loop((n_rows, 3, 64, 64), device,
                                        progress=False).cpu()
    columns.append(ancestral)
    labels = [f"DDIM {c}" for c in counts] + ["ancestral\n1000"]

    fig, axes = plt.subplots(1, len(columns), figsize=(1.55 * len(columns), 6.2))
    for ax, col, label in zip(axes, columns, labels):
        panel(ax, col, 1, label)
    fig.suptitle("Same x_T, increasing sampling budget", fontsize=12, y=0.96)
    save(fig, RUNS / "ddpm" / "figures" / "ddim_steps.png")


def fig_training_progress(epochs=(0, 2, 5, 9, 14, 19)) -> None:
    """The per-epoch preview grids written during training, laid out in a row.

    Reads the PNGs both training loops already save, so it costs nothing and
    needs no checkpoint.
    """
    from PIL import Image

    for model, note in [("vae", "VAE samples"), ("ddpm", "DDPM samples (EMA weights)")]:
        available = [e for e in epochs
                     if (RUNS / model / "samples" / f"epoch_{e:03d}.png").exists()]
        fig, axes = plt.subplots(1, len(available),
                                 figsize=(2.05 * len(available), 2.5))
        for ax, e in zip(axes, available):
            ax.imshow(Image.open(RUNS / model / "samples" / f"epoch_{e:03d}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"epoch {e + 1}", fontsize=10, pad=4)
        fig.suptitle(note, fontsize=12, y=1.02)
        fig.tight_layout()
        save(fig, RUNS / model / "figures" / "training_progress.png")


# --------------------------------------------------------------------------- #
# Feature-space figures: memorisation and failure cases
# --------------------------------------------------------------------------- #


def unit(features: torch.Tensor) -> torch.Tensor:
    """L2-normalise feature rows so a dot product is a cosine similarity."""
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)


def nearest(query: torch.Tensor, bank: torch.Tensor, k: int, device):
    """Top-k cosine neighbours of each query row within `bank`.

    Args:
        query: (Q, D) features.
        bank: (N, D) features to search.
        k: Neighbours to return.
        device: Device for the matmul.

    Returns:
        `(values, indices)`, each (Q, k), sorted by decreasing similarity.
    """
    q = unit(query).to(device)
    best_v = torch.full((q.shape[0], k), -2.0, device=device)
    best_i = torch.zeros((q.shape[0], k), dtype=torch.long, device=device)
    for start in range(0, bank.shape[0], 4096):
        chunk = unit(bank[start:start + 4096]).to(device)
        sim = q @ chunk.T
        v, i = torch.cat([best_v, sim], dim=1).topk(k, dim=1)
        pool = torch.cat([best_i, torch.arange(start, start + chunk.shape[0],
                                               device=device).expand(q.shape[0], -1)],
                         dim=1)
        best_v, best_i = v, pool.gather(1, i)
    return best_v.cpu(), best_i.cpu()


def fig_nearest_neighbours(features, real_ds, device, n_show: int = 6,
                           k: int = 4) -> None:
    """For a few DDPM samples, the closest real images in Inception space.

    A generative model that had merely memorised the training set would produce
    samples whose nearest real neighbour is the same image, pixel for pixel.
    """
    from metrics import PngFolder

    gen_features = features["ddpm_gen"]
    real_features = features["real"]
    torch.manual_seed(0)
    picks = torch.randperm(gen_features.shape[0])[:n_show].tolist()
    sims, idx = nearest(gen_features[picks], real_features, k, device)

    gen_folder = PngFolder(RUNS / "ddpm" / "generated")
    rows = []
    for row, p in enumerate(picks):
        rows.append(gen_folder[p] * 2 - 1)
        rows.extend(real_ds[int(j)] for j in idx[row])
    images = torch.stack(rows)

    fig, ax = plt.subplots(figsize=(6.5, 9))
    panel(ax, images, k + 1,
          f"DDPM sample (left column) and its {k} nearest real CelebA images\n"
          f"(cosine similarity in Inception feature space, "
          f"{real_features.shape[0]:,} reals searched)")
    save(fig, RUNS / "ddpm" / "figures" / "nearest_neighbours.png")
    print("  nearest-neighbour cosine similarities:",
          [round(float(s), 3) for s in sims[:, 0]])
    return sims


def fig_typical_atypical(features, device, n_show: int = 8) -> None:
    """The most and least real-looking samples of each model, ranked by feature
    similarity to the nearest real image."""
    from metrics import PngFolder

    fig, axes = plt.subplots(2, 2, figsize=(13, 3.4))
    stats = {}
    for col, model in enumerate(["vae", "ddpm"]):
        sims, _ = nearest(features[f"{model}_gen"], features["real"], 1, device)
        order = sims[:, 0].argsort()
        folder = PngFolder(RUNS / model / "generated")
        best = torch.stack([folder[int(i)] * 2 - 1 for i in order[-n_show:]])
        worst = torch.stack([folder[int(i)] * 2 - 1 for i in order[:n_show]])
        panel(axes[0][col], best, n_show, f"{model.upper()}  -  most typical")
        panel(axes[1][col], worst, n_show, f"{model.upper()}  -  least typical")
        stats[model] = {"median_nn_similarity": float(sims[:, 0].median()),
                        "min_nn_similarity": float(sims[:, 0].min())}
    fig.suptitle("Ranked by cosine similarity to the closest real image "
                 "(Inception features)", fontsize=12, y=1.06)
    fig.tight_layout()
    save(fig, SHARED_FIGURES / "typical_atypical.png")
    return stats


# --------------------------------------------------------------------------- #
# The DDIM step / FID sweep
# --------------------------------------------------------------------------- #


def sweep(args, device) -> dict:
    """FID and IS for the same DDPM at several DDIM step counts.

    Both are computed over the full `--n` images, so they are directly
    comparable with the headline numbers in `results.json`.
    """
    import metrics

    diffusion = ddpm_module.load_ddpm(RUNS / "ddpm", device)
    inception = metrics.Inception().to(device)
    real_ds = reference_dataset(args.data_root, 64, args.n)
    print("extracting features from real images")
    real_features, _ = metrics.extract_features(real_ds, inception, device,
                                                max_images=args.n, model_space=True)

    out = {}
    scratch = HERE / "runs" / "_sweep"
    for steps in args.steps:
        if scratch.exists():
            shutil.rmtree(scratch)
        t0 = time.perf_counter()
        ddpm_module.generate(RUNS / "ddpm", device, args.n, out_dir=scratch,
                             steps=steps)
        elapsed = time.perf_counter() - t0

        f, lg = metrics.extract_features(scratch, inception, device,
                                         max_images=args.n)
        fid = metrics.compute_fid(real_features, f)
        is_mean, is_std = metrics.compute_inception_score(lg)
        out[steps] = {"fid": fid, "is_mean": is_mean, "is_std": is_std,
                      "generate_seconds": elapsed, "num_images": args.n}
        print(f"\nDDIM {steps:4d} steps:  FID {fid:6.2f}   IS {is_mean:.2f} "
              f"+/- {is_std:.2f}   ({elapsed / 60:.1f} min for {args.n})\n")

    if scratch.exists():
        shutil.rmtree(scratch)
    path = HERE / "sweep.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def figures(args, device) -> None:
    """Write every figure."""
    vae = vae_module.load_vae(RUNS / "vae", device)
    diffusion = ddpm_module.load_ddpm(RUNS / "ddpm", device)

    real_ds = CelebA(args.data_root, 64, indices=list(range(args.n_real)), flip=False)
    real_batch = torch.stack([real_ds[i] for i in range(16)])

    fig_comparison(vae, diffusion, real_batch, device)
    fig_vae_interpolation(vae, real_batch, device)
    fig_ddim_interpolation(diffusion, device)
    fig_trajectory(diffusion, device)
    fig_ddim_steps(diffusion, device)
    fig_training_progress()

    import metrics
    inception = metrics.Inception().to(device)
    features = {}
    print("inception features: real")
    features["real"], _ = metrics.extract_features(real_ds, inception, device,
                                                   model_space=True)
    for model in ("vae", "ddpm"):
        print(f"inception features: {model} samples")
        features[f"{model}_gen"], _ = metrics.extract_features(
            RUNS / model / "generated", inception, device, max_images=10000)

    sims = fig_nearest_neighbours(features, real_ds, device)
    stats = fig_typical_atypical(features, device)
    stats["shown_sample_nn_similarity"] = [round(float(s), 4) for s in sims[:, 0]]
    (HERE / "runs" / "figures" / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main() -> None:
    """Parse arguments and run the requested stage."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["figures", "sweep", "all"])
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=10000, help="images per sweep point")
    p.add_argument("--n-real", type=int, default=20000,
                   help="real images searched for nearest neighbours")
    p.add_argument("--steps", type=int, nargs="+", default=[10, 25, 50])
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    if args.stage in ("sweep", "all"):
        sweep(args, device)
    if args.stage in ("figures", "all"):
        figures(args, device)


if __name__ == "__main__":
    main()
