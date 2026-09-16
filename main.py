"""
The single entry point for everything: training, sampling and evaluation.

    python main.py train-vae                 train the VAE
    python main.py train-ddpm                train the DDPM
    python main.py sample --model vae        write a grid of samples
    python main.py generate --model ddpm     write PNGs for FID
    python main.py evaluate --model vae      compute FID and IS
    python main.py reproduce --preset quick  do all of the above, in order

Run `python main.py <command> --help` for the flags of any one command.

Every command takes `--data-root`, `--device` and `--seed`. Paths default to
sensible places next to this file, so the short forms above work as written.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import ddpm as ddpm_module
import vae as vae_module
from data import make_loaders, reference_dataset, save_grid

HERE = Path(__file__).resolve().parent
# VAE-and-DDPM/ -> CySh/, which is where CelebA lives.
DEFAULT_DATA_ROOT = HERE.parent / "CelebA"
RUNS = HERE / "runs"
RESULTS = HERE / "results.json"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Choose a compute device.

    Args:
        prefer: "auto", "cuda" or "cpu".

    Returns:
        The selected `torch.device`. "auto" picks CUDA when it is available.
    """
    if prefer == "auto":
        prefer = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(prefer)


def record_result(label: str, values: dict) -> None:
    """Merge one model's metrics into `results.json`.

    Args:
        label: Key for this model, e.g. "vae" or "ddpm".
        values: The metrics dict to store under that key.
    """
    all_results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    all_results[label] = values
    RESULTS.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {RESULTS}")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_train_vae(args: argparse.Namespace) -> None:
    """Train the VAE.

    Args:
        args: Parsed command-line arguments.
    """
    device = get_device(args.device)
    set_seed(args.seed)
    train_loader, val_loader = make_loaders(
        args.data_root, args.image_size, args.batch_size, args.limit,
        num_workers=args.num_workers, seed=args.seed)
    vae_module.train_vae(train_loader, val_loader, device, RUNS / "vae",
                         image_size=args.image_size, latent_dim=args.latent_dim,
                         epochs=args.epochs, lr=args.lr)


def cmd_train_ddpm(args: argparse.Namespace) -> None:
    """Train the DDPM.

    Args:
        args: Parsed command-line arguments.
    """
    device = get_device(args.device)
    set_seed(args.seed)
    # The DDPM has no validation loss worth watching -- the noise-prediction MSE
    # on held-out data tracks the training loss almost exactly -- so we use the
    # whole set for training and judge quality by FID instead.
    train_loader, _ = make_loaders(
        args.data_root, args.image_size, args.batch_size, args.limit,
        val_fraction=0.001, num_workers=args.num_workers, seed=args.seed)
    ddpm_module.train_ddpm(train_loader, device, RUNS / "ddpm",
                           image_size=args.image_size, base_channels=args.base_channels,
                           timesteps=args.timesteps, epochs=args.epochs, lr=args.lr)


def cmd_sample(args: argparse.Namespace) -> None:
    """Write a grid of samples (and, for the VAE, a reconstruction grid).

    Args:
        args: Parsed command-line arguments.
    """
    device = get_device(args.device)
    set_seed(args.seed)
    run_dir = RUNS / args.model
    figures = run_dir / "figures"

    if args.model == "vae":
        model = vae_module.load_vae(run_dir, device)
        print(save_grid(model.sample(args.n, device), figures / "samples.png"))

        # Real images on top, their reconstructions below: the clearest single
        # picture of what the VAE has and has not learned.
        _, val_loader = make_loaders(args.data_root, model.image_size,
                                     batch_size=32, limit=args.limit,
                                     num_workers=0, seed=args.seed)
        real = next(iter(val_loader))[:32].to(device)
        pair = torch.cat([real, model.reconstruct(real)])
        print(save_grid(pair, figures / "reconstructions.png"))
    else:
        diffusion = ddpm_module.load_ddpm(run_dir, device)
        size = diffusion.model.config["image_size"]
        images = diffusion.ddim_sample((args.n, 3, size, size), device,
                                       num_steps=args.steps)
        print(save_grid(images, figures / f"samples_ddim{args.steps}.png"))


def cmd_generate(args: argparse.Namespace) -> None:
    """Write `--n` PNGs for FID scoring.

    Args:
        args: Parsed command-line arguments.
    """
    device = get_device(args.device)
    set_seed(args.seed)
    run_dir = RUNS / args.model
    if args.model == "vae":
        vae_module.generate(run_dir, device, args.n)
    else:
        ddpm_module.generate(run_dir, device, args.n, steps=args.steps)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Compute FID and IS for a model's generated folder.

    Args:
        args: Parsed command-line arguments.
    """
    import metrics

    device = get_device(args.device)
    generated = RUNS / args.model / "generated"
    real = reference_dataset(args.data_root, args.image_size, args.n)

    results = metrics.evaluate(generated, real, device, args.n)
    print(f"\n{args.model}:  FID {results['fid']:.2f}   "
          f"IS {results['is_mean']:.2f} +/- {results['is_std']:.2f}   "
          f"({results['num_images']} images)")
    record_result(args.model, results)


def cmd_reproduce(args: argparse.Namespace) -> None:
    """Run the whole pipeline end to end.

    Args:
        args: Parsed command-line arguments.
    """
    # quick exists to prove the pipeline runs on your machine; standard is what
    # the numbers in REPORT.md come from.
    presets = {
        "quick":    dict(vae_epochs=5,  ddpm_epochs=5,  limit=20000, n_eval=2500,  steps=50),
        "standard": dict(vae_epochs=25, ddpm_epochs=20, limit=None,  n_eval=10000, steps=100),
    }
    # The U-Net's activations are much larger than the VAE's, so it needs a
    # smaller batch to fit in the same memory.
    batch_sizes = {"vae": 128, "ddpm": 32}
    p = presets[args.preset]
    print(f"preset {args.preset}: {p}\n")

    ############################################### EDIT EDIT EDIT
    # for stage, model in [("train", "vae"), ("train", "ddpm"),
    #                      ("sample", "vae"), ("sample", "ddpm"),
    #                      ("generate", "vae"), ("generate", "ddpm"),
    #                      ("evaluate", "vae"), ("evaluate", "ddpm")]:
    for stage, model in [("sample", "vae"),
                         ("generate", "vae"),
                         ("evaluate", "vae"),]:
        print(f"\n{'=' * 70}\n{stage} {model}\n{'=' * 70}")
        sub = argparse.Namespace(**vars(args))
        sub.model = model
        sub.limit = p["limit"]
        sub.steps = p["steps"]
        sub.n = p["n_eval"] if stage in ("generate", "evaluate") else 64
        sub.epochs = p[f"{model}_epochs"]
        sub.batch_size = batch_sizes[model]

        if stage == "train":
            (cmd_train_vae if model == "vae" else cmd_train_ddpm)(sub)
        elif stage == "sample":
            cmd_sample(sub)
        elif stage == "generate":
            cmd_generate(sub)
        else:
            cmd_evaluate(sub)

    print(f"\nAll done. Metrics in {RESULTS}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured `ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add the flags every command accepts."""
        p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                       help="folder containing img_align_celeba/")
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--image-size", type=int, default=64)
        p.add_argument("--limit", type=int, default=None,
                       help="use only the first N images (for smoke tests)")
        p.add_argument("--num-workers", type=int, default=4)
        return p

    tv = shared(sub.add_parser("train-vae", help="train the VAE"))
    tv.add_argument("--epochs", type=int, default=25)
    tv.add_argument("--batch-size", type=int, default=128)
    tv.add_argument("--latent-dim", type=int, default=128)
    tv.add_argument("--lr", type=float, default=2e-4)
    tv.set_defaults(func=cmd_train_vae)

    td = shared(sub.add_parser("train-ddpm", help="train the DDPM"))
    td.add_argument("--epochs", type=int, default=20)
    # 32 rather than the VAE's 128: the U-Net activations are far larger, and
    # this is what fits comfortably in ~8GB at 64x64.
    td.add_argument("--batch-size", type=int, default=32)
    td.add_argument("--base-channels", type=int, default=64)
    td.add_argument("--timesteps", type=int, default=1000)
    td.add_argument("--lr", type=float, default=2e-4)
    td.set_defaults(func=cmd_train_ddpm)

    sa = shared(sub.add_parser("sample", help="write a grid of samples"))
    sa.add_argument("--model", required=True, choices=["vae", "ddpm"])
    sa.add_argument("--n", type=int, default=64)
    sa.add_argument("--steps", type=int, default=50, help="DDIM steps (ddpm only)")
    sa.set_defaults(func=cmd_sample)

    ge = shared(sub.add_parser("generate", help="write PNGs for FID"))
    ge.add_argument("--model", required=True, choices=["vae", "ddpm"])
    ge.add_argument("--n", type=int, default=10000)
    ge.add_argument("--steps", type=int, default=50, help="DDIM steps (ddpm only)")
    ge.set_defaults(func=cmd_generate)

    ev = shared(sub.add_parser("evaluate", help="compute FID and IS"))
    ev.add_argument("--model", required=True, choices=["vae", "ddpm"])
    ev.add_argument("--n", type=int, default=10000)
    ev.set_defaults(func=cmd_evaluate)

    rp = shared(sub.add_parser("reproduce", help="run the whole pipeline"))
    rp.add_argument("--preset", default="quick", choices=["quick", "standard"])
    rp.add_argument("--latent-dim", type=int, default=128)
    rp.add_argument("--base-channels", type=int, default=64)
    rp.add_argument("--timesteps", type=int, default=1000)
    rp.add_argument("--lr", type=float, default=2e-4)
    rp.set_defaults(func=cmd_reproduce)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected command."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
