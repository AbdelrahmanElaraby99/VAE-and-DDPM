# Minimal VAE + DDPM on CelebA

A plain Variational Autoencoder and an unconditional Denoising Diffusion
Probabilistic Model, both written from scratch in PyTorch, trained on CelebA
faces at 64x64, and compared with FID and Inception Score.

"From scratch" means the models are built only out of primitive layers —
`Conv2d`, `Linear`, `GroupNorm`, activations. No VAE library, no diffusion
library, no pretrained generator. The only borrowed network is InceptionV3,
and only because FID and IS are *defined* as statistics of its activations, so
computing them any other way would mean computing a different number.

Everything runs through `main.py`.

---

## 1. Install

You need Python 3.9+ and a CUDA GPU (it will run on CPU, just slowly).

```bash
pip install -r requirements.txt
```

Six packages, listed in [`requirements.txt`](requirements.txt) with a comment
on each explaining why it is there:

| Package | What it is for |
|---|---|
| `torch` | every layer of both models |
| `torchvision` | InceptionV3 weights for FID/IS, and the image transforms |
| `numpy` | seeding, and the arrays around the FID covariance |
| `scipy` | one function — `linalg.sqrtm`, the matrix square root in FID |
| `pillow` | reading CelebA JPEGs, writing generated PNGs |
| `tqdm` | progress bars, and nothing else |

Only `tqdm` is cosmetic; removing it would leave the project functionally
identical. Note what is *absent*: no VAE library, no diffusion library, no FID
package. Those are the parts the task asks you to write, so they are written.

The versions in the file are floors — the oldest release with the API this code
calls — and the comment after each is the version the numbers in
[REPORT.md](REPORT.md) were produced with (Python 3.11.5, Windows 11, CUDA
12.6). The one floor worth knowing about is `torch>=2.3`, because the training
loops use `torch.amp.GradScaler("cuda")`, which replaced
`torch.cuda.amp.GradScaler` in that release.

If you need a particular CUDA build, install torch first and then the rest —
pip will leave the torch you already have in place:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Then put CelebA where the code expects it, or point at it with `--data-root`:

```
CySh/
├── CelebA/
│   └── img_align_celeba/       <- 202,599 .jpg files
└── VAE-and-DDPM/               <- this folder
```

Only the image folder is needed. The attribute file `list_attr_celeba.txt` is
not read, because both models here are unconditional.

## 2. Check it works

```bash
python test_minimal.py
```

23 checks, under a minute, CPU only. They verify the mathematics rather than
just the shapes — that the KL matches its closed form, that the diffusion
forward process preserves variance, that FID of a set against itself is zero,
and so on. Section 5 below lists what each one is guarding against.

Expected output ends with:

```
23 passed, 0 failed
```

## 3. Run everything

One command does all eight stages in order — train both models, write sample
grids, dump images, score them:

```bash
python main.py reproduce --preset quick
```

`quick` uses 20,000 images and 5 epochs each. It finishes in about 2 hours on
one GPU and produces recognisable but clearly under-trained faces. Use it first
to confirm the pipeline runs on your machine.

```bash
python main.py reproduce --preset standard
```

`standard` uses the full dataset, 25 VAE epochs and 20 DDPM epochs, and scores
10,000 images. About 20 hours. These are the settings the numbers in
[REPORT.md](REPORT.md) come from.

Results land in `results.json`, figures in `runs/<model>/figures/`.

## 4. Or run the stages yourself

```bash
# train
python main.py train-vae  --epochs 25
python main.py train-ddpm --epochs 20

# look at the results
python main.py sample --model vae
python main.py sample --model ddpm --steps 50

# score them
python main.py generate --model vae  --n 10000
python main.py generate --model ddpm --n 10000 --steps 50
python main.py evaluate --model vae  --n 10000
python main.py evaluate --model ddpm --n 10000
```

Add `--limit 5000 --epochs 1` to any training command for a fast trial run.
`python main.py <command> --help` lists every flag.

### Reading the progress bars

Every long loop shows a `tqdm` bar. The bar is erased when its stage finishes,
so the permanent log is still one summary line per epoch.

```
epoch   3: 45%|####      | 2841/6332 [03:12<03:56, 14.8it/s, ema=0.9992, loss=0.0284, lr=2.0e-04]
```

`loss` and the VAE's `recon`/`kl` are running averages over the epoch so far,
not single-batch values, so they settle instead of flickering. `ema` is the one
worth glancing at on the DDPM: it climbs from 0 towards 0.9999, and a run that
ends while it is still low has the exact failure described at the end of
section 5 — good loss, grey samples.

### Rough timings on one modern GPU

| Stage | quick | standard |
|---|---|---|
| VAE training | 3 min | 30 min |
| DDPM training | 40 min | 14 h |
| VAE generation (all images) | 10 s | 40 s |
| DDPM generation (all images) | 25 min | 4 h |
| FID + IS, both models | 8 min | 25 min |

The DDPM dominates everything. That is not an implementation problem — it is
the fundamental trade-off, and section 4 of the report is about it.

## 5. The files

| File | Lines | What is in it |
|---|---|---|
| [`data.py`](data.py) | 228 | CelebA dataset, train/val split, PNG writing |
| [`vae.py`](vae.py) | 508 | Encoder, decoder, ELBO, training loop, sampling |
| [`ddpm.py`](ddpm.py) | 1130 | Noise schedule, U-Net, diffusion math, EMA, training, DDIM |
| [`metrics.py`](metrics.py) | 395 | Inception features, FID, Inception Score |
| [`main.py`](main.py) | 304 | Command-line interface |
| [`qualitative.py`](qualitative.py) | 501 | The figures in section 5 of the report |
| [`test_minimal.py`](test_minimal.py) | 601 | 23 correctness checks |

Read them in that order. Each file opens with a docstring explaining the idea
before any code appears.

### What the tests check

| Area | Checks |
|---|---|
| VAE | shapes; reparameterisation has the right mean and variance and passes gradients; KL matches its closed form and is zero at the prior; reconstruction error is *summed* over pixels, not averaged |
| Schedule | betas increase, `alpha_bar` decreases to ~0; `sqrt(ab)² + sqrt(1-ab)² = 1`; the closed-form jump to step *t* matches simulating *t* steps; `predict_x_start` exactly inverts `q_sample`; the posterior mean collapses to `sqrt(alpha_bar_{t-1}) · x₀` |
| U-Net | output shape matches input; every skip connection is consumed; a zero-initialised network outputs exactly zero; GroupNorm does not cancel the timestep bias; the output actually changes with *t* |
| Sampling | DDIM at `eta=0` is deterministic and at `eta=1` is not; an untrained model scores ~1.0 on the noise-prediction loss |
| EMA | the warm-up flushes the random initialisation within 100 steps; the decay ramps monotonically to its ceiling without overshooting; `copy_to` works |
| Metrics | FID of a set against itself is 0; FID grows by exactly `D·s²` under a shift of `s`; IS is 1 for a collapsed set and *K* for a perfectly diverse one; a rank-deficient covariance raises a warning |

Two of these exist because of bugs that actually happened during development,
and both are the kind that a shape check would never catch:

- **The EMA warm-up.** A DDPM reached a noise-prediction MSE of 0.018 — a good
  number — and produced grey mush, because the EMA used a fixed decay of 0.9999
  and its shadow was still 33% random initialisation after 11,000 steps
  (`0.9999^11136 = 0.33`). Nothing in training reveals this, because training
  never reads the EMA.
- **GroupNorm group size.** The timestep enters `ResBlock` as a per-channel
  constant, and the next operation normalises. If a group holds exactly one
  channel, GroupNorm degenerates to InstanceNorm and subtracts precisely that
  constant. The conditioning is cancelled exactly, and the model trains happily
  while being completely time-blind.

## 6. If something goes wrong

**`FileNotFoundError: No .jpg files under ...`**
Point `--data-root` at the folder that *contains* `img_align_celeba/`, not at
`img_align_celeba/` itself.

**`CUDA out of memory` during DDPM training**
Lower `--batch-size` (32 is the default; 16 fits in about 4 GB), or
`--base-channels 32` to halve the network width.

**DDPM samples are grey mush but the loss looks fine**
Almost certainly the EMA. Check the warning `load_ddpm` prints at startup: if
it says a large percentage of the shadow is still the random initialisation,
the model has not trained long enough for the average to be meaningful. Train
longer, or call `load_ddpm(..., use_ema=False)` to sample from the raw weights.

**DDPM loss sits near 1.0 and will not fall**
A model that predicts zero scores exactly 1.0, because the regression target is
unit-variance noise. So 1.0 means nothing is being learned. Check the learning
rate, and that `--timesteps` is 1000 — the beta schedule constants are tuned
for that value, and a much smaller T leaves `x_T` far from pure noise.

**FID looks absurdly high (several hundred)**
Either the model is under-trained, or you scored too few images. FID needs more
samples than the 2,048 feature dimensions or the covariance is rank-deficient;
the code warns when this happens. 10,000 is the number worth quoting.

**The numbers do not match published FID values**
They will not exactly. Papers use the original TensorFlow Inception graph,
whose weights differ slightly from torchvision's port. Every model here is
scored with the same extractor, so comparisons *between* the two models are
valid — and that comparison is the point.
