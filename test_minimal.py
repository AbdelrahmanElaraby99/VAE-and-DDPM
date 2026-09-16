"""
Tests that check the mathematics, not just that the code runs.

    python test_minimal.py

Everything here runs on the CPU with tiny tensors, so it finishes in well under
a minute and never touches the GPU or downloads anything. The metric tests use
synthetic feature vectors rather than real Inception activations, so no network
access is needed either.

Each check verifies a property we can state exactly in advance. Where a formula
has a known closed form, we compare against it; where it has an exact algebraic
identity, we assert the identity. A test that only asserted "the shape is
right" would have passed for every bug we actually care about.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ddpm as D            # noqa: E402
import metrics as M         # noqa: E402
import vae as V             # noqa: E402

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #


def test_vae_shapes() -> None:
    """Every tensor in the VAE has the shape the maths requires."""
    model = V.VAE(image_size=32, latent_dim=8, channels=(8, 16))
    x = torch.randn(4, 3, 32, 32)

    mu, logvar = model.encoder(x)
    assert mu.shape == (4, 8), f"mu shape {mu.shape}"
    assert logvar.shape == (4, 8), f"logvar shape {logvar.shape}"

    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape, f"reconstruction shape {x_hat.shape}"

    # The decoder ends in tanh, so its output must lie inside [-1, 1] -- the
    # same range data.py normalises the data to.
    assert x_hat.min() >= -1.0 and x_hat.max() <= 1.0, "decoder escaped [-1, 1]"

    samples = model.sample(5, DEVICE)
    assert samples.shape == (5, 3, 32, 32), f"sample shape {samples.shape}"


def test_reparameterisation_moments() -> None:
    """z = mu + sigma*eps must have mean mu and standard deviation sigma."""
    torch.manual_seed(0)
    n = 200_000
    target_mu = torch.tensor([2.0, -3.0])
    target_std = torch.tensor([0.5, 2.0])
    logvar = torch.log(target_std ** 2)                     # logvar = log(sigma^2)

    z = V.VAE.reparameterize(target_mu.expand(n, 2), logvar.expand(n, 2))

    assert torch.allclose(z.mean(dim=0), target_mu, atol=0.02), \
        f"mean {z.mean(dim=0)} should be {target_mu}"
    assert torch.allclose(z.std(dim=0), target_std, atol=0.02), \
        f"std {z.std(dim=0)} should be {target_std}"

    # The gradient must reach mu -- that is the entire reason for the trick.
    mu = torch.zeros(1, 2, requires_grad=True)
    V.VAE.reparameterize(mu, torch.zeros(1, 2)).sum().backward()
    assert mu.grad is not None and torch.allclose(mu.grad, torch.ones(1, 2)), \
        "dz/dmu should be 1"


def test_kl_closed_form() -> None:
    """The KL term matches its closed form, and is zero at the prior."""
    images = torch.zeros(7, 3, 8, 8)        # irrelevant here; only the KL is checked

    # A posterior that equals the prior (mu = 0, sigma = 1) must have KL of 0.
    _, _, kl = V.vae_loss(images, images, torch.zeros(7, 5), torch.zeros(7, 5))
    assert abs(kl.item()) < 1e-6, f"KL at the prior should be 0, got {kl.item()}"

    # A general case, checked against 0.5*(sigma^2 + mu^2 - 1 - log sigma^2)
    # computed independently with numpy.
    torch.manual_seed(0)
    mu = torch.randn(7, 4)
    logvar = torch.randn(7, 4) * 0.5
    _, _, kl = V.vae_loss(images, images, mu, logvar)

    m, lv = mu.numpy(), logvar.numpy()
    expected = (0.5 * (np.exp(lv) + m ** 2 - 1.0 - lv)).sum(axis=1).mean()
    assert abs(kl.item() - expected) < 1e-5, f"KL {kl.item()} vs numpy {expected}"


def test_recon_is_summed_over_pixels() -> None:
    """Reconstruction error is summed over pixels, not averaged.

    This is the single most consequential scaling choice in the VAE loss. If it
    were averaged, the term would be 12288x smaller than it should be, the KL
    would dominate completely, and the model would collapse to the dataset mean.
    """
    x = torch.zeros(2, 3, 8, 8)
    x_hat = torch.ones(2, 3, 8, 8)          # squared error of 1 at every pixel
    _, recon, _ = V.vae_loss(x_hat, x, torch.zeros(2, 4), torch.zeros(2, 4))

    n_pixels = 3 * 8 * 8
    assert abs(recon.item() - n_pixels) < 1e-4, \
        f"expected {n_pixels} (summed), got {recon.item()}"


# --------------------------------------------------------------------------- #
# DDPM schedule
# --------------------------------------------------------------------------- #


def test_schedule_properties() -> None:
    """The schedule destroys information monotonically and completely."""
    s = D.build_schedule(1000)

    betas, ab = s["betas"], s["alphas_bar"]
    assert torch.all(betas[1:] > betas[:-1]), "betas must increase with t"
    assert torch.all(ab[1:] < ab[:-1]), "alphas_bar must decrease with t"
    assert ab[0] > 0.999, f"alpha_bar_0 should be ~1, got {ab[0]}"
    # By the final step almost none of the original signal is left, which is
    # what lets us start generation from pure noise.
    assert ab[-1] < 1e-3, f"alpha_bar_T should be ~0, got {ab[-1]}"


def test_variance_preserved() -> None:
    """sqrt(ab)^2 + sqrt(1-ab)^2 = 1 at every timestep.

    This identity is why x_t keeps unit variance throughout the forward process
    for unit-variance data -- and therefore why the images must be scaled to
    [-1, 1] rather than [0, 1].
    """
    s = D.build_schedule(1000)
    total = s["sqrt_alphas_bar"] ** 2 + s["sqrt_one_minus_alphas_bar"] ** 2
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
        f"max deviation {(total - 1).abs().max().item()}"


def test_q_sample_matches_step_by_step() -> None:
    """The closed-form jump to step t agrees with simulating t steps.

    The closed form is what makes training cheap, so it had better be the same
    process. We compare the two in distribution: run the one-step recursion
    many times and check the resulting variance against sqrt(1 - alpha_bar_t).
    """
    torch.manual_seed(0)
    timesteps, target_t = 50, 20
    s = D.build_schedule(timesteps)
    betas = D.linear_beta_schedule(timesteps).float()

    x0 = torch.ones(4000, 1, 1, 1)              # a constant "image"
    # Simulate the forward process one step at a time.
    x = x0.clone()
    for i in range(target_t + 1):
        x = torch.sqrt(1 - betas[i]) * x + torch.sqrt(betas[i]) * torch.randn_like(x)

    expected_mean = s["sqrt_alphas_bar"][target_t].item()
    expected_std = s["sqrt_one_minus_alphas_bar"][target_t].item()
    assert abs(x.mean().item() - expected_mean) < 0.02, \
        f"mean {x.mean().item():.4f} vs closed form {expected_mean:.4f}"
    assert abs(x.std().item() - expected_std) < 0.02, \
        f"std {x.std().item():.4f} vs closed form {expected_std:.4f}"


def test_predict_x_start_inverts_q_sample() -> None:
    """Given the true noise, predict_x_start recovers x_0 exactly."""
    torch.manual_seed(0)
    diffusion = D.Diffusion(_tiny_unet(), timesteps=100)

    x0 = torch.randn(6, 3, 8, 8)
    t = torch.randint(0, 100, (6,))
    noise = torch.randn_like(x0)

    x_t = diffusion.q_sample(x0, t, noise)
    recovered = diffusion.predict_x_start(x_t, t, noise)
    assert torch.allclose(recovered, x0, atol=1e-4), \
        f"max error {(recovered - x0).abs().max().item()}"


def test_posterior_mean_identity() -> None:
    """The posterior mean is exactly sqrt(alpha_bar_{t-1}) * x_0 at the noiseless x_t.

    Substituting x_t = sqrt(alpha_bar_t) * x_0 into

        mean = coef1 * x_0 + coef2 * x_t

    and using alpha_bar_t = alpha_t * alpha_bar_{t-1} collapses the whole
    expression to sqrt(alpha_bar_{t-1}) * x_0. If either coefficient is wrong
    this identity breaks, so it pins both of them at once.
    """
    timesteps = 200
    s = D.build_schedule(timesteps)
    betas = D.linear_beta_schedule(timesteps).float()
    alphas_bar = s["alphas_bar"]
    alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

    x0 = 1.0
    for t in [1, 17, 99, 199]:
        x_t = torch.sqrt(alphas_bar[t]) * x0
        mean = s["posterior_mean_coef1"][t] * x0 + s["posterior_mean_coef2"][t] * x_t
        expected = torch.sqrt(alphas_bar_prev[t])
        assert abs(mean.item() - expected.item()) < 1e-5, \
            f"t={t}: posterior mean {mean.item():.6f} vs {expected.item():.6f}"
        assert betas[t] > 0


# --------------------------------------------------------------------------- #
# U-Net and sampling
# --------------------------------------------------------------------------- #


def _tiny_unet(perturb: bool = False) -> D.UNet:
    """A U-Net small enough to run instantly on the CPU.

    Args:
        perturb: Give the zero-initialised convolutions random weights, which
            is what one optimizer step would do. Needed by any test that looks
            at what the network computes, because at strict initialisation it
            computes zero (see `test_unet_zero_init_outputs_zero`).

    Returns:
        A `UNet` at 16x16 with two levels and one residual block each.
    """
    net = D.UNet(image_size=16, base_channels=16, channel_mults=(1, 2),
                 num_res_blocks=1, attention_resolutions=(8,), dropout=0.0)
    if perturb:
        for module in net.modules():
            # The zero-initialised convs are exactly the ones we need to wake up:
            # ResBlock.conv2, SelfAttention.proj and the output head.
            if isinstance(module, torch.nn.Conv2d) and module.weight.abs().sum() == 0:
                torch.nn.init.normal_(module.weight, std=0.1)
    return net


def test_unet_zero_init_outputs_zero() -> None:
    """A freshly built U-Net predicts exactly zero noise.

    This is deliberate. `ResBlock.conv2`, `SelfAttention.proj` and the output
    convolution are all zero-initialised, which makes every residual block the
    identity at step 0. Deep residual stacks train far more stably from that
    starting point than from random noise, and it means the very first training
    step sees a neutral prediction rather than garbage.

    The consequence worth knowing: the timestep pathway is also dead at
    initialisation, because the time embedding is added *before* `conv2`. Any
    test of what the network computes has to perturb those weights first.
    """
    net = _tiny_unet()
    out = net(torch.randn(2, 3, 16, 16), torch.randint(0, 1000, (2,)))
    assert out.abs().max().item() == 0.0, \
        f"zero-init network should output exactly 0, got {out.abs().max().item()}"


def test_unet_shape_and_skips() -> None:
    """The U-Net returns the shape it was given, and consumes every skip."""
    net = _tiny_unet()
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 1000, (2,))
    out = net(x, t)
    assert out.shape == x.shape, f"U-Net returned {out.shape}, expected {x.shape}"

    # The decoder pops one skip per block; if the counts did not match, the
    # forward pass above would already have raised. Assert it explicitly by
    # checking the bookkeeping list is drained.
    skips = []
    h = net.stem(x)
    skips.append(h)
    t_emb = net.time_mlp(t)
    for level in net.down_levels:
        h = level(h, t_emb, skips)
    pushed = len(skips)
    h = net.mid_block2(net.mid_attn(net.mid_block1(h, t_emb)), t_emb)
    for level in net.up_levels:
        h = level(h, t_emb, skips)
    assert len(skips) == 0, f"{len(skips)} of {pushed} skips were never consumed"


def test_unet_depends_on_timestep() -> None:
    """The same image at two timesteps must produce different predictions.

    This is the whole point of the time embedding: one set of weights has to
    behave differently at different noise levels. If the embedding were dropped
    or added in the wrong place, the model would still train to a plausible-
    looking loss but could never denoise properly.
    """
    torch.manual_seed(0)
    net = _tiny_unet(perturb=True)
    net.eval()

    x = torch.randn(1, 3, 16, 16)
    a = net(x, torch.tensor([5]))
    b = net(x, torch.tensor([900]))
    relative = ((a - b).abs().mean() / a.abs().mean()).item()
    # A relative change of ~1e-6 would mean the signal is being cancelled and
    # only floating-point noise survives; see `test_group_norm_preserves_time`.
    assert relative > 1e-3, f"output barely changed with t: relative {relative:.2e}"


def test_group_norm_preserves_time() -> None:
    """Every GroupNorm must hold at least 2 channels per group.

    With one channel per group, GroupNorm becomes InstanceNorm and subtracts
    exactly the per-channel constant that `ResBlock` just added as the time
    embedding. The conditioning is cancelled to within floating-point noise and
    the model becomes time-blind while still reporting a falling loss -- a
    failure that no shape check would catch.
    """
    for channels in [16, 32, 64, 128, 256, 512]:
        gn = D.group_norm(channels)
        per_group = channels // gn.num_groups
        assert channels % gn.num_groups == 0, \
            f"{gn.num_groups} groups does not divide {channels} channels"
        assert per_group >= 2, \
            f"{channels} channels -> {per_group} channel(s) per group; time bias would cancel"

    # Demonstrate the failure directly: normalising a single-channel group
    # removes a constant shift entirely, while a two-channel group keeps it.
    x = torch.randn(1, 2, 4, 4)
    shift = torch.tensor([1.0, -1.0]).view(1, 2, 1, 1)
    one_per_group = torch.nn.GroupNorm(2, 2)
    two_per_group = torch.nn.GroupNorm(1, 2)
    assert torch.allclose(one_per_group(x), one_per_group(x + shift), atol=1e-5), \
        "1 channel per group should cancel a per-channel shift"
    assert not torch.allclose(two_per_group(x), two_per_group(x + shift), atol=1e-3), \
        "2 channels per group should preserve a per-channel shift"


def test_ddim_eta_zero_is_deterministic() -> None:
    """With eta = 0 the sampler is a deterministic function of the start noise.

    That determinism is the property that makes DDIM latents interpolatable,
    and it is easy to break by leaving a stray `randn` in the update.
    """
    torch.manual_seed(0)
    net = _tiny_unet(perturb=True)
    net.eval()
    diffusion = D.Diffusion(net, timesteps=50)

    torch.manual_seed(123)
    a = diffusion.ddim_sample((2, 3, 16, 16), DEVICE, num_steps=5, eta=0.0, progress=False)
    torch.manual_seed(123)
    b = diffusion.ddim_sample((2, 3, 16, 16), DEVICE, num_steps=5, eta=0.0, progress=False)
    assert torch.allclose(a, b, atol=1e-6), \
        f"eta=0 was not deterministic, max diff {(a - b).abs().max().item()}"

    # eta > 0 injects noise, so the two runs must now differ.
    torch.manual_seed(123)
    c = diffusion.ddim_sample((2, 3, 16, 16), DEVICE, num_steps=5, eta=1.0, progress=False)
    assert not torch.allclose(a, c, atol=1e-6), "eta=1 should add noise"


def test_ddpm_loss_is_about_one_at_init() -> None:
    """An untrained network scores ~1.0, because the target is unit noise.

    This is the reference point that makes the training loss readable: 1.0 is
    "predicting nothing", and a healthy CelebA run reaches 0.02-0.03.
    """
    torch.manual_seed(0)
    diffusion = D.Diffusion(_tiny_unet(), timesteps=100)
    x = torch.randn(64, 3, 16, 16)
    loss = diffusion.loss(x).item()
    assert 0.9 < loss < 1.1, f"untrained loss should be ~1.0, got {loss}"


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #


def test_ema_warmup_forgets_initialisation() -> None:
    """The warm-up must flush the random initialisation out within ~100 steps.

    Without the ramp, a decay of 0.9999 leaves 0.9999^N of the initialisation in
    the shadow: 99% after 100 steps, 33% after 11,000. Sampling from weights
    that are one third random noise produces grey mush while the training loss
    looks perfect, because training never reads the shadow.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    initial = {k: v.clone() for k, v in model.named_parameters()}

    ema = D.EMA(model, decay=0.9999)
    # Drive the model to a completely different set of weights and hold it there.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(10.0)
    for _ in range(100):
        ema.update(model)

    # After 100 warmed-up updates the shadow should have essentially converged
    # on the target, with no measurable trace of the initialisation.
    residual = max((ema.shadow[k] - 10.0).abs().max().item() for k in ema.shadow)
    assert residual < 1e-3, f"shadow still {residual} away from the target"
    assert any(v.abs().max() > 0 for v in initial.values())

    # And confirm the failure mode is real: a FIXED decay would still be
    # overwhelmingly the initialisation at this point.
    assert 0.9999 ** 100 > 0.98, "sanity: fixed decay retains ~99% after 100 steps"


def test_ema_decay_ramps_to_ceiling() -> None:
    """current_decay() rises from 0.1 toward the configured ceiling."""
    model = torch.nn.Linear(2, 2)
    ema = D.EMA(model, decay=0.99)

    assert abs(ema.current_decay() - 0.1) < 1e-9, "step 0 should give 1/10"
    for _ in range(10_000):
        ema.num_updates += 1
    assert abs(ema.current_decay() - 0.99) < 1e-9, "should reach the ceiling"

    # And it must never exceed the ceiling on the way.
    ema.num_updates = 0
    seen = []
    for _ in range(50):
        seen.append(ema.current_decay())
        ema.num_updates += 1
    assert all(d <= 0.99 + 1e-12 for d in seen), "decay overshot its ceiling"
    assert seen == sorted(seen), "decay should increase monotonically"


def test_ema_copy_to_replaces_weights() -> None:
    """copy_to writes the shadow into a model, which is how sampling uses it."""
    model = torch.nn.Linear(3, 3)
    ema = D.EMA(model)
    with torch.no_grad():
        for p in ema.shadow.values():
            p.fill_(7.0)

    ema.copy_to(model)
    for p in model.parameters():
        assert torch.allclose(p, torch.full_like(p, 7.0)), "copy_to did not take effect"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_fid_of_identical_sets_is_zero() -> None:
    """FID between a set and itself must be 0: the Gaussians are identical."""
    torch.manual_seed(0)
    features = torch.randn(200, 32)
    fid = M.compute_fid(features, features.clone())
    assert abs(fid) < 1e-4, f"FID against itself should be 0, got {fid}"


def test_fid_grows_with_distance() -> None:
    """Shifting one set further away must strictly increase the FID.

    The mean term of the Frechet distance is ||mu1 - mu2||^2, so a shift of s
    across D dimensions should add exactly D * s^2.
    """
    torch.manual_seed(0)
    a = torch.randn(500, 16)
    near = M.compute_fid(a, a + 0.5)
    far = M.compute_fid(a, a + 1.0)
    assert far > near > 0, f"expected 0 < {near} < {far}"

    # The increase should match D*s^2 = 16*1 = 16 for the unit shift, since the
    # covariances are identical and contribute nothing.
    assert abs(far - 16.0) < 0.5, f"unit shift should add ~16, got {far}"


def test_inception_score_bounds() -> None:
    """IS is 1 for a collapsed set and K for a perfectly diverse, confident one.

    These are the two extremes of the definition, and they pin the formula:
      * every image classified the same way -> p(y|x) = p(y) -> KL = 0 -> IS = 1
      * images spread evenly over K classes, each certain -> IS = K
    """
    n_classes, n = 10, 100

    # Collapsed: every image is confidently class 3.
    logits = torch.full((n, n_classes), -20.0)
    logits[:, 3] = 20.0
    mean, _ = M.compute_inception_score(logits, splits=2)
    assert abs(mean - 1.0) < 0.01, f"collapsed set should score 1.0, got {mean}"

    # Diverse: image i is confidently class i % 10.
    logits = torch.full((n, n_classes), -20.0)
    for i in range(n):
        logits[i, i % n_classes] = 20.0
    mean, _ = M.compute_inception_score(logits, splits=2)
    assert abs(mean - n_classes) < 0.1, \
        f"perfectly diverse set should score {n_classes}, got {mean}"


def test_fid_warns_on_too_few_samples() -> None:
    """A rank-deficient covariance must produce a loud warning, not a silent number."""
    import warnings
    features = torch.randn(10, 64)          # 10 samples, 64 dimensions
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        M.gaussian_statistics(features)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
        "expected a RuntimeWarning about rank deficiency"


# --------------------------------------------------------------------------- #
# Data (skipped if CelebA is not present)
# --------------------------------------------------------------------------- #


def test_dataset_if_available() -> None:
    """CelebA images load at the right size, range and split determinism."""
    from main import DEFAULT_DATA_ROOT
    import data

    if not (Path(DEFAULT_DATA_ROOT) / "img_align_celeba").exists():
        print("      (CelebA not found -- skipped)")
        return

    ds = data.CelebA(DEFAULT_DATA_ROOT, image_size=32, indices=list(range(4)), flip=False)
    img = ds[0]
    assert img.shape == (3, 32, 32), f"image shape {img.shape}"
    assert img.min() >= -1.0 and img.max() <= 1.0, "image outside [-1, 1]"

    # The same seed must produce the same split, or validation losses are not
    # comparable between runs.
    a, _ = data.make_loaders(DEFAULT_DATA_ROOT, 32, 4, limit=100, num_workers=0, seed=7)
    b, _ = data.make_loaders(DEFAULT_DATA_ROOT, 32, 4, limit=100, num_workers=0, seed=7)
    assert a.dataset.paths == b.dataset.paths, "split is not reproducible"

    # And view space must be the [0, 1] range PNG writing expects.
    view = data.to_view(img)
    assert view.min() >= 0.0 and view.max() <= 1.0, "to_view left [0, 1]"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


ALL_TESTS = [
    ("VAE: tensor shapes", test_vae_shapes),
    ("VAE: reparameterisation moments", test_reparameterisation_moments),
    ("VAE: KL closed form", test_kl_closed_form),
    ("VAE: reconstruction summed over pixels", test_recon_is_summed_over_pixels),
    ("DDPM: schedule is monotone and complete", test_schedule_properties),
    ("DDPM: forward process preserves variance", test_variance_preserved),
    ("DDPM: closed form == step-by-step", test_q_sample_matches_step_by_step),
    ("DDPM: predict_x_start inverts q_sample", test_predict_x_start_inverts_q_sample),
    ("DDPM: posterior mean identity", test_posterior_mean_identity),
    ("UNet: shape preserved, skips balanced", test_unet_shape_and_skips),
    ("UNet: zero-init predicts exactly zero", test_unet_zero_init_outputs_zero),
    ("UNet: GroupNorm does not cancel the time bias", test_group_norm_preserves_time),
    ("UNet: output depends on t", test_unet_depends_on_timestep),
    ("DDIM: eta=0 deterministic, eta=1 not", test_ddim_eta_zero_is_deterministic),
    ("DDPM: untrained loss is ~1.0", test_ddpm_loss_is_about_one_at_init),
    ("EMA: warm-up forgets the initialisation", test_ema_warmup_forgets_initialisation),
    ("EMA: decay ramps to its ceiling", test_ema_decay_ramps_to_ceiling),
    ("EMA: copy_to replaces weights", test_ema_copy_to_replaces_weights),
    ("FID: identical sets score 0", test_fid_of_identical_sets_is_zero),
    ("FID: grows with distance, by D*s^2", test_fid_grows_with_distance),
    ("IS: bounds are 1 and K", test_inception_score_bounds),
    ("FID: warns when rank-deficient", test_fid_warns_on_too_few_samples),
    ("Data: CelebA loads correctly", test_dataset_if_available),
]


def main() -> int:
    """Run every check and print a summary.

    Returns:
        0 if everything passed, 1 otherwise (so CI can use the exit code).
    """
    torch.manual_seed(0)
    passed, failed = 0, []

    print(f"running {len(ALL_TESTS)} checks on {DEVICE}\n")
    for name, fn in ALL_TESTS:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:                       # noqa: BLE001 - report everything
            print(f"  FAIL  {name}")
            print(f"        {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            failed.append(name)

    print(f"\n{passed} passed, {len(failed)} failed")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
