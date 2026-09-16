# DDPM: architecture and mathematical reference

Technical reference for the denoising diffusion probabilistic model implemented in [ddpm.py](ddpm.py).
Every equation below is cross-referenced to the function that implements it.

The accompanying architecture diagram is [ddpm_architecture.svg](ddpm_architecture.svg).

All numerical values quoted in this document were measured from the implementation at its default
configuration: `image_size=64`, `base_channels=64`, `channel_mults=(1,2,2,4)`, `num_res_blocks=2`,
`attention_resolutions=(16,)`, `timesteps=1000`.

> LaTeX in this file renders in the VS Code Markdown preview (`Ctrl+Shift+V`), on GitHub, and in Obsidian.

**Contents**

1. [Overview](#1-overview)
2. [Notation](#2-notation)
3. [The forward process](#3-the-forward-process)
4. [The reverse process](#4-the-reverse-process)
5. [The training objective](#5-the-training-objective)
6. [The noise-prediction network](#6-the-noise-prediction-network)
7. [Ancestral sampling](#7-ancestral-sampling)
8. [DDIM sampling](#8-ddim-sampling)
9. [Training procedure](#9-training-procedure)
10. [Comparison with the VAE baseline](#10-comparison-with-the-vae-baseline)
11. [Summary of equations](#11-summary-of-equations)
12. [Design rationale](#12-design-rationale)

---

## 1. Overview

A diffusion model defines a **fixed** forward process that destroys an image by adding a small amount of
Gaussian noise at each of $T$ steps, and a **learned** reverse process that removes noise one step at a time.
The only trained component is a U-Net $\epsilon_\theta(x_t, t)$ that takes a noisy image and predicts the
noise that was added to it. Generation starts from pure Gaussian noise and applies the reverse process.

Three properties of the construction determine the shape of the implementation:

| Property | Consequence |
|---|---|
| Composing $t$ Gaussian steps yields a Gaussian with a closed form | Training samples any $t$ directly; the chain is never simulated during training |
| The variational bound reduces to a noise-prediction MSE | A single loss term, with no adversarial objective and no learned posterior |
| The posterior $q(x_{t-1} \mid x_t, x_0)$ is Gaussian with a known closed form | The reverse variance is fixed rather than learned; only the mean is parameterised |

---

## 2. Notation

| Symbol | Meaning | Code |
|--------|---------|------|
| $x_0$ | Clean image, scaled to $[-1,1]$ | `x_start` |
| $x_t$ | Image at noise level $t$ | `x_noisy`, `x` |
| $T$ | Number of diffusion steps (1000) | `timesteps` |
| $\beta_t$ | Noise variance added at step $t$ | `betas` |
| $\alpha_t = 1-\beta_t$ | | `alphas` |
| $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ | Cumulative signal retention | `alphas_bar` |
| $\epsilon$ | Sampled noise, $\epsilon \sim \mathcal N(0,I)$ | `noise` |
| $\epsilon_\theta(x_t,t)$ | Network prediction of $\epsilon$ | `model(x_noisy, t)` |
| $\tilde\beta_t$ | Posterior variance | `posterior_variance` |

---

## 3. The forward process

The forward process has no trainable parameters.

### 3.1 Single step

$$
q(x_t \mid x_{t-1}) \;=\; \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)
$$

In reparameterised form:

$$
x_t \;=\; \sqrt{1-\beta_t}\;x_{t-1} \;+\; \sqrt{\beta_t}\;z ,\qquad z\sim\mathcal N(0,I)
$$

The $\sqrt{1-\beta_t}$ factor makes the process **variance-preserving**. If $\mathrm{Var}[x_{t-1}]=1$ then

$$
\mathrm{Var}[x_t] = (1-\beta_t)\cdot 1 + \beta_t = 1 .
$$

Without it the variance would grow without bound. The schedule therefore assumes input of approximately
unit variance, which is why images are scaled to $[-1,1]$ in `data.py`.

### 3.2 Noise schedule

$$
\beta_t = \text{linspace}(10^{-4},\ 0.02,\ T)
$$

Implemented in `linear_beta_schedule()`. The schedule is computed in float64: $\bar\alpha_t$ is a cumulative
product of 1000 terms, and float32 rounding at that stage produces visible artefacts in generated samples.
`build_schedule()` returns the derived constants cast back to float32.

### 3.3 Closed form for arbitrary $t$

Composing $t$ steps yields another Gaussian:

$$
q(x_t \mid x_0) \;=\; \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)
$$

$$
x_t \;=\; \sqrt{\bar\alpha_t}\,x_0 \;+\; \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
$$

Implemented in `Diffusion.q_sample()`. This result is what makes training tractable: the cost of producing a
training pair is independent of $t$.

<details>
<summary>Derivation</summary>

Expand two steps and induct. Starting from

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,z_t,
\qquad
x_{t-1} = \sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,z_{t-1}
$$

substitution gives

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} \;+\; \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,z_{t-1} + \sqrt{1-\alpha_t}\,z_t}_{\text{two independent Gaussians}}
$$

The sum of independent zero-mean Gaussians is Gaussian with the variances added:

$$
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) \;=\; \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t \;=\; 1 - \alpha_t\alpha_{t-1}
$$

so $x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar z$. Induction over $t$
gives the stated result with $\bar\alpha_t=\prod_s \alpha_s$. $\blacksquare$
</details>

The two coefficients satisfy $(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$, so signal and noise
trade off on a unit circle and $\mathrm{Var}[x_t] = 1$ holds at every $t$.

### 3.4 Schedule values

Measured from `build_schedule(1000)`:

| $t$ | $\beta_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ (signal) | $\sqrt{1-\bar\alpha_t}$ (noise) | $\sigma_t$ (posterior std) |
|----:|----------:|---------------:|-------------------------------:|--------------------------------:|---------------------------:|
| 0   | 0.000100 | 0.999900 | 0.9999 | 0.0100 | 0.0074 |
| 50  | 0.001096 | 0.969952 | 0.9849 | 0.1733 | 0.0325 |
| 100 | 0.002092 | 0.895142 | 0.9461 | 0.3238 | 0.0453 |
| 250 | 0.005080 | 0.521423 | 0.7221 | 0.6918 | 0.0711 |
| 500 | 0.010060 | 0.077797 | 0.2789 | 0.9603 | 0.1003 |
| 750 | 0.015040 | 0.003300 | 0.0574 | 0.9983 | 0.1226 |
| 900 | 0.018028 | 0.000270 | 0.0164 | 0.9999 | 0.1343 |
| 999 | 0.020000 | 0.000040 | 0.0064 | 1.0000 | 0.1414 |

Although $\beta_t$ is linear in $t$, the cumulative effect is strongly non-linear. By $t=500$ only 28% of the
signal amplitude remains, and by $t=750$ the sample is essentially noise; most of the perceptually
significant corruption occurs within the first few hundred steps.

At the endpoint, $\bar\alpha_T = 4\times10^{-5}$, so $x_T \approx \mathcal N(0,I)$. This is a requirement
rather than an incidental property: generation begins by sampling from $\mathcal N(0,I)$, so the schedule
must drive the forward process close enough to that distribution. The constants $10^{-4}$ and $0.02$ are
tuned for $T=1000$ and do not transfer to substantially smaller $T$.

---

## 4. The reverse process

### 4.1 Parameterisation

The target is $p_\theta(x_{t-1}\mid x_t)$. For sufficiently small $\beta_t$ the reverse conditional is also
approximately Gaussian, so it is parameterised as

$$
p_\theta(x_{t-1}\mid x_t) \;=\; \mathcal N\!\left(x_{t-1};\ \mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t)\right) .
$$

$q(x_{t-1}\mid x_t)$ on its own is intractable, since evaluating it would require the full data
distribution. Conditioned on $x_0$, however, it has a closed form.

### 4.2 Posterior conditioned on $x_0$

$$
q(x_{t-1}\mid x_t, x_0) \;=\; \mathcal N\!\left(x_{t-1};\ \tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\right)
$$

$$
\tilde\mu_t(x_t,x_0) \;=\; \underbrace{\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}}_{c_1(t)}\,x_0 \;+\; \underbrace{\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}}_{c_2(t)}\,x_t
$$

$$
\tilde\beta_t \;=\; \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

Implemented in `build_schedule()` as `posterior_mean_coef1` ($c_1$), `posterior_mean_coef2` ($c_2$) and
`posterior_variance` ($\tilde\beta_t$).

<details>
<summary>Derivation</summary>

By Bayes' rule,

$$
q(x_{t-1}\mid x_t,x_0) = q(x_t\mid x_{t-1})\,\frac{q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)},
$$

where all three factors are known Gaussians. Writing out the exponents:

$$
\propto \exp\left(-\tfrac12\left[\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{\beta_t} + \frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0)^2}{1-\bar\alpha_{t-1}} - \frac{(x_t-\sqrt{\bar\alpha_t}x_0)^2}{1-\bar\alpha_t}\right]\right)
$$

Collecting powers of $x_{t-1}$, the quadratic coefficient gives the precision:

$$
\frac{1}{\tilde\beta_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}
\;=\;\frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})}
\;=\;\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}
$$

using $\alpha_t - \bar\alpha_t + 1 - \alpha_t = 1-\bar\alpha_t$. Inverting gives $\tilde\beta_t$; the linear
coefficient divided by the precision gives $\tilde\mu_t$. $\blacksquare$
</details>

**Numerical note.** $\tilde\beta_0 = 0$ exactly, since conditioning on $x_0$ leaves no uncertainty at $t=0$.
Because $\log 0 = -\infty$ would propagate through the sampler, `build_schedule()` stores
`posterior_log_variance` with the $t{=}0$ entry replaced by the $t{=}1$ entry.

### 4.3 Reparameterising the mean through $\epsilon$

$\tilde\mu_t$ depends on $x_0$, which is unavailable at sampling time. Inverting the forward closed form
gives an estimate from the network output:

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

Implemented in `Diffusion.predict_x_start()`. Predicting the noise and predicting the clean image are
therefore equivalent up to this rearrangement. Substituting $\hat x_0$ into $\tilde\mu_t$ and simplifying
recovers the form given in Ho et al. (2020):

$$
\mu_\theta(x_t,t) \;=\; \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)
$$

This implementation uses the two-stage $\hat x_0$ route rather than the single-line expression above,
because it exposes $\hat x_0$ for the clamping step described in §7.

The $\epsilon$-parameterisation is preferred empirically. The target $\epsilon$ has unit variance at every
$t$, so the regression problem is equally well-scaled across the schedule. Direct $x_0$ prediction is
poorly conditioned at high $t$, where little information about $x_0$ remains in $x_t$.

---

## 5. The training objective

### 5.1 Variational bound

$$
-\log p_\theta(x_0) \;\le\; \mathbb{E}_q\!\left[\underbrace{D_{\mathrm{KL}}(q(x_T|x_0)\,\|\,p(x_T))}_{L_T}
+ \sum_{t>1}\underbrace{D_{\mathrm{KL}}\!\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)}_{L_{t-1}}
- \underbrace{\log p_\theta(x_0|x_1)}_{L_0}\right]
$$

$L_T$ contains no trainable parameters: the forward process is fixed and $q(x_T|x_0)\approx\mathcal N(0,I)$
by construction, so the term is constant and can be dropped. $L_{t-1}$ is a KL divergence between two
Gaussians with the same fixed variance, which admits a closed form.

### 5.2 Reduction to a squared error

For two Gaussians with equal variance,

$$
D_{\mathrm{KL}}\big(\mathcal N(\tilde\mu,\sigma^2 I)\,\|\,\mathcal N(\mu_\theta,\sigma^2 I)\big) = \frac{\|\tilde\mu - \mu_\theta\|^2}{2\sigma^2},
$$

so the divergence reduces to a squared distance between means. Substituting the $\epsilon$-parameterisation
into both means, the $x_t$ terms cancel and the result is

$$
L_{t-1} \;=\; \frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\;\big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2 .
$$

### 5.3 Simplified objective

Ho et al. (2020) observe that discarding the $t$-dependent weight improves sample quality, as it
up-weights the high-$t$ steps that carry most of the structural work:

$$
L_{\text{simple}} \;=\; \mathbb{E}_{x_0,\;t\sim\mathcal U\{1..T\},\;\epsilon\sim\mathcal N(0,I)}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t\big)\big\|^2\Big]
$$

This is what `Diffusion.loss()` implements:

```python
t       = torch.randint(0, self.timesteps, (b,))   # one random t per image
noise   = torch.randn_like(x_start)                # regression target
x_noisy = self.q_sample(x_start, t, noise)         # jump directly to step t
return F.mse_loss(self.model(x_noisy, t), noise)
```

The maximum-likelihood problem has been reduced to supervised regression against synthetically generated
targets. There is no discriminator, no second loss term to balance, and no learned posterior.

**Reference values.** The target has unit variance, so a network that always outputs zero scores exactly
1.0. A converged CelebA run at this configuration settles around 0.02–0.03. A loss near 1.0 indicates the
network is not learning.

---

## 6. The noise-prediction network

$\epsilon_\theta$ is a U-Net. See panel B of [ddpm_architecture.svg](ddpm_architecture.svg) for the
corresponding diagram.

### 6.1 Choice of architecture

1. The network is shape-preserving: input and output are both $3\times64\times64$, since the output is a
   per-pixel noise map.
2. Skip connections carry high-frequency content around the bottleneck. Noise is entirely high-frequency,
   so a plain autoencoder bottleneck would discard the quantity being predicted.
3. A single conditioning vector can be broadcast into every block.

### 6.2 Shape walkthrough

| Stage | Channels | Resolution | Notes |
|-------|---------:|-----------:|-------|
| input | 3 | 64×64 | $x_t \in [-1,1]$ |
| stem Conv 3×3 | 64 | 64×64 | also pushed as a skip |
| DownLevel 0 | 64 | 64×64 → 32×32 | 2 × ResBlock, then stride-2 conv |
| DownLevel 1 | 128 | 32×32 → 16×16 | 2 × ResBlock, then stride-2 conv |
| DownLevel 2 | 128 | 16×16 → 8×8 | 2 × [ResBlock + self-attention] |
| DownLevel 3 | 256 | 8×8 | 2 × ResBlock, no downsample (last level) |
| Bottleneck | 256 | 8×8 | ResBlock → self-attention → ResBlock |
| UpLevel 3 | 256 | 8×8 → 16×16 | 3 × [concat skip + ResBlock] |
| UpLevel 2 | 128 | 16×16 → 32×32 | 3 × [concat + ResBlock + self-attention] |
| UpLevel 1 | 128 | 32×32 → 64×64 | 3 × [concat + ResBlock] |
| UpLevel 0 | 64 | 64×64 | 3 × [concat + ResBlock], no upsample |
| head | 3 | 64×64 | GroupNorm → SiLU → Conv 3×3 |

Skip tensors are managed as a stack (12 in total):

```
push:  [64 | 64, 64, 64 | 128, 128, 128 | 128, 128, 128 | 256, 256]
        stem   level 0     level 1         level 2        level 3

pop:   UpLevel 3 ← 256, 256, 128     UpLevel 2 ← 128, 128, 128
       UpLevel 1 ← 128, 128,  64     UpLevel 0 ←  64,  64,  64
```

Each decoder level contains `num_res_blocks + 1` = 3 blocks against 2 in the corresponding encoder level,
because the encoder also pushes its downsampler output.

Parameter distribution, measured:

| Component | Parameters | Share |
|-----------|-----------:|------:|
| time MLP | 0.08M | 0.5% |
| encoder | 4.13M | 24.0% |
| bottleneck | 2.76M | 16.0% |
| decoder | 10.25M | 59.5% |
| **total** | **17.22M** | |

The decoder accounts for the majority because each of its ResBlocks receives a concatenated input, roughly
doubling the width of its first convolution.

### 6.3 Timestep conditioning

A single set of weights must handle both low-$t$ inputs (remove faint grain) and high-$t$ inputs (recover
structure from near-noise), so the timestep has to be supplied explicitly. Feeding the raw integer is
ineffective: as a scalar it provides no features to build on, and its scale is inconsistent with the
non-linear effect of $t$ shown in §3.4.

The implementation uses the sinusoidal positional encoding of Vaswani et al. (2017), which has no
parameters:

$$
\text{emb}(t)_{i} = \sin\!\left(\frac{t}{10000^{\,i/(d/2)}}\right),\qquad
\text{emb}(t)_{d/2+i} = \cos\!\left(\frac{t}{10000^{\,i/(d/2)}}\right)
$$

Low-index components oscillate rapidly and distinguish adjacent timesteps; high-index components oscillate
slowly and encode coarse position. A learned MLP follows:

$$
t_{\text{emb}} = \text{Linear}_{256\to256}\big(\text{SiLU}(\text{Linear}_{64\to256}(\text{emb}(t)))\big)
$$

Inside each ResBlock the embedding is projected and added as a per-channel bias:

$$
h \;\leftarrow\; h \;+\; \text{Linear}_{256\to C'}(\text{SiLU}(t_{\text{emb}}))\big[:,:,\text{None},\text{None}\big]
$$

### 6.4 ResBlock

$$
\begin{aligned}
h &= \text{Conv}_{3\times3}\big(\text{SiLU}(\text{GN}(x))\big) \\
h &= h + \text{proj}(t_{\text{emb}}) \\
h &= \text{Conv}_{3\times3}\big(\text{Dropout}(\text{SiLU}(\text{GN}(h)))\big) \\
\text{out} &= h + \text{skip}(x)
\end{aligned}
$$

`skip` is `Identity` when the channel count is unchanged, and a 1×1 convolution otherwise.

### 6.5 Self-attention

Applied at 16×16 and at the 8×8 bottleneck only. The spatial grid is flattened into $N = HW$ tokens of
dimension $C$:

$$
Q,K,V = \text{Conv}_{1\times1}(\text{GN}(x)),\qquad
A = \text{softmax}\!\left(\frac{Q^\top K}{\sqrt{C}}\right)\in\mathbb R^{N\times N},\qquad
\text{out} = x + W_o(AV)
$$

Convolutions are local, so a purely convolutional network has no mechanism enforcing consistency between
distant regions of the image. Attention supplies that mechanism.

The attention matrix is $N \times N$, so cost scales as $O((HW)^2)$, or the fourth power of the side length.
At 16×16 the matrix has $65{,}536$ entries; at 64×64 it would have $1.68\times10^{7}$. This is the reason
attention is restricted to the lower resolutions.

### 6.6 Normalisation and initialisation constraints

**Zero-initialised output convolutions.** The second convolution of every ResBlock, the output projection
of every attention block, and the final output head are all zero-initialised. Each block therefore begins
as an exact identity, and the untrained network predicts $\epsilon = 0$. Deep residual stacks train more
stably under this initialisation.

**GroupNorm rather than BatchNorm.** Sampling frequently runs at small batch sizes, where BatchNorm's
batch statistics become unreliable. GroupNorm normalises within a single sample, so the result does not
depend on batch size.

**At least two channels per GroupNorm group.** `ResBlock` injects $t_{\text{emb}}$ as a per-channel
constant, and the next operation is a normalisation. If a group contains exactly one channel, GroupNorm
reduces to InstanceNorm and subtracts that channel's own mean, which is precisely the constant just added.
The conditioning is cancelled exactly and the model becomes insensitive to $t$. With two or more channels
per group only the group average of the shifts is removed, so inter-channel differences survive.

The failure mode is difficult to detect from training metrics, because training never evaluates sample
quality: the loss, validation loss and gradient norms all remain plausible while generated samples are
unusable. `group_norm()` caps the group count at `channels // 2` to prevent it. For the default widths
(64, 128, 256) this selects 32 groups either way, so the standard configuration is unaffected.

---

## 7. Ancestral sampling

Implemented in `Diffusion.p_sample_loop()`. This is the sampler defined by the reverse process in §4.

1. $x_T \sim \mathcal N(0,I)$
2. For $t = T-1,\dots,0$:
   1. $\hat\epsilon = \epsilon_\theta(x_t, t)$ — one full forward pass of the U-Net
   2. $\hat x_0 = \dfrac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}$, clamped to $[-1,1]$
   3. $\mu = c_1(t)\,\hat x_0 + c_2(t)\,x_t$
   4. $x_{t-1} = \mu + \sigma_t z$, with $z\sim\mathcal N(0,I)$ and $\sigma_t = \exp(\tfrac12\log\tilde\beta_t)$
   5. At $t=0$, return $\mu$ without adding noise
3. Return $x_0 \in [-1,1]$

**On the clamp in step 2.2.** At high $t$, $\sqrt{\bar\alpha_t}$ is small (0.0064 at $t=999$), so the
division amplifies any error in $\hat\epsilon$ by a factor of roughly 156. Without the clamp, early steps
can produce far out-of-range estimates of $x_0$ that compound over subsequent steps.

**On the fixed variance.** $\sigma_t^2 = \tilde\beta_t$ is taken directly from the schedule rather than
learned. This is the fixed-variance choice from the original paper; learning the reverse variance, as in
Improved DDPM, primarily benefits likelihood rather than sample quality.

**Cost.** $T = 1000$ sequential network evaluations per batch. This is the principal computational
limitation of the method, and the motivation for §8.

---

## 8. DDIM sampling

Implemented in `Diffusion.ddim_sample()`. Used for the per-epoch previews during training and for bulk
generation during evaluation.

Song et al. (2021) observe that the training objective constrains only the marginals $q(x_t\mid x_0)$.
Multiple reverse processes share those marginals, including non-Markovian ones that skip timesteps
entirely. A trained $\epsilon_\theta$ can therefore be reused with a shorter trajectory without retraining.

Given a subsequence $\tau_1 < \dots < \tau_S$ of $\{0,\dots,T-1\}$ (this implementation uses $S$ evenly
spaced indices, default 50), the update is

$$
x_{\tau_{i-1}} = \underbrace{\sqrt{\bar\alpha_{\tau_{i-1}}}\;\hat x_0}_{\text{predicted } x_0} \;+\; \underbrace{\sqrt{1-\bar\alpha_{\tau_{i-1}} - \sigma^2}\;\hat\epsilon}_{\text{direction toward } x_t} \;+\; \underbrace{\sigma\, z}_{\text{injected noise}}
$$

$$
\sigma \;=\; \eta\,\sqrt{\frac{1-\bar\alpha_{\tau_{i-1}}}{1-\bar\alpha_{\tau_i}}}\;\sqrt{1-\frac{\bar\alpha_{\tau_i}}{\bar\alpha_{\tau_{i-1}}}}
$$

| $\eta$ | Behaviour |
|-------:|-----------|
| 0 | $\sigma = 0$; the process is fully deterministic and a given initial noise always produces the same image |
| 1 | Recovers the stochastic ancestral update of §7 |

With $\sigma = 0$ the two coefficients satisfy
$(\sqrt{\bar\alpha_{\text{prev}}})^2 + (\sqrt{1-\bar\alpha_{\text{prev}}})^2 = 1$: the update is `q_sample`
re-applied at the previous noise level, substituting the model's $\hat x_0$ and $\hat\epsilon$ for the true
values.

The implementation applies `relu` under the square root in the direction term to guard against small
negative values arising from rounding, and returns $\hat x_0$ directly at the final step.

At 50 steps rather than 1000 this is a 20× reduction in sampling cost for a modest quality penalty, which
is what makes FID evaluation over thousands of images practical. $\eta = 0$ additionally gives
reproducible outputs for a fixed seed.

---

## 9. Training procedure

Implemented in `train_ddpm()`.

| Setting | Value | Rationale |
|---------|-------|-----------|
| Optimiser | Adam, lr $2\times10^{-4}$ | Standard for this model class |
| LR warm-up | Linear over 500 steps | Gradients in the first few hundred steps are large, as the network has no information with which to predict the noise |
| Gradient clipping | Global norm 1.0 | Gradients are unscaled before clipping so the threshold applies to true gradient magnitudes rather than loss-scaled ones |
| Mixed precision | fp16 autocast on CUDA | Throughput |
| Weight EMA | Decay ramping to 0.9999 | Sampling uses the EMA weights; see §9.1 |

### 9.1 Weight EMA

$$
\theta_{\text{EMA}} \leftarrow d\cdot\theta_{\text{EMA}} + (1-d)\cdot\theta
$$

Diffusion sample quality is sensitive to weight noise, and the final optimiser iterate produces visibly
worse samples than an average of recent iterates. Sampling is therefore performed from an EMA copy of the
weights, both for the per-epoch previews and in `load_ddpm()`.

**Initialisation contamination.** The shadow weights are initialised as a copy of the random
initialisation. Under a fixed decay of $d = 0.9999$, the fraction of that random initialisation still
present after $N$ updates is $0.9999^N$:

| Updates $N$ | Residual random initialisation |
|------------:|-------------------------------:|
| 1,000 | 90% |
| 11,000 | 33% |
| 30,000 | 5% |

A network is not a linear function of its weights, so retaining a substantial fraction of a random
initialisation does not degrade output gracefully — it destroys it. The condition is not visible in
training metrics, since training never reads the shadow weights.

**Mitigation.** The decay is ramped in from a low value, following ADM, `timm` and the Karras codebase:

$$
d_{\text{eff}}(N) \;=\; \min\!\left(d,\ \frac{1+N}{10+N}\right)
$$

At $N = 0$ this evaluates to 0.1, so the shadow is immediately dominated by trained weights, and the
residual initialisation decays as $N^{-9}$ rather than exponentially slowly. `EMA.current_decay()`
implements this, and the value is reported in the training progress bar.

Note that the problem scales inversely with run length: a 300,000-step run is unaffected
($0.9999^{300000} \approx 0$), while a short run is not. `load_ddpm()` recomputes the contamination
fraction from the stored update count and emits a warning above 1%.

---

## 10. Comparison with the VAE baseline

Both models in this repository are trained on the same data at the same resolution.

| | VAE | DDPM |
|---|---|---|
| Latent | Learned, low-dimensional ($z\in\mathbb R^{128}$) | Fixed, same shape as the image |
| Encoder | Learned $q_\phi(z\mid x)$ | None; the forward process is fixed and parameter-free |
| Objective | ELBO: reconstruction + $\beta\cdot$KL (two competing terms) | Single MSE term |
| Characteristic failure | Posterior collapse; blurred samples | Sampling cost |
| Generation cost | 1 forward pass | 1000 forward passes, or 50 with DDIM |
| Likelihood | Lower bound | Lower bound (tighter) |
| FID (2500 images) | 222.9 | 71.7 |
| Inception Score | 2.58 ± 0.10 | 2.28 ± 0.07 |

The blurring characteristic of the VAE follows from its objective and its single-step decoder: the
Gaussian reconstruction likelihood is minimised by predicting the conditional mean over all plausible
images, and an average over faces is a blur. The diffusion model never performs that mapping in one step.
Each reverse step removes a small amount of noise, and the noise injected at each step causes the
trajectory to commit to a single sample rather than averaging across modes.

Inception Score is marginally higher for the VAE despite a substantially worse FID. IS measures class
diversity under an ImageNet classifier, which is not meaningful on a single-category dataset such as
CelebA; FID is the appropriate metric for this comparison.

---

## 11. Summary of equations

$$
\textbf{Forward (fixed):}\qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
$$

$$
\textbf{Objective:}\qquad L = \mathbb E_{x_0,t,\epsilon}\big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2
$$

$$
\textbf{Inversion:}\qquad \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}
$$

$$
\textbf{Reverse step:}\qquad x_{t-1} = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\hat x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \sigma_t z
$$

$$
\textbf{DDIM step:}\qquad x_{\text{prev}} = \sqrt{\bar\alpha_{\text{prev}}}\,\hat x_0 + \sqrt{1-\bar\alpha_{\text{prev}}-\sigma^2}\,\hat\epsilon + \sigma z
$$

| Equation | Implementation |
|---|---|
| $\beta_t$, $\bar\alpha_t$, $\tilde\beta_t$, $c_1$, $c_2$ | `build_schedule()` |
| $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | `Diffusion.q_sample()` |
| $\hat x_0$ from $\hat\epsilon$ | `Diffusion.predict_x_start()` |
| $L_{\text{simple}}$ | `Diffusion.loss()` |
| Ancestral reverse loop | `Diffusion.p_sample_loop()` |
| DDIM loop | `Diffusion.ddim_sample()` |
| $\epsilon_\theta$ | `UNet.forward()` |

---

## 12. Design rationale

Notes on decisions that are not evident from the code itself.

**Noise prediction rather than image prediction.** The two are algebraically equivalent (§4.3), but the
noise target has unit variance at every $t$, so the regression is equally well-conditioned across the
whole schedule. Direct $x_0$ prediction is badly conditioned at high $t$.

**Training stability relative to GANs.** The objective is supervised regression against targets generated
during the training step itself. There is no min-max optimisation, no discriminator to keep in balance,
and no mode collapse.

**Choice of $T = 1000$.** Two requirements bound it from opposite directions: $\beta_t$ must be small
enough that the reverse conditional is approximately Gaussian, and $\bar\alpha_T$ must be small enough that
$x_T$ is close to $\mathcal N(0,I)$. With the linear schedule and a substantially smaller $T$, the
endpoint does not reach pure noise and generation begins from the wrong distribution.

**Source of stochasticity in a sample.** The initial $x_T$, plus the $\sigma_t z$ term injected at each of
the reverse steps. Under DDIM with $\eta = 0$ the only source is $x_T$.

**Attention placement.** Cost scales as $O((HW)^2)$, so attention is applied at 16×16 and below. See §6.5.

**Known limitation and available mitigations.** Sampling cost is the main constraint. Options in the
literature are reduced-step samplers (DDIM, DPM-Solver), operating in a compressed latent space rather
than pixel space (latent diffusion), and distilling a multi-step sampler into a few-step or single-step
student.

---

## References

- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, J., Meng, C., Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR.
- Nichol, A., Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. (Sinusoidal positional encoding.)
