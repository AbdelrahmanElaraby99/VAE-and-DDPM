"""
The unconditional DDPM: noise schedule, U-Net, diffusion math, training, sampling.

Built only from primitive layers. No diffusion library is used anywhere.

THE IDEA
--------
A diffusion model defines a fixed FORWARD process that destroys an image by
adding a little Gaussian noise at each of T steps:

    q(x_t | x_{t-1}) = N( sqrt(1 - beta_t) * x_{t-1},  beta_t * I )

`beta_t` is a small number that grows with t. After T = 1000 steps the image is
indistinguishable from pure noise. Then we train a network to undo one step at
a time, and generate by starting from noise and running the reverse process.

THE ONE FORMULA THAT MAKES IT CHEAP
-----------------------------------
Composing t Gaussian steps gives another Gaussian, with a closed form:

    q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I )

    where  alpha_t = 1 - beta_t   and   alpha_bar_t = prod_{s<=t} alpha_s

so we can jump straight from a clean image to its state at step 700 with one
multiply-add. Training never simulates the chain; it samples a random t and
uses this formula. That is why DDPM training is stable and parallel while
sampling is slow and sequential.

Note that the two coefficients satisfy a^2 + b^2 = 1. That keeps the variance
of x_t at 1 throughout, provided the data has unit variance -- which is exactly
why the images are scaled to [-1, 1] in data.py.

THE TRAINING OBJECTIVE
----------------------
The full variational bound on log p(x) simplifies (Ho et al. 2020, eq. 14) to a
plain regression: look at a noisy image and guess which noise was added.

    L = E_{x_0, t, eps}  || eps  -  eps_theta(x_t, t) ||^2

No adversarial game, no posterior to learn, no balancing of two terms. This is
the practical reason diffusion models train more reliably than GANs, and the
main structural difference from the VAE, which optimises a two-term ELBO.

A useful number: a network that always outputs zero scores exactly 1.0 on this
loss, since the target is unit-variance noise. A healthy CelebA run settles
around 0.02-0.03. If your loss is near 1.0, nothing is being learned.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import save_grid


# --------------------------------------------------------------------------- #
# 1. The noise schedule
# --------------------------------------------------------------------------- #


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    """Betas spaced linearly, the schedule from the original DDPM paper.

    These constants were tuned for T = 1000. With a much smaller T the final
    x_T would not be close enough to pure noise, and generation would start
    from the wrong distribution.

    Args:
        timesteps: Number of diffusion steps T.
        beta_start: Noise variance added at t = 0.
        beta_end: Noise variance added at t = T-1.

    Returns:
        Tensor of shape (T,) in float64. The precision matters: `alphas_bar` is
        a cumulative product of 1000 terms and float32 rounding there is
        visible in the samples.
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def build_schedule(timesteps: int = 1000) -> Dict[str, torch.Tensor]:
    """Precompute every schedule-derived constant the model needs.

    All of these are fixed once T is chosen, so we compute them at start-up and
    the training loop and sampler only ever index into them.

    The posterior q(x_{t-1} | x_t, x_0) is the quantity the reverse model is
    trained to match. Conditioned on the clean image it is Gaussian with

        variance = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        mean     = coef1 * x_0  +  coef2 * x_t

    We reuse this exact variance at sampling time rather than learning it,
    which is the "fixed small" choice from the original paper.

    Args:
        timesteps: Number of diffusion steps T.

    Returns:
        A dict of float32 tensors, each of shape (T,), keyed by name.
    """
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    # alpha_bar_{t-1}, with alpha_bar_{-1} defined as 1: no noise before t=0.
    alphas_bar_prev = torch.cat([torch.ones(1, dtype=betas.dtype), alphas_bar[:-1]])

    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    # posterior_variance[0] is exactly 0 -- once you condition on x_0 there is
    # nothing left to be uncertain about -- and log(0) = -inf would poison the
    # arithmetic, so we substitute the t=1 entry.
    posterior_log_variance = torch.log(
        torch.cat([posterior_variance[1:2], posterior_variance[1:]]))

    out = {
        "betas": betas,
        "alphas_bar": alphas_bar,
        "sqrt_alphas_bar": torch.sqrt(alphas_bar),
        "sqrt_one_minus_alphas_bar": torch.sqrt(1.0 - alphas_bar),
        "posterior_log_variance": posterior_log_variance,
        "posterior_mean_coef1": betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar),
        "posterior_mean_coef2": (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar),
    }
    return {k: v.to(torch.float32) for k, v in out.items()}


def extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Pick `values[t[i]]` per batch element and reshape it for broadcasting.

    Every image in a training batch sits at a different timestep, so we need a
    different schedule constant for each one, shaped so it multiplies an image.

    Example:
        values has shape (1000,), t = [5, 900, 12], shape = (3, 3, 64, 64)
        -> result has shape (3, 1, 1, 1), which broadcasts over C, H and W.

    Args:
        values: A 1-D schedule tensor of length T.
        t: Integer tensor of shape (B,) with a timestep per batch element.
        shape: Shape of the image tensor this result will multiply.

    Returns:
        Tensor of shape (B, 1, 1, ...) with `len(shape)` dimensions.
    """
    out = values.to(t.device).gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


# --------------------------------------------------------------------------- #
# 2. The noise-prediction network (U-Net)
# --------------------------------------------------------------------------- #
#
# Why a U-Net and not any other architecture:
#   1. Input and output have the same shape -- an image in, a noise map out.
#   2. Skip connections carry high-frequency detail around the bottleneck. Noise
#      is entirely high-frequency, so a plain autoencoder would destroy exactly
#      the thing we are asking the network to predict.
#   3. The timestep is easy to inject into every block.


class SinusoidalTimeEmbedding(nn.Module):
    """Turn an integer timestep into a smooth, high-dimensional vector.

    Feeding the raw integer t to a convolution does not work: as a single
    scalar it gives the network nothing to build features from, and the scale
    is wrong (t=1 and t=2 differ by as much as the smallest representable step
    while t=1 and t=999 differ by a thousand of them).

    Instead we use the Transformer positional-encoding construction:

        emb(t)[i]        = sin( t / 10000^(i/half) )
        emb(t)[half + i] = cos( t / 10000^(i/half) )

    Low-index entries oscillate quickly and distinguish neighbouring timesteps;
    high-index entries oscillate slowly and encode coarse position. It has no
    parameters -- the MLP that follows does the learning.
    """

    def __init__(self, dim: int) -> None:
        """Store the embedding width.

        Args:
            dim: Output width; must be even so sin and cos pair up.
        """
        super().__init__()
        assert dim % 2 == 0, "embedding dim must be even (sin/cos pairs)"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of timesteps.

        Args:
            t: Integer tensor of shape (B,) with values in [0, T).

        Returns:
            Float tensor of shape (B, dim).
        """
        half = self.dim // 2
        # Frequencies spaced geometrically from 1 down to 1/10000, computed in
        # log space for numerical stability.
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)          # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


def group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a group count that divides `channels`, at least 2 per group.

    GroupNorm rather than BatchNorm throughout the diffusion network, because
    sampling often runs with very small batches where BatchNorm's batch
    statistics become unreliable. GroupNorm normalises within a single sample,
    so batch size never affects the result.

    WHY AT LEAST TWO CHANNELS PER GROUP. `ResBlock` injects the timestep as a
    per-channel constant, and the very next operation is this normalisation. If
    a group holds exactly one channel, GroupNorm degenerates into InstanceNorm:
    it subtracts that channel's own mean, which is precisely the constant the
    time embedding just added. The conditioning is cancelled exactly, and the
    model silently becomes time-blind -- it still trains, still reports a
    falling loss, and can never denoise properly.

    With two or more channels per group only the group *average* of the shifts
    is removed, so the differences between channels survive and carry the
    signal. Capping the group count at `channels // 2` guarantees that. For the
    default widths (64, 128, 256) this picks 32 groups either way, so nothing
    about the standard configuration changes.

    Args:
        channels: Number of channels to normalise.
        groups: Preferred group count, reduced until it divides evenly.

    Returns:
        A configured `nn.GroupNorm` with at least 2 channels per group
        (except for the degenerate case of a single-channel input).
    """
    g = min(groups, max(1, channels // 2))
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    """Two convolutions with a residual connection, conditioned on the timestep.

    Data flow:

        h = conv1( silu( norm( x ) ) )
        h = h + time_projection(t_emb)          <- conditioning enters here
        h = conv2( dropout( silu( norm( h ) ) ) )
        out = h + skip(x)

    Adding the time embedding as a per-channel bias is the simplest effective
    conditioning: every channel is shifted by an amount that depends on t, so
    the same weights can behave differently at different noise levels.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 dropout: float = 0.1) -> None:
        """Build the block.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
            time_dim: Width of the time embedding.
            dropout: Dropout probability before the second convolution.
        """
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        # Projects (B, time_dim) down to (B, out_ch) so it can be broadcast.
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # A 1x1 conv on the residual path when the channel count changes,
        # otherwise the shapes will not add.
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Zero-initialising the last conv makes the whole block the identity at
        # step 0. Deep residual stacks train far more stably that way; it is a
        # standard trick in diffusion implementations.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Feature map of shape (B, in_ch, H, W).
            t_emb: Time embedding of shape (B, time_dim).

        Returns:
            Feature map of shape (B, out_ch, H, W).
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # (B, out_ch) -> (B, out_ch, 1, 1) so it broadcasts over H and W.
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over the pixels of a feature map.

    Convolutions only see a local neighbourhood, so nothing in a purely
    convolutional network enforces agreement between distant parts of the image.
    Attention lets every pixel look at every other pixel, which is what keeps
    global structure coherent -- two eyes that match, a symmetric face.

    The attention matrix is (H*W) x (H*W), so the cost grows with the fourth
    power of the side length. We therefore only use it at 16x16 and below.
    """

    def __init__(self, channels: int) -> None:
        """Build the attention block.

        Args:
            channels: Channel count of the feature map.
        """
        super().__init__()
        self.norm = group_norm(channels)
        # One 1x1 convolution produces query, key and value together.
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        # Zero-init the output projection so the block starts as the identity.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention with a residual connection.

        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Feature map of the same shape.
        """
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        # Flatten the spatial grid into a sequence of N = H*W tokens.
        q, k, v = qkv.reshape(b, 3, c, h * w).unbind(dim=1)      # each (B, C, N)

        # Scaled dot product: how much does pixel i attend to pixel j?
        # The 1/sqrt(C) factor stops the logits growing with channel count.
        attn = torch.einsum("bci,bcj->bij", q, k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)                              # rows sum to 1

        out = torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Halve the resolution with a strided convolution.

    Preferred over average pooling because a strided convolution can *learn*
    which information is worth keeping on the way down.
    """

    def __init__(self, channels: int) -> None:
        """Args:
            channels: Channel count, unchanged by this operation.
        """
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by 2x.

        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Feature map (B, C, H/2, W/2).
        """
        return self.op(x)


class Upsample(nn.Module):
    """Double the resolution: nearest-neighbour interpolation, then a 3x3 conv.

    Interpolate-then-convolve rather than `ConvTranspose2d`, because transposed
    convolutions leave checkerboard artefacts that are very visible in
    generated images.
    """

    def __init__(self, channels: int) -> None:
        """Args:
            channels: Channel count, unchanged by this operation.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2x.

        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Feature map (B, C, 2H, 2W).
        """
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class DownLevel(nn.Module):
    """One resolution level of the encoder: residual blocks, then a downsample.

    Every intermediate feature map is pushed onto the `skips` list so the
    decoder can concatenate it later.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, n_blocks: int,
                 use_attn: bool, downsample: bool, dropout: float) -> None:
        """Build the level.

        Args:
            in_ch: Channels entering this level.
            out_ch: Channels produced by this level.
            time_dim: Width of the time embedding.
            n_blocks: Residual blocks at this resolution.
            use_attn: Insert self-attention after each block.
            downsample: Halve the resolution at the end of the level.
            dropout: Dropout probability inside the residual blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        ch = in_ch
        for _ in range(n_blocks):
            self.blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
            self.attns.append(SelfAttention(out_ch) if use_attn else nn.Identity())
            ch = out_ch
        self.downsample = Downsample(ch) if downsample else None

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        """Run the level, appending to `skips` in place.

        Args:
            h: Incoming feature map (B, in_ch, H, W).
            t_emb: Time embedding (B, time_dim).
            skips: List that receives every intermediate feature map.

        Returns:
            The outgoing feature map.
        """
        for block, attn in zip(self.blocks, self.attns):
            h = attn(block(h, t_emb))
            skips.append(h)
        if self.downsample is not None:
            h = self.downsample(h)
            skips.append(h)
        return h


class UpLevel(nn.Module):
    """One resolution level of the decoder: pop a skip, concatenate, convolve.

    There is one more block here than in the matching `DownLevel`, because the
    encoder pushed one extra feature map (the downsampler output).
    """

    def __init__(self, in_ch: int, out_ch: int, skip_chs: List[int], time_dim: int,
                 use_attn: bool, upsample: bool, dropout: float) -> None:
        """Build the level.

        Args:
            in_ch: Channels entering this level.
            out_ch: Channels produced by this level.
            skip_chs: Channel count of each skip this level will consume, in
                the order they will be popped.
            time_dim: Width of the time embedding.
            use_attn: Insert self-attention after each block.
            upsample: Double the resolution at the end of the level.
            dropout: Dropout probability inside the residual blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        ch = in_ch
        for skip_ch in skip_chs:
            # ch + skip_ch because the skip is concatenated along channels.
            self.blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
            self.attns.append(SelfAttention(out_ch) if use_attn else nn.Identity())
            ch = out_ch
        self.upsample = Upsample(ch) if upsample else None

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        """Run the level, popping from `skips` in place.

        Args:
            h: Incoming feature map.
            t_emb: Time embedding (B, time_dim).
            skips: Stack of encoder feature maps; popped from the end.

        Returns:
            The outgoing feature map.
        """
        for block, attn in zip(self.blocks, self.attns):
            h = torch.cat([h, skips.pop()], dim=1)
            h = attn(block(h, t_emb))
        if self.upsample is not None:
            h = self.upsample(h)
        return h


class UNet(nn.Module):
    """Predicts the noise that was added to `x` at timestep `t`.

    Shape walkthrough at 64x64 with base=64 and mults=(1, 2, 2, 4):

        input    3 x 64 x 64
        stem    64 x 64 x 64
        level 0  64 x 64 x 64  -> downsample ->  64 x 32 x 32
        level 1 128 x 32 x 32  -> downsample -> 128 x 16 x 16
        level 2 128 x 16 x 16  (+ attention)  -> downsample -> 128 x 8 x 8
        level 3 256 x  8 x  8
        middle  256 x  8 x  8  (res -> attention -> res)
        ... mirrored back up, concatenating skips ...
        output   3 x 64 x 64

    Example:
        >>> net = UNet(image_size=64)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (2,))
        >>> net(x, t).shape
        torch.Size([2, 3, 64, 64])
    """

    def __init__(self, image_size: int = 64, base_channels: int = 64,
                 channel_mults: Sequence[int] = (1, 2, 2, 4), num_res_blocks: int = 2,
                 attention_resolutions: Sequence[int] = (16,), dropout: float = 0.1) -> None:
        """Assemble the encoder, bottleneck and decoder.

        Args:
            image_size: Input/output side length.
            base_channels: Channel width of the first level; all others scale off it.
            channel_mults: Channel multiplier per resolution level.
            num_res_blocks: Residual blocks per level.
            attention_resolutions: Spatial sizes at which to use self-attention.
            dropout: Dropout probability inside residual blocks.
        """
        super().__init__()
        self.config = {"image_size": image_size, "base_channels": base_channels,
                       "channel_mults": list(channel_mults),
                       "num_res_blocks": num_res_blocks,
                       "attention_resolutions": list(attention_resolutions),
                       "dropout": dropout}

        time_dim = base_channels * 4        # 4x base is the conventional width
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        # --- encoder ---------------------------------------------------------
        # `skip_chs` records the channel count of every tensor the encoder will
        # push, so the decoder can size its convolutions correctly.
        self.down_levels = nn.ModuleList()
        skip_chs: List[int] = [base_channels]           # the stem output is a skip too
        ch = base_channels
        resolution = image_size
        n_levels = len(channel_mults)

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            is_last = level == n_levels - 1
            self.down_levels.append(
                DownLevel(ch, out_ch, time_dim, num_res_blocks,
                          resolution in attention_resolutions, not is_last, dropout))
            skip_chs += [out_ch] * num_res_blocks
            ch = out_ch
            if not is_last:
                skip_chs.append(ch)                     # the downsampler output
                resolution //= 2

        # --- bottleneck -------------------------------------------------------
        # res -> attention -> res at the lowest resolution, where global mixing
        # is both cheapest and most valuable.
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # --- decoder -----------------------------------------------------------
        self.up_levels = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            # num_res_blocks + 1 skips per level: one extra for the downsampler
            # output the encoder pushed.
            consumed = [skip_chs.pop() for _ in range(num_res_blocks + 1)]
            self.up_levels.append(
                UpLevel(ch, out_ch, consumed, time_dim,
                        resolution in attention_resolutions, level != 0, dropout))
            ch = out_ch
            if level != 0:
                resolution *= 2

        # --- output head --------------------------------------------------------
        self.out_norm = group_norm(ch)
        self.out_conv = nn.Conv2d(ch, 3, kernel_size=3, padding=1)
        # Zero-init means the network predicts exactly zero noise before any
        # training, a neutral starting point rather than random garbage.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise in `x` at timestep `t`.

        Args:
            x: Noisy images of shape (B, 3, H, W).
            t: Timesteps of shape (B,), integer dtype, values in [0, T).

        Returns:
            Predicted noise of shape (B, 3, H, W) -- the same shape as `x`.
        """
        t_emb = self.time_mlp(t)

        h = self.stem(x)
        skips = [h]
        for level in self.down_levels:
            h = level(h, t_emb, skips)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for level in self.up_levels:
            h = level(h, t_emb, skips)

        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
# 3. The diffusion process
# --------------------------------------------------------------------------- #


class Diffusion(nn.Module):
    """Ties a noise-prediction network to a fixed schedule.

    It subclasses `nn.Module` only so the schedule tensors register as buffers
    and move with `.to(device)`. It owns no trainable parameters itself.
    """

    def __init__(self, model: nn.Module, timesteps: int = 1000) -> None:
        """Attach a schedule to a model.

        Args:
            model: The noise-prediction network, called as `model(x, t)`.
            timesteps: Number of diffusion steps T.
        """
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        for name, tensor in build_schedule(timesteps).items():
            # persistent=False: these are recomputed from T, not worth storing.
            self.register_buffer(name, tensor, persistent=False)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Jump straight to step t of the forward process.

            x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps

        Args:
            x_start: Clean images (B, 3, H, W) in [-1, 1].
            t: Timesteps (B,), integer, in [0, T).
            noise: Noise of the same shape as `x_start`. Passed in rather than
                drawn here because `loss` needs the identical tensor as both
                the corruption and the regression target.

        Returns:
            The noisy images x_t, shape (B, 3, H, W).
        """
        a = extract(self.sqrt_alphas_bar, t, x_start.shape)
        b = extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape)
        return a * x_start + b * noise

    def predict_x_start(self, x_t: torch.Tensor, t: torch.Tensor,
                        noise: torch.Tensor) -> torch.Tensor:
        """Invert `q_sample` to estimate the clean image.

        Rearranging x_t = sqrt(ab)*x_0 + sqrt(1-ab)*eps gives

            x_0 = ( x_t - sqrt(1 - alpha_bar_t) * eps ) / sqrt(alpha_bar_t)

        Args:
            x_t: Noisy images (B, 3, H, W).
            t: Timesteps (B,).
            noise: The network's predicted noise (B, 3, H, W).

        Returns:
            The implied clean image x_0.
        """
        a = extract(self.sqrt_alphas_bar, t, x_t.shape)
        b = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        return (x_t - b * noise) / a

    def loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """One training step: corrupt an image, then predict what was added.

        Args:
            x_start: Clean images (B, 3, H, W) in [-1, 1].

        Returns:
            A scalar MSE loss.
        """
        b = x_start.shape[0]
        # One uniformly random timestep per image. Uniform sampling is the
        # standard unweighted estimator of the expectation over t.
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        return F.mse_loss(self.model(x_noisy, t), noise)

    @torch.no_grad()
    def p_sample_loop(self, shape: Sequence[int], device: torch.device,
                      progress: bool = True) -> torch.Tensor:
        """Full ancestral sampling: pure noise -> image in T sequential steps.

        Each step:
            1. predict the noise in x_t,
            2. convert it to an estimate of x_0,
            3. clamp that estimate to [-1, 1] -- cheap, and a very effective
               stabiliser, since early steps can otherwise predict wildly
               out-of-range images that then amplify,
            4. compute the posterior mean and variance for x_{t-1},
            5. sample from it -- except at t = 0, where we return the mean,
               because adding noise to the finished image would only make it
               grainy.

        This is the most faithful sampler and the slowest: T network
        evaluations, 1000 by default.

        Args:
            shape: Output shape, e.g. (16, 3, 64, 64).
            device: Device to run on.
            progress: Show a tqdm bar. False for the per-epoch preview, which
                would otherwise interleave with the training bar.

        Returns:
            Images of shape `shape` in [-1, 1].
        """
        x = torch.randn(*shape, device=device)
        for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps,
                      desc="DDPM sampling", leave=False, disable=not progress):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = self.model(x, t)
            x_start = self.predict_x_start(x, t, eps).clamp(-1.0, 1.0)

            mean = (extract(self.posterior_mean_coef1, t, x.shape) * x_start
                    + extract(self.posterior_mean_coef2, t, x.shape) * x)
            if i == 0:
                x = mean
            else:
                log_var = extract(self.posterior_log_variance, t, x.shape)
                # exp(0.5 * log_var) is the standard deviation.
                x = mean + torch.exp(0.5 * log_var) * torch.randn_like(x)

        return x

    @torch.no_grad()
    def ddim_sample(self, shape: Sequence[int], device: torch.device,
                    num_steps: int = 50, eta: float = 0.0,
                    progress: bool = True) -> torch.Tensor:
        """Fast sampling: the same trained model, far fewer steps.

        DDIM (Song et al. 2021) observes that the training objective only ever
        constrains the marginals q(x_t | x_0). Many reverse processes share
        those marginals, including a deterministic one that may skip timesteps
        entirely. The update is

            x_prev = sqrt(ab_prev) * x_0_hat
                   + sqrt(1 - ab_prev - sigma^2) * eps
                   + sigma * noise

            sigma = eta * sqrt((1 - ab_prev)/(1 - ab)) * sqrt(1 - ab/ab_prev)

        eta = 0 makes sigma zero, so the process is fully deterministic and the
        same starting noise always gives the same image. eta = 1 recovers the
        stochastic ancestral update. 50 steps instead of 1000 is a 20x speedup
        for a small quality cost, which is what makes FID evaluation over
        thousands of images practical at all.

        Args:
            shape: Output shape, e.g. (16, 3, 64, 64).
            device: Device to run on.
            num_steps: How many of the T timesteps to actually visit.
            eta: 0.0 = deterministic DDIM, 1.0 = ancestral DDPM.
            progress: Show a tqdm bar. False for the per-epoch preview, which
                would otherwise interleave with the training bar.

        Returns:
            Images of shape `shape` in [-1, 1].
        """
        # Visit `num_steps` timesteps evenly spaced over [0, T), descending.
        times = torch.linspace(0, self.timesteps - 1, num_steps).long().tolist()
        times = list(reversed(times))
        # Pair each step with its successor; -1 marks the final step.
        pairs = list(zip(times, times[1:] + [-1]))

        x = torch.randn(*shape, device=device)
        for t_now, t_next in tqdm(pairs, desc=f"DDIM {num_steps} steps",
                                  leave=False, disable=not progress):
            t = torch.full((shape[0],), t_now, device=device, dtype=torch.long)
            eps = self.model(x, t)
            x_start = self.predict_x_start(x, t, eps).clamp(-1.0, 1.0)

            if t_next < 0:
                x = x_start                     # final step: return the estimate
                break

            ab = extract(self.alphas_bar, t, x.shape)
            t_prev = torch.full((shape[0],), t_next, device=device, dtype=torch.long)
            ab_prev = extract(self.alphas_bar, t_prev, x.shape)

            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab)) * torch.sqrt(1 - ab / ab_prev)
            # relu guards against a tiny negative under the sqrt from rounding.
            direction = torch.sqrt(torch.relu(1 - ab_prev - sigma ** 2)) * eps
            x = torch.sqrt(ab_prev) * x_start + direction
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        return x


# --------------------------------------------------------------------------- #
# 4. Exponential moving average of the weights
# --------------------------------------------------------------------------- #


class EMA:
    """A smoothed copy of the weights, used for sampling instead of the raw ones.

    Diffusion sample quality is unusually sensitive to weight noise: the final
    SGD iterate produces visibly worse images than an average of recent
    iterates. Every serious DDPM implementation samples from an EMA copy.

        shadow = decay * shadow + (1 - decay) * current

    THE WARM-UP IS NOT OPTIONAL. This is the single most common way to get a
    DDPM whose loss curve looks perfect and whose samples are grey mush. The
    shadow starts as a copy of the RANDOM INITIALISATION, and with a fixed
    decay of 0.9999 the fraction of that noise still present after N updates is
    0.9999^N -- 90% after 1,000 steps, 33% after 11,000, still 5% after 30,000.
    A network is not a linear function of its weights, so mixing in a third of
    a random initialisation does not blur the output, it destroys it.

    Nothing in training reveals this, because training never reads the shadow.
    The training loss and the gradient norms are both perfectly healthy while
    the samples you actually look at are ruined.

    The fix, used by ADM, `timm` and the Karras codebase, is to ramp the decay
    in from near zero:

        effective_decay = min(decay, (1 + step) / (10 + step))

    At step 0 that is 0.1, so the shadow is immediately dominated by real
    weights; the residual initialisation then falls off as N^-9 instead of
    exponentially slowly.

    Note the bug scales the wrong way, which is why it survives in published
    configs: a 300,000-step run is immune (0.9999^300000 is effectively zero),
    while a short run is destroyed.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """Snapshot the model's current weights as the starting shadow.

        Args:
            model: The model whose weights will be averaged.
            decay: The ceiling decay rate, reached once warm-up completes.
        """
        self.decay = decay
        self.num_updates = 0
        self.shadow = {name: p.detach().clone()
                       for name, p in model.named_parameters() if p.requires_grad}

    def current_decay(self) -> float:
        """The decay actually applied at this step, including warm-up.

        Returns:
            `min(decay, (1 + step) / (10 + step))`.
        """
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model's current weights into the shadow.

        Args:
            model: The model being trained.
        """
        d = self.current_decay()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite a model's weights with the shadow, for sampling.

        Args:
            model: The model to write into (usually a throwaway copy).
        """
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])


# --------------------------------------------------------------------------- #
# 5. Training and generation
# --------------------------------------------------------------------------- #


def train_ddpm(train_loader, device: torch.device, out_dir: str | Path,
               image_size: int = 64, base_channels: int = 64, timesteps: int = 1000,
               epochs: int = 15, lr: float = 2e-4, warmup_steps: int = 500,
               grad_clip: float = 1.0, ema_decay: float = 0.9999,
               amp: bool = True, preview_steps: int = 50) -> Path:
    """Train the DDPM and write checkpoints, previews and a loss history.

    Args:
        train_loader: DataLoader over training images in [-1, 1].
        device: Device to train on.
        out_dir: Run folder; created if missing.
        image_size: Image side length.
        base_channels: U-Net base width.
        timesteps: Number of diffusion steps T.
        epochs: Passes over the training set.
        lr: Adam learning rate, after warm-up.
        warmup_steps: Linear learning-rate ramp at the start. Diffusion training
            is unstable in the first few hundred steps without it.
        grad_clip: Clip gradients to this global norm; 0 disables.
        ema_decay: Ceiling decay for the weight EMA.
        amp: Use float16 autocast on CUDA.
        preview_steps: DDIM steps for the per-epoch preview grid.

    Returns:
        Path to the run folder.
    """
    import copy

    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    unet = UNet(image_size=image_size, base_channels=base_channels).to(device)
    diffusion = Diffusion(unet, timesteps).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    ema = EMA(unet, ema_decay)

    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"DDPM: {n_params / 1e6:.1f}M parameters, T={timesteps}, device={device}")

    history: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(epochs):
        unet.train()
        start = time.time()
        total, n_batches = 0.0, 0

        # leave=False erases the bar when the epoch finishes, so only the
        # one-line summary printed below survives in the log.
        bar = tqdm(train_loader, desc=f"epoch {epoch:3d}", leave=False)
        for x in bar:
            x = x.to(device, non_blocking=True)

            # Linear learning-rate warm-up. The first few hundred steps see
            # enormous gradients because the network is predicting noise it has
            # no information about yet.
            if global_step < warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = lr * (global_step + 1) / warmup_steps

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = diffusion.loss(x)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                # Unscale first, or the clip threshold would apply to the
                # loss-scaled gradients and mean nothing.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(unet)

            total += loss.item()
            n_batches += 1
            global_step += 1
            # `ema` is worth watching live: it starts near 0 and ramps towards
            # `ema_decay`, and a run that ends while it is still low is the
            # failure mode described in the `EMA` docstring.
            bar.set_postfix(loss=f"{total / n_batches:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                            ema=f"{ema.current_decay():.4f}")

        train_loss = total / max(n_batches, 1)
        elapsed = time.time() - start
        print(f"epoch {epoch:3d} | {elapsed:6.1f}s | loss {train_loss:.4f} "
              f"| ema decay {ema.current_decay():.5f} | steps {global_step}")
        history.append({"epoch": epoch, "seconds": elapsed, "loss": train_loss,
                        "steps": global_step})

        torch.save({"model": unet.state_dict(),
                    "ema": ema.shadow,
                    "ema_updates": ema.num_updates,
                    "ema_decay": ema_decay,
                    "config": unet.config,
                    "timesteps": timesteps}, out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Preview from the EMA weights -- the same weights generation will use.
        preview_net = copy.deepcopy(unet)
        ema.copy_to(preview_net)
        preview_net.eval()
        preview = Diffusion(preview_net, timesteps).to(device)
        save_grid(preview.ddim_sample((16, 3, image_size, image_size), device,
                                      num_steps=preview_steps, progress=False),
                  out_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=4)
        del preview_net, preview

    return out_dir


def load_ddpm(run_dir: str | Path, device: torch.device,
              use_ema: bool = True) -> Diffusion:
    """Rebuild a trained DDPM from its run folder.

    Args:
        run_dir: Folder written by `train_ddpm`.
        device: Device to place the model on.
        use_ema: Load the EMA weights rather than the raw ones. Almost always
            what you want; `False` is the escape hatch if an old checkpoint has
            a contaminated EMA (see the `EMA` docstring).

    Returns:
        A `Diffusion` wrapping the loaded network, in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    path = Path(run_dir) / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    unet = UNet(**ckpt["config"]).to(device)
    unet.load_state_dict(ckpt["model"])

    if use_ema and "ema" in ckpt:
        updates = ckpt.get("ema_updates", 0)
        # How much of the shadow is still the random initialisation it started
        # from. See the EMA docstring for why this is worth checking.
        contamination = ckpt.get("ema_decay", 0.9999) ** max(updates, 0)
        if contamination > 0.01:
            print(f"WARNING: the EMA has only {updates} updates, so roughly "
                  f"{contamination:.0%} of it is still the random initialisation. "
                  f"Train longer, or pass use_ema=False.")
        for name, p in unet.named_parameters():
            if name in ckpt["ema"]:
                p.data.copy_(ckpt["ema"][name].to(device))
        print(f"loaded EMA weights ({updates} updates)")
    else:
        print("using raw weights")

    unet.eval()
    return Diffusion(unet, ckpt["timesteps"]).to(device)


@torch.no_grad()
def generate(run_dir: str | Path, device: torch.device, n: int,
             out_dir: Optional[str | Path] = None, batch_size: int = 64,
             steps: int = 50) -> Path:
    """Generate `n` images with DDIM and write them as PNGs for FID.

    Args:
        run_dir: Folder written by `train_ddpm`.
        device: Device to run on.
        n: Number of images to generate.
        out_dir: Destination folder; defaults to `<run_dir>/generated`.
        batch_size: Images per sampling run.
        steps: DDIM steps. 50 is the usual quality/speed compromise.

    Returns:
        The destination folder.
    """
    from data import save_pngs

    diffusion = load_ddpm(run_dir, device)
    size = diffusion.model.config["image_size"]
    out_dir = Path(out_dir) if out_dir else Path(run_dir) / "generated"

    written = 0
    # One outer bar over images; the inner per-batch DDIM bar is disabled so the
    # two do not fight over the same line.
    with tqdm(total=n, desc=f"ddpm generate ({steps} steps)", unit="img") as bar:
        while written < n:
            b = min(batch_size, n - written)
            batch = diffusion.ddim_sample((b, 3, size, size), device,
                                          num_steps=steps, progress=False)
            just_written = save_pngs(batch, out_dir, start_index=written)
            written += just_written
            bar.update(just_written)
    print(f"wrote {written} images to {out_dir}")
    return out_dir
