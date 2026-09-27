#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPU v5e teacher x2 upscaler
~32K parameters, JAX + Flax + Optax.

Designed for TPU v5e. The model is intentionally small enough to train
quickly, while being a stronger teacher than the <=5K student.

Input : RGB LR [B,H,W,3], values [0,1]
Output: RGB HR [B,2H,2W,3], values approximately [0,1]

Recommended environment:
  pip install -U "jax[tpu]" flax optax orbax-checkpoint pillow numpy

The script accepts:
  1) NPZ cache with arrays "lr" and "hr"
  2) PyTorch .pt cache produced by the previous training script
     (torch is only used on CPU to read the cache)

Important:
- TPU training uses bfloat16 for matmul/convolution, float32 for losses.
- No CUDA/AMP/GradScaler code is used.
- Checkpoints are saved with Orbax.
- The best teacher is also exported as a portable NPZ file.
"""

from __future__ import annotations

import os
import math
import time
import json
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import orbax.checkpoint as ocp


# ============================================================
# CONFIG
# ============================================================

SEED = 42

# Existing cache from the old script can be used directly.
CACHE_FILE = "./div2k_teacher_cache_x2_64.pt"

# Optional faster native cache. If this exists it is preferred.
NPZ_CACHE_FILE = "./div2k_teacher_cache_x2_64.npz"

# Put a real DIV2K validation NPZ here if available:
# keys: lr [N,3,32,32] or [N,32,32,3], hr [N,3,64,64] or [N,64,64,3]
VAL_NPZ_FILE = "./div2k_valid_x2.npz"

CHECKPOINT_DIR = "./teacher_tpu_checkpoints"
EXPORT_FILE = "./teacher_best_tpu.npz"

SCALE = 2

# TPU likes larger batches than a small GPU.
# Increase until TPU memory/throughput stops improving.
BATCH_SIZE = 128

EPOCHS = 80
LEARNING_RATE = 2e-3
MIN_LR = 2e-5
WEIGHT_DECAY = 1e-4

# Teacher losses. No GAN/perceptual loss: avoid hallucinated detail.
PIXEL_W = 1.00
GRAD_W = 0.12
DETAIL_W = 0.06

EMA_DECAY = 0.999

PRINT_EVERY = 100
SAVE_EVERY = 5

# If no separate validation file exists, a deterministic holdout is made
# from the training cache. A real DIV2K validation set is preferred.
HOLDOUT_COUNT = 512

# TPU distributed mode:
# JAX automatically sees all TPU devices. The script uses pmap.
USE_ALL_TPU_CORES = True


# ============================================================
# MODEL ~32K PARAMS
# ============================================================

class ResidualBlock(nn.Module):
    channels: int = 32

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(
            self.channels, (3, 3), padding="SAME",
            dtype=jnp.bfloat16, param_dtype=jnp.float32
        )(x)
        y = nn.relu(y)
        y = nn.Conv(
            self.channels, (3, 3), padding="SAME",
            dtype=jnp.bfloat16, param_dtype=jnp.float32
        )(y)

        # Small residual scale keeps the teacher stable early in training.
        return x + jnp.asarray(0.1, dtype=y.dtype) * y


class TeacherX2(nn.Module):
    channels: int = 32
    blocks: int = 2

    @nn.compact
    def __call__(self, x):
        # Bicubic/base branch is supplied outside the network using JAX.
        # The network predicts a detail residual.
        y = nn.Conv(
            self.channels, (3, 3), padding="SAME",
            dtype=jnp.bfloat16, param_dtype=jnp.float32
        )(x)
        y = nn.relu(y)

        for _ in range(self.blocks):
            y = ResidualBlock(self.channels)(y)

        y = nn.Conv(
            self.channels, (3, 3), padding="SAME",
            dtype=jnp.bfloat16, param_dtype=jnp.float32
        )(y)
        y = nn.relu(y)

        # 32 -> 12 channels, then depth-to-space x2.
        y = nn.Conv(
            3 * SCALE * SCALE, (3, 3), padding="SAME",
            dtype=jnp.bfloat16, param_dtype=jnp.float32
        )(y)

        # PixelShuffle / depth-to-space.
        y = jnp.reshape(
            y,
            (y.shape[0], y.shape[1], y.shape[2], SCALE, SCALE, 3)
        )
        y = jnp.transpose(y, (0, 1, 3, 2, 4, 5))
        y = jnp.reshape(
            y,
            (y.shape[0], y.shape[1] * SCALE, y.shape[2] * SCALE, 3)
        )

        return y


def count_params(params):
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params)))


# ============================================================
# IMAGE OPS
# ============================================================

def bicubic_resize(x, out_h, out_w):
    return jax.image.resize(
        x, (x.shape[0], out_h, out_w, x.shape[-1]),
        method="bicubic",
        antialias=False,
    )


def charbonnier(x, y, eps=1e-3):
    d = x.astype(jnp.float32) - y.astype(jnp.float32)
    return jnp.mean(jnp.sqrt(d * d + eps * eps))


def sobel_edges(x):
    # x: NHWC
    gray = (
        x[..., 0:1] * 0.299 +
        x[..., 1:2] * 0.587 +
        x[..., 2:3] * 0.114
    )

    kx = jnp.asarray(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]], dtype=jnp.float32
    ).reshape(3, 3, 1, 1)

    ky = jnp.asarray(
        [[-1., -2., -1.],
         [ 0.,  0.,  0.],
         [ 1.,  2.,  1.]], dtype=jnp.float32
    ).reshape(3, 3, 1, 1)

    gx = jax.lax.conv_general_dilated(
        gray.astype(jnp.float32), kx,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    gy = jax.lax.conv_general_dilated(
        gray.astype(jnp.float32), ky,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return jnp.sqrt(gx * gx + gy * gy + 1e-6)


def laplacian(x):
    # Detail/high-frequency component.
    k = jnp.asarray(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]], dtype=jnp.float32
    ).reshape(3, 3, 1, 1)

    # Apply independently to RGB.
    k = jnp.tile(k, (1, 1, 3, 1))

    return jax.lax.conv_general_dilated(
        x.astype(jnp.float32), k,
        window_strides=(1, 1),
        padding="SAME",
        feature_group_count=3,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )


# ============================================================
# DATA
# ============================================================

def chw_to_hwc(a):
    a = np.asarray(a)
    if a.ndim != 4:
        raise ValueError(f"Expected 4D array, got {a.shape}")
    if a.shape[1] == 3:
        a = np.transpose(a, (0, 2, 3, 1))
    return a


def load_cache():
    npz_path = Path(NPZ_CACHE_FILE)
    if npz_path.exists():
        print(f"Loading native NPZ cache: {npz_path}")
        z = np.load(npz_path)
        lr = z["lr"]
        hr = z["hr"]
    else:
        pt_path = Path(CACHE_FILE)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Cache not found:\n  {NPZ_CACHE_FILE}\n  {CACHE_FILE}"
            )

        print(f"Loading old PyTorch cache on CPU: {pt_path}")
        import torch
        data = torch.load(pt_path, map_location="cpu")
        lr = data["lr"].numpy()
        hr = data["hr"].numpy()

    lr = chw_to_hwc(lr)
    hr = chw_to_hwc(hr)

    lr = lr.astype(np.float32)
    hr = hr.astype(np.float32)

    if lr.max() > 1.5:
        lr /= 255.0
    if hr.max() > 1.5:
        hr /= 255.0

    lr = np.clip(lr, 0.0, 1.0)
    hr = np.clip(hr, 0.0, 1.0)

    print(f"LR cache: {lr.shape} {lr.dtype}")
    print(f"HR cache: {hr.shape} {hr.dtype}")

    return lr, hr


def load_validation():
    path = Path(VAL_NPZ_FILE)
    if not path.exists():
        return None, None

    z = np.load(path)
    lr = chw_to_hwc(z["lr"]).astype(np.float32)
    hr = chw_to_hwc(z["hr"]).astype(np.float32)

    if lr.max() > 1.5:
        lr /= 255.0
    if hr.max() > 1.5:
        hr /= 255.0

    return np.clip(lr, 0, 1), np.clip(hr, 0, 1)


def augment_numpy(lr, hr, rng):
    """Same geometric augmentation for LR and HR."""
    b = lr.shape[0]

    flip_h = rng.random(b) < 0.5
    flip_v = rng.random(b) < 0.5
    rot = rng.integers(0, 4, size=b)

    out_lr = lr.copy()
    out_hr = hr.copy()

    for i in range(b):
        a = out_lr[i]
        c = out_hr[i]

        if flip_h[i]:
            a = np.flip(a, axis=1)
            c = np.flip(c, axis=1)

        if flip_v[i]:
            a = np.flip(a, axis=0)
            c = np.flip(c, axis=0)

        if rot[i]:
            a = np.rot90(a, int(rot[i]), axes=(0, 1))
            c = np.rot90(c, int(rot[i]), axes=(0, 1))

        out_lr[i] = a
        out_hr[i] = c

    return out_lr.copy(), out_hr.copy()


def make_batches(lr, hr, batch_size, rng, shuffle=True):
    n = len(lr)
    idx = np.arange(n)

    if shuffle:
        rng.shuffle(idx)

    usable = (n // batch_size) * batch_size
    idx = idx[:usable]

    for s in range(0, usable, batch_size):
        ii = idx[s:s + batch_size]
        yield lr[ii], hr[ii]


# ============================================================
# TRAIN STATE
# ============================================================

class TrainState(train_state.TrainState):
    ema_params: any


def create_state(model, rng, sample):
    variables = model.init(rng, sample)
    params = variables["params"]

    schedule = optax.cosine_decay_schedule(
        init_value=LEARNING_RATE,
        decay_steps=EPOCHS,
        alpha=MIN_LR / LEARNING_RATE,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=WEIGHT_DECAY,
            b1=0.9,
            b2=0.99,
            eps=1e-8,
        ),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_params=params,
    )


def ema_update(old, new, decay):
    return jax.tree_util.tree_map(
        lambda a, b: decay * a + (1.0 - decay) * b,
        old, new
    )


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def loss_fn(params, apply_fn, lr, hr):
    residual = apply_fn({"params": params}, lr)
    base = bicubic_resize(lr, hr.shape[1], hr.shape[2])

    # Residual reconstruction. Teacher learns details on top of bicubic.
    pred = base + residual.astype(jnp.float32)

    pixel = charbonnier(pred, hr)

    pred_g = sobel_edges(pred)
    hr_g = sobel_edges(hr)
    grad = charbonnier(pred_g, hr_g)

    pred_l = laplacian(pred)
    hr_l = laplacian(hr)
    detail = jnp.mean(jnp.abs(pred_l - hr_l))

    total = (
        PIXEL_W * pixel +
        GRAD_W * grad +
        DETAIL_W * detail
    )

    return total, (pixel, grad, detail, pred)


@jax.jit
def train_step(state, lr, hr):
    def wrapped(params):
        return loss_fn(params, state.apply_fn, lr, hr)

    (loss, aux), grads = jax.value_and_grad(
        wrapped, has_aux=True
    )(state.params)

    # Average gradients across TPU replicas.
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")

    state = state.apply_gradients(grads=grads)

    new_ema = ema_update(state.ema_params, state.params, EMA_DECAY)
    state = state.replace(ema_params=new_ema)

    pixel, grad, detail, _ = aux
    pixel = jax.lax.pmean(pixel, axis_name="data")
    grad = jax.lax.pmean(grad, axis_name="data")
    detail = jax.lax.pmean(detail, axis_name="data")

    return state, (loss, pixel, grad, detail)


@jax.pmap(axis_name="data")
def p_train_step(state, lr, hr):
    return train_step(state, lr, hr)


@jax.jit
def eval_step(params, apply_fn, lr, hr):
    residual = apply_fn({"params": params}, lr)
    base = bicubic_resize(lr, hr.shape[1], hr.shape[2])
    pred = base + residual.astype(jnp.float32)

    pred = jnp.clip(pred, 0.0, 1.0)

    mse = jnp.mean((pred - hr) ** 2)
    pixel = charbonnier(pred, hr)
    grad = charbonnier(sobel_edges(pred), sobel_edges(hr))

    return mse, pixel, grad


def psnr_from_mse(mse):
    return -10.0 * math.log10(max(float(mse), 1e-12))


def validate(state, lr, hr, batch_size):
    total_mse = 0.0
    total_pixel = 0.0
    total_grad = 0.0
    count = 0

    for s in range(0, len(lr), batch_size):
        a = lr[s:s + batch_size]
        b = hr[s:s + batch_size]

        if len(a) == 0:
            continue

        # eval_jit does not require a TPU replica axis.
        mse, pix, grd = eval_step(
            state.ema_params,
            state.apply_fn,
            jnp.asarray(a),
            jnp.asarray(b),
        )

        n = len(a)
        total_mse += float(mse) * n
        total_pixel += float(pix) * n
        total_grad += float(grd) * n
        count += n

    mse = total_mse / max(count, 1)
    return psnr_from_mse(mse), total_pixel / max(count, 1), total_grad / max(count, 1)


# ============================================================
# CHECKPOINT / EXPORT
# ============================================================

def save_checkpoint(state, epoch, best_psnr):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    ckpt = ocp.StandardCheckpointer()
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}")
    ckpt.save(path, {"params": state.params, "ema_params": state.ema_params},
              force=True)

    with open(os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.json"), "w") as f:
        json.dump(
            {"epoch": epoch, "best_psnr": best_psnr},
            f, indent=2
        )


def export_npz(params):
    flat = {}

    def visit(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                visit(v, f"{prefix}{k}/")
        else:
            flat[prefix.rstrip("/")] = np.asarray(jax.device_get(tree))

    visit(params)

    np.savez(EXPORT_FILE, **flat)
    print(f"Exported teacher weights: {EXPORT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("TPU v5e Teacher x2")
    print("=" * 72)

    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())
    print("Local device count:", jax.local_device_count())
    print("Global device count:", jax.device_count())

    if not any("TPU" in str(d).upper() for d in jax.devices()):
        print("WARNING: TPU device was not detected. Check your TPU runtime.")

    lr_all, hr_all = load_cache()

    val_lr, val_hr = load_validation()

    # Deterministic holdout if a real DIV2K validation NPZ was not supplied.
    if val_lr is None:
        print("No separate validation NPZ found.")
        print(f"Using deterministic holdout of {HOLDOUT_COUNT} patches.")
        rng_split = np.random.default_rng(SEED)
        perm = rng_split.permutation(len(lr_all))

        n_val = min(HOLDOUT_COUNT, max(64, len(lr_all) // 10))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        val_lr = lr_all[val_idx]
        val_hr = hr_all[val_idx]
        lr_train = lr_all[train_idx]
        hr_train = hr_all[train_idx]
    else:
        lr_train = lr_all
        hr_train = hr_all
        print(f"Using external validation set: {val_lr.shape}")

    if BATCH_SIZE % jax.local_device_count() != 0:
        raise ValueError(
            f"BATCH_SIZE={BATCH_SIZE} must be divisible by "
            f"local_device_count={jax.local_device_count()}."
        )

    per_device_batch = BATCH_SIZE // jax.local_device_count()
    print(f"Global batch: {BATCH_SIZE}")
    print(f"Per TPU device batch: {per_device_batch}")

    # Important: model sees per-device batches under pmap.
    dummy = jnp.zeros(
        (per_device_batch, lr_train.shape[1], lr_train.shape[2], 3),
        dtype=jnp.float32
    )

    model = TeacherX2(channels=32, blocks=1)

    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)

    state = create_state(model, init_rng, dummy)

    n_params = count_params(state.params)
    print(f"Teacher parameters: {n_params:,}")

    if not (28_000 <= n_params <= 40_000):
        print("WARNING: parameter count is outside the requested ~30K range.")

    # Replicate state over all local TPU devices.
    state_repl = jax.device_put_replicated(
        state, jax.local_devices()
    )

    rng_np = np.random.default_rng(SEED)

    best_psnr = -1.0
    global_step = 0

    steps_per_epoch = len(lr_train) // BATCH_SIZE
    print(f"Training samples: {len(lr_train)}")
    print(f"Validation samples: {len(val_lr)}")
    print(f"Steps/epoch: {steps_per_epoch}")
    print(f"Epochs: {EPOCHS}")
    print(f"Total optimizer steps: {steps_per_epoch * EPOCHS}")
    print()

    # Compile once before timing real throughput.
    first_lr, first_hr = next(
        make_batches(lr_train, hr_train, BATCH_SIZE, rng_np, shuffle=True)
    )
    first_lr, first_hr = augment_numpy(first_lr, first_hr, rng_np)

    first_lr = first_lr.reshape(
        (jax.local_device_count(), per_device_batch,
         first_lr.shape[1], first_lr.shape[2], 3)
    )
    first_hr = first_hr.reshape(
        (jax.local_device_count(), per_device_batch,
         first_hr.shape[1], first_hr.shape[2], 3)
    )

    state_repl, metrics = p_train_step(
        state_repl,
        jnp.asarray(first_lr),
        jnp.asarray(first_hr)
    )
    jax.block_until_ready(metrics[0])
    global_step += 1

    print("First TPU compilation completed.")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        sums = np.zeros(4, dtype=np.float64)
        n_steps = 0

        for batch_lr, batch_hr in make_batches(
            lr_train, hr_train, BATCH_SIZE, rng_np, shuffle=True
        ):
            batch_lr, batch_hr = augment_numpy(
                batch_lr, batch_hr, rng_np
            )

            batch_lr = batch_lr.reshape(
                (jax.local_device_count(), per_device_batch,
                 batch_lr.shape[1], batch_lr.shape[2], 3)
            )
            batch_hr = batch_hr.reshape(
                (jax.local_device_count(), per_device_batch,
                 batch_hr.shape[1], batch_hr.shape[2], 3)
            )

            state_repl, metrics = p_train_step(
                state_repl,
                jnp.asarray(batch_lr),
                jnp.asarray(batch_hr)
            )

            metrics_np = np.asarray(jax.device_get(metrics))
            # Metrics have one value per replica.
            metrics_np = metrics_np.mean(axis=1)

            sums += metrics_np
            n_steps += 1
            global_step += 1

            if global_step % PRINT_EVERY == 0:
                print(
                    f"step {global_step:6d} | "
                    f"loss {sums[0]/n_steps:.6f} | "
                    f"pixel {sums[1]/n_steps:.6f} | "
                    f"grad {sums[2]/n_steps:.6f} | "
                    f"detail {sums[3]/n_steps:.6f}"
                )

        avg = sums / max(n_steps, 1)
        elapsed = time.time() - epoch_start

        # Get one copy of the replicated state.
        state_host = jax.tree_util.tree_map(
            lambda x: jax.device_get(x[0]),
            state_repl
        )

        psnr, val_pixel, val_grad = validate(
            state_host, val_lr, val_hr,
            batch_size=min(256, BATCH_SIZE)
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train {avg[0]:.6f} | "
            f"PSNR EMA {psnr:.3f} dB | "
            f"val pixel {val_pixel:.6f} | "
            f"time {elapsed:.1f}s"
        )

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(state_host, epoch, max(best_psnr, psnr))

        if psnr > best_psnr:
            best_psnr = psnr
            print(f"  NEW BEST: {best_psnr:.3f} dB")
            export_npz(state_host.ema_params)

            with open(
                os.path.join(CHECKPOINT_DIR, "best.json"), "w"
            ) as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "psnr": best_psnr,
                        "parameters": n_params,
                        "global_step": global_step,
                    },
                    f, indent=2
                )

    print()
    print("=" * 72)
    print("Training complete")
    print(f"Best EMA PSNR: {best_psnr:.3f} dB")
    print(f"Best weights: {EXPORT_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
