import os
import gc
import glob
import random
import multiprocessing
from pathlib import Path

from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ============================================================
# TEACHER #2 — SAME 32K ARCHITECTURE, MAX-EFFICIENCY DISTILLATION
# ============================================================
# Goal:
#   Train a fresh 32,184-param teacher from Teacher #1 while keeping
#   real HR as the primary authority. Teacher #1 supplies soft output,
#   gradients and intermediate representations.
#
# Important:
#   Teacher #1 is FROZEN. Teacher #2 starts from scratch so it can
#   learn a slightly different solution instead of merely copying T1.
# ============================================================

DIV2K_DIR       = "./DIV2K_train_HR"
VALID_DIR       = "./DIV2K_valid_HR"
TRAIN_CACHE     = "./div2k_teacher_cache_x2_64.pt"
TEACHER1_PATH   = "./teacher_checkpoints/teacher_best.pth"

SCALE_FACTOR    = 2
PATCH_SIZE      = 64
LR_SIZE         = PATCH_SIZE // SCALE_FACTOR

TEACHER_C1      = 40
TEACHER_C2      = 28

BATCH_SIZE      = 64
EPOCHS          = 100
NUM_WORKERS     = 2
PREFETCH        = 4

LEARNING_RATE   = 2e-4
MIN_LR          = 1e-5
WEIGHT_DECAY    = 1e-5
USE_AMP         = True

VAL_IMAGES      = 50
VAL_PATCHES     = 8
VAL_EVERY       = 5
EMA_DECAY       = 0.999

CHKPT_DIR      = "./teacher2_checkpoints"
BEST_NAME      = "teacher2_best.pth"

# ------------------------------------------------------------
# Distillation weights
# ------------------------------------------------------------
# HR remains the strongest authority. T1 is a soft teacher, not truth.
W_HR_PIXEL     = 0.55
W_KD_RGB       = 0.20
W_HR_GRAD      = 0.10
W_KD_GRAD      = 0.05
W_FEAT1        = 0.05
W_FEAT2        = 0.05

assert abs(
    W_HR_PIXEL + W_KD_RGB + W_HR_GRAD +
    W_KD_GRAD + W_FEAT1 + W_FEAT2 - 1.0
) < 1e-8

os.makedirs(CHKPT_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    print("CUDA не найдена — CPU")

# ============================================================
# IMAGE DISCOVERY / CACHE
# ============================================================

def find_images(directory):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp",
                "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP"):
        paths += glob.glob(os.path.join(directory, "**", ext), recursive=True)
    return sorted(set(paths))


def build_train_cache_if_needed():
    """Same conservative clean degradation as Teacher #1."""
    if os.path.exists(TRAIN_CACHE):
        print(f"Кэш найден: {TRAIN_CACHE}")
        return

    paths = find_images(DIV2K_DIR)
    if not paths:
        raise RuntimeError(f"Не найдены изображения в {DIV2K_DIR}")

    print(f"Train images: {len(paths)}")
    random.seed(42)

    lr_list = []
    hr_list = []
    to_tensor = T.ToTensor()

    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Пропуск {path}: {e}")
            continue

        w, h = img.size
        if w < PATCH_SIZE or h < PATCH_SIZE:
            continue

        # Keep exactly the same distribution as T1.
        for _ in range(120):
            x = 0 if w == PATCH_SIZE else random.randint(0, w - PATCH_SIZE)
            y = 0 if h == PATCH_SIZE else random.randint(0, h - PATCH_SIZE)

            hr = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

            if random.random() < 0.25:
                hr_for_lr = hr.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.45))
                )
            else:
                hr_for_lr = hr

            lr = hr_for_lr.resize((LR_SIZE, LR_SIZE), Image.Resampling.BICUBIC)

            hr_list.append((to_tensor(hr) * 255.0).round().clamp(0, 255).byte())
            lr_list.append((to_tensor(lr) * 255.0).round().clamp(0, 255).byte())

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)} images, patches: {len(hr_list)}")

    n = len(hr_list)
    if n == 0:
        raise RuntimeError("Не удалось создать ни одного патча.")

    lr_tensor = torch.empty((n, 3, LR_SIZE, LR_SIZE), dtype=torch.uint8)
    hr_tensor = torch.empty((n, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.uint8)

    for i, t in enumerate(lr_list):
        lr_tensor[i] = t
    for i, t in enumerate(hr_list):
        hr_tensor[i] = t

    del lr_list, hr_list
    gc.collect()

    torch.save({"lr": lr_tensor, "hr": hr_tensor}, TRAIN_CACHE)
    print(f"Кэш сохранён: LR={tuple(lr_tensor.shape)}, HR={tuple(hr_tensor.shape)}")


class CachedTrainDataset(Dataset):
    def __init__(self, cache_file):
        data = torch.load(cache_file, map_location="cpu")
        self.lr = data["lr"]
        self.hr = data["hr"]
        print(f"Train dataset: {len(self.lr)} pairs")

    def __len__(self):
        return self.lr.shape[0]

    def __getitem__(self, idx):
        return self.lr[idx].float().div_(255.0), self.hr[idx].float().div_(255.0)


class ValidationDataset(Dataset):
    def __init__(self, directory, patches_per_image=8):
        paths = find_images(directory)
        if not paths:
            raise RuntimeError(f"Не найдены validation images в {directory}")
        paths = paths[:VAL_IMAGES]
        self.samples = []

        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            if w < PATCH_SIZE or h < PATCH_SIZE:
                continue

            coords = [
                (0, 0),
                (max(0, w - PATCH_SIZE), 0),
                (0, max(0, h - PATCH_SIZE)),
                (max(0, w - PATCH_SIZE), max(0, h - PATCH_SIZE)),
                (max(0, (w - PATCH_SIZE) // 2), max(0, (h - PATCH_SIZE) // 2)),
            ]
            # Stable per-file seed, independent of Python's randomized hash().
            seed = sum((j + 1) * ord(c) for j, c in enumerate(path)) & 0xffffffff
            rng = random.Random(seed)
            while len(coords) < patches_per_image:
                coords.append((rng.randint(0, w - PATCH_SIZE), rng.randint(0, h - PATCH_SIZE)))

            for x, y in coords[:patches_per_image]:
                self.samples.append((path, x, y))

        print(f"Validation: {len(self.samples)} patches from {len(set(s[0] for s in self.samples))} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, x, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        hr = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        lr = hr.resize((LR_SIZE, LR_SIZE), Image.Resampling.BICUBIC)
        to_tensor = T.ToTensor()
        return to_tensor(lr), to_tensor(hr)

# ============================================================
# AUGMENTATION
# ============================================================

def augment_batch(lr, hr):
    b = lr.size(0)
    horizontal = torch.rand(b, device=lr.device) > 0.5
    vertical = torch.rand(b, device=lr.device) > 0.5

    if horizontal.any():
        lr[horizontal] = torch.flip(lr[horizontal], dims=[3])
        hr[horizontal] = torch.flip(hr[horizontal], dims=[3])
    if vertical.any():
        lr[vertical] = torch.flip(lr[vertical], dims=[2])
        hr[vertical] = torch.flip(hr[vertical], dims=[2])

    rotations = torch.randint(0, 4, (b,), device=lr.device)
    for k in range(1, 4):
        mask = rotations == k
        if mask.any():
            lr[mask] = torch.rot90(lr[mask], k=k, dims=[2, 3])
            hr[mask] = torch.rot90(hr[mask], k=k, dims=[2, 3])
    return lr, hr

# ============================================================
# MODEL
# ============================================================

class TeacherESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        assert scale_factor == 2
        self.conv1 = nn.Conv2d(3, TEACHER_C1, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(TEACHER_C1, TEACHER_C2, 5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(TEACHER_C2, 3 * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x, return_features=False):
        f1 = self.relu1(self.conv1(x))
        f2 = self.relu2(self.conv2(f1))
        out = self.pixel_shuffle(self.conv3(f2))
        if return_features:
            return out, f1, f2
        return out

# ============================================================
# LOSSES
# ============================================================

def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def feature_loss(pred, target):
    # Normalize each channel globally per sample. This matches structure
    # rather than forcing T2 to reproduce arbitrary activation magnitude.
    p = F.normalize(pred.flatten(2), dim=2)
    t = F.normalize(target.flatten(2), dim=2)
    return F.l1_loss(p, t)


def distill_loss(out2, hr, out1, f1_2, f2_2, f1_1, f2_1):
    hr_pixel = charbonnier_loss(out2, hr)
    kd_rgb = charbonnier_loss(out2, out1)
    hr_grad = gradient_loss(out2, hr)
    kd_grad = gradient_loss(out2, out1)
    feat1 = feature_loss(f1_2, f1_1)
    feat2 = feature_loss(f2_2, f2_1)

    total = (
        W_HR_PIXEL * hr_pixel +
        W_KD_RGB   * kd_rgb +
        W_HR_GRAD  * hr_grad +
        W_KD_GRAD  * kd_grad +
        W_FEAT1    * feat1 +
        W_FEAT2    * feat2
    )
    return total, hr_pixel, kd_rgb, hr_grad, kd_grad, feat1, feat2

# ============================================================
# EMA
# ============================================================

@torch.no_grad()
def update_ema(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)

# ============================================================
# TEACHER #1 LOADING
# ============================================================

def load_teacher1():
    if not os.path.exists(TEACHER1_PATH):
        raise FileNotFoundError(
            f"Не найден Teacher #1: {TEACHER1_PATH}\n"
            "Положи teacher_best.pth в teacher_checkpoints."
        )

    t1 = TeacherESPCN(scale_factor=SCALE_FACTOR).to(device)
    ckpt = torch.load(TEACHER1_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    t1.load_state_dict(state, strict=True)
    t1.eval()
    for p in t1.parameters():
        p.requires_grad_(False)

    print(f"Teacher #1 loaded: {TEACHER1_PATH}")
    if isinstance(ckpt, dict) and "psnr" in ckpt:
        print(f"Teacher #1 saved PSNR: {ckpt['psnr']:.3f} dB")
    return t1

# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_psnr = 0.0
    total_loss = 0.0
    count = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
            out = model(lr).clamp(0.0, 1.0)
            loss = charbonnier_loss(out, hr) + 0.10 * gradient_loss(out, hr)

        mse = torch.mean((out - hr) ** 2, dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
        total_psnr += psnr.sum().item()
        total_loss += loss.item() * lr.size(0)
        count += lr.size(0)

    model.train()
    return total_psnr / max(count, 1), total_loss / max(count, 1)

# ============================================================
# TRAIN
# ============================================================

def train():
    build_train_cache_if_needed()

    train_dataset = CachedTrainDataset(TRAIN_CACHE)
    val_dataset = ValidationDataset(VALID_DIR, patches_per_image=VAL_PATCHES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=(NUM_WORKERS if device.type == "cuda" else 0),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda" and NUM_WORKERS > 0),
        prefetch_factor=(PREFETCH if NUM_WORKERS > 0 else None),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=(NUM_WORKERS if device.type == "cuda" else 0),
        pin_memory=(device.type == "cuda"),
    )

    teacher1 = load_teacher1()

    model = TeacherESPCN(scale_factor=SCALE_FACTOR).to(device)
    ema_model = TeacherESPCN(scale_factor=SCALE_FACTOR).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())

    print()
    print("=" * 72)
    print("TEACHER #2 — DISTILLATION")
    print("=" * 72)
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LEARNING_RATE} -> {MIN_LR}")
    print("T1: frozen, loaded from teacher_best.pth")
    print("Loss:")
    print(f"  {W_HR_PIXEL:.2f} * HR Charbonnier")
    print(f"  {W_KD_RGB:.2f} * T1 RGB Charbonnier")
    print(f"  {W_HR_GRAD:.2f} * HR Gradient")
    print(f"  {W_KD_GRAD:.2f} * T1 Gradient")
    print(f"  {W_FEAT1:.2f} * T1 conv1 feature")
    print(f"  {W_FEAT2:.2f} * T1 conv2 feature")
    print("=" * 72)
    print()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR
    )

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_psnr = -1.0
    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        running_parts = [0.0] * 6

        for lr, hr in train_loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            lr, hr = augment_batch(lr, hr)

            optimizer.zero_grad(set_to_none=True)

            # Teacher #1 inference is frozen and excluded from autograd.
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out1, f1_1, f2_1 = teacher1(lr, return_features=True)
                    out1 = out1.clamp(0.0, 1.0)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out2, f1_2, f2_2 = model(lr, return_features=True)
                loss, p_hr, p_kd, g_hr, g_kd, fl1, fl2 = distill_loss(
                    out2, hr, out1, f1_2, f2_2, f1_1, f2_1
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            update_ema(ema_model, model, EMA_DECAY)

            running_loss += loss.item()
            parts = [p_hr, p_kd, g_hr, g_kd, fl1, fl2]
            for i, p in enumerate(parts):
                running_parts[i] += p.item()

        scheduler.step()
        epoch_loss = running_loss / max(len(train_loader), 1)
        current_lr = optimizer.param_groups[0]["lr"]
        part_avg = [x / max(len(train_loader), 1) for x in running_parts]

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] loss={epoch_loss:.6f} "
            f"lr={current_lr:.3e} step={global_step} "
            f"HR={part_avg[0]:.5f} KD={part_avg[1]:.5f} "
            f"GHR={part_avg[2]:.5f} GKD={part_avg[3]:.5f} "
            f"F1={part_avg[4]:.5f} F2={part_avg[5]:.5f}"
        )

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            raw_psnr, raw_val_loss = validate(model, val_loader)
            ema_psnr, ema_val_loss = validate(ema_model, val_loader)

            print(f"  RAW: loss={raw_val_loss:.6f} PSNR={raw_psnr:.3f} dB")
            print(f"  EMA: loss={ema_val_loss:.6f} PSNR={ema_psnr:.3f} dB")

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": epoch_loss,
                "psnr_raw": raw_psnr,
                "psnr_ema": ema_psnr,
                "global_step": global_step,
                "parameters": n_params,
                "teacher1": TEACHER1_PATH,
            }
            torch.save(checkpoint, os.path.join(CHKPT_DIR, f"teacher2_ep{epoch:03d}.pth"))

            if ema_psnr > best_psnr:
                best_psnr = ema_psnr
                torch.save(
                    {
                        "model": ema_model.state_dict(),
                        "psnr": ema_psnr,
                        "epoch": epoch,
                        "parameters": n_params,
                        "teacher1": TEACHER1_PATH,
                    },
                    os.path.join(CHKPT_DIR, BEST_NAME)
                )
                print(f"  -> Новый best Teacher #2: {best_psnr:.3f} dB")

    print()
    print("Обучение Teacher #2 завершено.")
    return ema_model


def export_teacher(model):
    path = "teacher2_best_state_dict.pth"
    torch.save(model.state_dict(), path)
    print(f"Teacher #2 weights: {path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    teacher2 = train()
    export_teacher(teacher2)
    print("[OK]")
    print(os.path.join(CHKPT_DIR, BEST_NAME))
