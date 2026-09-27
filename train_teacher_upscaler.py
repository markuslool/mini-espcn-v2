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
# TEACHER CONFIGURATION
# ============================================================

DIV2K_DIR       = "./DIV2K_train_HR"
VALID_DIR       = "./DIV2K_valid_HR"

TRAIN_CACHE     = "./div2k_teacher_cache_x2_64.pt"

SCALE_FACTOR    = 2
PATCH_SIZE      = 64
LR_SIZE         = PATCH_SIZE // SCALE_FACTOR

# Teacher is intentionally much larger than the ~13K student.
# ~34K parameters in the main convolutional body (~2.2x the original student).
# It is still deliberately compact so it can train quickly and act
# as a practical knowledge source for the tiny real-time network.
TEACHER_C1      = 40
TEACHER_C2      = 28

BATCH_SIZE      = 64
EPOCHS          = 100
NUM_WORKERS     = 2
PREFETCH        = 4

LEARNING_RATE   = 2e-4
MIN_LR         = 1e-5
WEIGHT_DECAY    = 1e-5

USE_AMP         = True

PATCHES_PER_IMAGE_CACHE = 120

VAL_IMAGES     = 50
VAL_PATCHES    = 8

VAL_EVERY      = 5
EMA_DECAY      = 0.999

CHKPT_DIR      = "./teacher_checkpoints"


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


os.makedirs(CHKPT_DIR, exist_ok=True)


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def find_images(directory):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp",
                "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP"):
        paths += glob.glob(os.path.join(directory, "**", ext), recursive=True)
    return sorted(set(paths))


# ============================================================
# TRAIN CACHE
# ============================================================

def build_train_cache_if_needed():
    """
    Creates paired HR/LR training patches.

    LR is generated from HR with a controlled degradation:
      HR -> mild blur (occasionally) -> bicubic x2 downsample

    The degradation is deliberately conservative for the first teacher:
    the teacher learns high-quality reconstruction rather than trying to
    model every possible game-engine artifact at once.
    """

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

        # Random crop coordinates rather than grid-only sampling.
        for _ in range(PATCHES_PER_IMAGE_CACHE):
            if w == PATCH_SIZE:
                x = 0
            else:
                x = random.randint(0, w - PATCH_SIZE)

            if h == PATCH_SIZE:
                y = 0
            else:
                y = random.randint(0, h - PATCH_SIZE)

            hr = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

            # Conservative degradation.
            if random.random() < 0.25:
                hr_for_lr = hr.filter(ImageFilter.GaussianBlur(
                    radius=random.uniform(0.15, 0.45)
                ))
            else:
                hr_for_lr = hr

            lr = hr_for_lr.resize(
                (LR_SIZE, LR_SIZE),
                Image.Resampling.BICUBIC
            )

            hr_list.append(
                (to_tensor(hr) * 255.0).round().clamp(0, 255).byte()
            )
            lr_list.append(
                (to_tensor(lr) * 255.0).round().clamp(0, 255).byte()
            )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)} images, "
                  f"patches: {len(hr_list)}")

    n = len(hr_list)

    if n == 0:
        raise RuntimeError("Не удалось создать ни одного патча.")

    print(f"Всего train patches: {n}")

    lr_tensor = torch.empty(
        (n, 3, LR_SIZE, LR_SIZE), dtype=torch.uint8
    )
    hr_tensor = torch.empty(
        (n, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.uint8
    )

    for i, t in enumerate(lr_list):
        lr_tensor[i] = t

    for i, t in enumerate(hr_list):
        hr_tensor[i] = t

    del lr_list, hr_list
    gc.collect()

    torch.save(
        {
            "lr": lr_tensor,
            "hr": hr_tensor,
        },
        TRAIN_CACHE,
    )

    print(
        f"Кэш сохранён: LR={tuple(lr_tensor.shape)}, "
        f"HR={tuple(hr_tensor.shape)}"
    )


# ============================================================
# TRAIN DATASET
# ============================================================

class CachedTrainDataset(Dataset):

    def __init__(self, cache_file):
        data = torch.load(cache_file, map_location="cpu")

        self.lr = data["lr"]
        self.hr = data["hr"]

        print(
            f"Train dataset: {len(self.lr)} pairs | "
            f"LR={tuple(self.lr.shape[1:])} | "
            f"HR={tuple(self.hr.shape[1:])}"
        )

    def __len__(self):
        return self.lr.shape[0]

    def __getitem__(self, idx):
        lr = self.lr[idx].float().div_(255.0)
        hr = self.hr[idx].float().div_(255.0)
        return lr, hr


# ============================================================
# VALIDATION DATASET
# ============================================================

class ValidationDataset(Dataset):
    """
    Uses completely separate DIV2K_valid_HR images.

    Patches are deterministic so validation remains comparable between
    epochs.
    """

    def __init__(self, directory, patches_per_image=8):
        paths = find_images(directory)

        if not paths:
            raise RuntimeError(
                f"Не найдены validation images в {directory}"
            )

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

            # Deterministic positions.
            coords = [
                (0, 0),
                (max(0, w - PATCH_SIZE), 0),
                (0, max(0, h - PATCH_SIZE)),
                (
                    max(0, w - PATCH_SIZE),
                    max(0, h - PATCH_SIZE)
                ),
                (
                    max(0, (w - PATCH_SIZE) // 2),
                    max(0, (h - PATCH_SIZE) // 2)
                ),
            ]

            # Add a few deterministic pseudo-random positions.
            rng = random.Random(hash(path) & 0xffffffff)

            while len(coords) < patches_per_image:
                coords.append((
                    rng.randint(0, w - PATCH_SIZE),
                    rng.randint(0, h - PATCH_SIZE),
                ))

            for x, y in coords[:patches_per_image]:
                self.samples.append((path, x, y))

        print(
            f"Validation: {len(self.samples)} patches "
            f"from {len(set(s[0] for s in self.samples))} images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, x, y = self.samples[idx]

        img = Image.open(path).convert("RGB")

        hr = img.crop(
            (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
        )

        lr = hr.resize(
            (LR_SIZE, LR_SIZE),
            Image.Resampling.BICUBIC
        )

        to_tensor = T.ToTensor()

        return to_tensor(lr), to_tensor(hr)


# ============================================================
# GPU AUGMENTATION
# ============================================================

def augment_batch(lr, hr):
    """
    Same spatial transform is applied to LR and HR.
    """

    b = lr.size(0)

    horizontal = torch.rand(b, device=lr.device) > 0.5
    vertical = torch.rand(b, device=lr.device) > 0.5

    if horizontal.any():
        lr[horizontal] = torch.flip(lr[horizontal], dims=[3])
        hr[horizontal] = torch.flip(hr[horizontal], dims=[3])

    if vertical.any():
        lr[vertical] = torch.flip(lr[vertical], dims=[2])
        hr[vertical] = torch.flip(hr[vertical], dims=[2])

    rotations = torch.randint(
        0, 4, (b,), device=lr.device
    )

    for k in range(1, 4):
        mask = rotations == k

        if mask.any():
            lr[mask] = torch.rot90(
                lr[mask], k=k, dims=[2, 3]
            )
            hr[mask] = torch.rot90(
                hr[mask], k=k, dims=[2, 3]
            )

    return lr, hr


# ============================================================
# TEACHER MODEL
# ============================================================

class TeacherESPCN(nn.Module):
    """
    Larger teacher version of the original MiniESPCN.

    Main body:
        Conv3x3  3  -> 40 + ReLU
        Conv5x5 40  -> 28 + ReLU
        Conv3x3 28  -> 12
        PixelShuffle x2

    Parameters are ~34K, roughly 2.2x the original MiniESPCN.

    The teacher does NOT need to match the student's HLSL architecture.
    Its job is to learn a stronger reconstruction mapping that can later
    supervise the small real-time student.
    """

    def __init__(self, scale_factor=2):
        super().__init__()

        assert scale_factor == 2

        self.conv1 = nn.Conv2d(
            3, TEACHER_C1,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            TEACHER_C1, TEACHER_C2,
            kernel_size=5,
            padding=2
        )

        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            TEACHER_C2,
            3 * (scale_factor ** 2),
            kernel_size=3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def gradient_loss(pred, target):
    """
    L1 loss on horizontal/vertical image gradients.

    This encourages reconstruction of:
      - text edges
      - thin lines
      - object boundaries
      - high-frequency detail
    """

    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    return (
        F.l1_loss(pred_dx, target_dx) +
        F.l1_loss(pred_dy, target_dy)
    )


def teacher_loss(pred, target):
    """
    Main reconstruction objective.

    Charbonnier keeps the result stable.
    Gradient loss gives additional pressure to recover detail.
    """

    pixel = charbonnier_loss(pred, target)
    grad = gradient_loss(pred, target)

    return pixel + 0.10 * grad


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

        with torch.amp.autocast(
            "cuda",
            enabled=(USE_AMP and device.type == "cuda")
        ):
            out = model(lr).clamp(0.0, 1.0)

            loss = teacher_loss(out, hr)

        mse = torch.mean(
            (out - hr) ** 2,
            dim=(1, 2, 3)
        )

        psnr = 10.0 * torch.log10(
            1.0 / (mse + 1e-12)
        )

        total_psnr += psnr.sum().item()
        total_loss += loss.item() * lr.size(0)
        count += lr.size(0)

    model.train()

    return (
        total_psnr / max(count, 1),
        total_loss / max(count, 1)
    )


# ============================================================
# EMA
# ============================================================

@torch.no_grad()
def update_ema(ema, model, decay):

    for pe, pm in zip(
        ema.parameters(),
        model.parameters()
    ):
        pe.mul_(decay).add_(
            pm.detach(),
            alpha=1.0 - decay
        )

    for be, bm in zip(
        ema.buffers(),
        model.buffers()
    ):
        be.copy_(bm)


# ============================================================
# TRAIN
# ============================================================

def train():

    build_train_cache_if_needed()

    train_dataset = CachedTrainDataset(TRAIN_CACHE)
    val_dataset = ValidationDataset(
        VALID_DIR,
        patches_per_image=VAL_PATCHES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=(
            NUM_WORKERS if device.type == "cuda" else 0
        ),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            device.type == "cuda" and NUM_WORKERS > 0
        ),
        prefetch_factor=(
            PREFETCH if NUM_WORKERS > 0 else None
        ),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=(
            NUM_WORKERS if device.type == "cuda" else 0
        ),
        pin_memory=(device.type == "cuda"),
    )

    model = TeacherESPCN(
        scale_factor=SCALE_FACTOR
    ).to(device)

    ema_model = TeacherESPCN(
        scale_factor=SCALE_FACTOR
    ).to(device)

    ema_model.load_state_dict(model.state_dict())

    for p in ema_model.parameters():
        p.requires_grad_(False)

    n_params = sum(
        p.numel() for p in model.parameters()
    )

    print()
    print("=" * 60)
    print("TEACHER")
    print("=" * 60)
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LEARNING_RATE} -> {MIN_LR}")
    print("Loss: Charbonnier + 0.10 * Gradient")
    print("=" * 60)
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

    use_amp = (
        USE_AMP and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_psnr = -1.0
    global_step = 0

    model.train()

    for epoch in range(1, EPOCHS + 1):

        running_loss = 0.0

        for lr, hr in train_loader:

            lr = lr.to(
                device,
                non_blocking=True
            )

            hr = hr.to(
                device,
                non_blocking=True
            )

            lr, hr = augment_batch(lr, hr)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp
            ):
                output = model(lr)

                loss = teacher_loss(
                    output,
                    hr
                )

            scaler.scale(loss).backward()

            # Prevent occasional exploding gradients from unusual
            # high-frequency patches.
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            update_ema(
                ema_model,
                model,
                EMA_DECAY
            )

            running_loss += loss.item()

        scheduler.step()

        epoch_loss = (
            running_loss /
            max(len(train_loader), 1)
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"loss={epoch_loss:.6f} "
            f"lr={current_lr:.3e} "
            f"step={global_step}"
        )

        if (
            epoch % VAL_EVERY == 0
            or epoch == EPOCHS
        ):

            raw_psnr, raw_val_loss = validate(
                model,
                val_loader
            )

            ema_psnr, ema_val_loss = validate(
                ema_model,
                val_loader
            )

            print(
                f"  RAW: loss={raw_val_loss:.6f} "
                f"PSNR={raw_psnr:.3f} dB"
            )

            print(
                f"  EMA: loss={ema_val_loss:.6f} "
                f"PSNR={ema_psnr:.3f} dB"
            )

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
            }

            torch.save(
                checkpoint,
                os.path.join(
                    CHKPT_DIR,
                    f"teacher_ep{epoch:03d}.pth"
                )
            )

            if ema_psnr > best_psnr:

                best_psnr = ema_psnr

                torch.save(
                    {
                        "model": ema_model.state_dict(),
                        "psnr": ema_psnr,
                        "epoch": epoch,
                        "parameters": n_params,
                    },
                    os.path.join(
                        CHKPT_DIR,
                        "teacher_best.pth"
                    )
                )

                print(
                    f"  -> Новый best teacher: "
                    f"{best_psnr:.3f} dB"
                )

    print()
    print("Обучение teacher завершено.")

    return ema_model


# ============================================================
# EXPORT TEACHER
# ============================================================

def export_teacher(model):
    path = "teacher_best_state_dict.pth"

    torch.save(
        model.state_dict(),
        path
    )

    print(f"Teacher weights: {path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    teacher = train()

    export_teacher(teacher)

    print()
    print("[OK]")
    print("Teacher checkpoint:")
    print(
        os.path.join(
            CHKPT_DIR,
            "teacher_best.pth"
        )
    )
