# upscale_cpu.py
"""
CPU-апскейлер x2 на базе TeacherESPCN + билинейный baseline.

Для каждого входного изображения сохраняются ДВА результата:
    <name>_bilinear_x2.png   — билинейная интерполяция (baseline)
    <name>_teacher_x2.png    — результат вашей сетки TeacherESPCN

Использование:
    python upscale_cpu.py input.png
    python upscale_cpu.py input.png output_dir
    python upscale_cpu.py ./input_dir ./output_dir
    python upscale_cpu.py ./input_dir ./output_dir --side-by-side
    python upscale_cpu.py ./input_dir ./output_dir --tile 256 --overlap 16

Формат входа: PNG / JPG / JPEG / BMP / WEBP.
Формат выхода: PNG (без потерь).
"""

import os
import gc
import glob
import time
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG (должен совпадать с train_teacher_upscaler.py)
# ============================================================

SCALE_FACTOR = 2

TEACHER_C1 = 40
TEACHER_C2 = 28

DEFAULT_WEIGHTS = "teacher_best_state_dict.pth"
FALLBACK_CKPT   = "./teacher_checkpoints/teacher_best.pth"

DEFAULT_TILE    = 256
DEFAULT_OVERLAP = 16

NUM_THREADS = 0


# ============================================================
# DEVICE / THREADS
# ============================================================

def setup_cpu():
    if NUM_THREADS > 0:
        torch.set_num_threads(NUM_THREADS)
    else:
        n = max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(n)

    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Device: cpu")


# ============================================================
# MODEL
# ============================================================

class TeacherESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        assert scale_factor == 2

        self.conv1 = nn.Conv2d(3, TEACHER_C1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(TEACHER_C1, TEACHER_C2,
                               kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(TEACHER_C2, 3 * (scale_factor ** 2),
                               kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x


# ============================================================
# WEIGHTS LOADER
# ============================================================

def resolve_weights(path_or_none):
    candidates = []
    if path_or_none:
        candidates.append(path_or_none)
    candidates.append(DEFAULT_WEIGHTS)
    candidates.append(FALLBACK_CKPT)

    for c in candidates:
        if c and os.path.isfile(c):
            return c

    raise FileNotFoundError(
        "Не найден файл с весами.\n"
        "Ожидался один из:\n  - "
        + "\n  - ".join(str(c) for c in candidates if c)
        + "\nСначала запустите train_teacher_upscaler.py"
    )


def load_model(weights_path):
    print(f"Loading weights: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu",
                      weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt and not any(
        k.startswith("conv") for k in ckpt.keys()
    ):
        if "ema" in ckpt and ckpt["ema"]:
            state = ckpt["ema"]
            source = "ema"
        else:
            state = ckpt["model"]
            source = "model"

        if "psnr" in ckpt:
            print(f"  checkpoint PSNR: {ckpt['psnr']:.3f} dB")
        if "epoch" in ckpt:
            print(f"  checkpoint epoch: {ckpt['epoch']}")
        print(f"  using state: {source}")
    else:
        state = ckpt
        print("  using state: raw state_dict")

    model = TeacherESPCN(scale_factor=SCALE_FACTOR)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    return model


# ============================================================
# BILINEAR BASELINE
# ============================================================

def upscale_bilinear(img: Image.Image):
    """
    Апскейл x2 билинейной интерполяцией. Возвращает PIL RGB.
    """
    img = img.convert("RGB")
    W, H = img.size
    return img.resize(
        (W * SCALE_FACTOR, H * SCALE_FACTOR),
        Image.Resampling.BILINEAR
    )


# ============================================================
# TILED UPSCALE (Teacher)
# ============================================================

@torch.no_grad()
def upscale_tile(model, tile_np):
    x = torch.from_numpy(tile_np)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()

    y = model(x).clamp(0.0, 1.0)

    y = y.squeeze(0).permute(1, 2, 0).contiguous()
    return y.numpy()


def make_weight_mask(h, w, overlap, scale):
    ov = overlap * scale
    if ov <= 0 or h <= 2 * ov or w <= 2 * ov:
        return np.ones((h, w, 1), dtype=np.float32)

    def ramp(n):
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        t = 0.5 - 0.5 * np.cos(np.pi * t)
        return t

    ax = np.ones(w, dtype=np.float32)
    ay = np.ones(h, dtype=np.float32)

    ramp_ov = ramp(ov)
    ax[:ov] = ramp_ov
    ax[-ov:] = ramp_ov[::-1]
    ay[:ov] = ramp_ov
    ay[-ov:] = ramp_ov[::-1]

    mask = ay[:, None] * ax[None, :]
    return mask[..., None].astype(np.float32)


def upscale_image_teacher(model, img: Image.Image,
                          tile=DEFAULT_TILE,
                          overlap=DEFAULT_OVERLAP,
                          verbose=True):
    img = img.convert("RGB")
    W, H = img.size

    lr = np.asarray(img, dtype=np.float32) / 255.0

    if H <= tile and W <= tile:
        sr = upscale_tile(model, lr)
        return Image.fromarray(
            (sr * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        )

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile должен быть больше overlap")

    out = np.zeros((H * SCALE_FACTOR, W * SCALE_FACTOR, 3),
                   dtype=np.float32)
    weight = np.zeros((H * SCALE_FACTOR, W * SCALE_FACTOR, 1),
                      dtype=np.float32)

    n_y = (H - tile + stride - 1) // stride + 1
    n_x = (W - tile + stride - 1) // stride + 1
    total = n_y * n_x
    done = 0

    t0 = time.time()

    for y0 in range(0, H, stride):
        y1 = min(y0 + tile, H)
        y0_eff = max(0, y1 - tile)

        for x0 in range(0, W, stride):
            x1 = min(x0 + tile, W)
            x0_eff = max(0, x1 - tile)

            tile_lr = lr[y0_eff:y1, x0_eff:x1]

            sr_tile = upscale_tile(model, tile_lr)

            h_sr, w_sr = sr_tile.shape[:2]
            ys = y0_eff * SCALE_FACTOR
            xs = x0_eff * SCALE_FACTOR

            mask = make_weight_mask(h_sr, w_sr,
                                    overlap=overlap,
                                    scale=SCALE_FACTOR)

            out[ys:ys + h_sr, xs:xs + w_sr] += sr_tile * mask
            weight[ys:ys + h_sr, xs:xs + w_sr] += mask

            done += 1
            if verbose and (done % 4 == 0 or done == total):
                elapsed = time.time() - t0
                print(f"    tiles {done}/{total} "
                      f"({elapsed:.1f}s)", end="\r", flush=True)

    if verbose:
        print()

    weight = np.maximum(weight, 1e-6)
    out = out / weight
    out = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    return Image.fromarray(out)


# ============================================================
# SIDE-BY-SIDE PREVIEW
# ============================================================

def _try_load_font(size=18):
    """
    Пытается загрузить системный шрифт; при неудаче — дефолтный.
    """
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_side_by_side(original: Image.Image,
                      bilinear: Image.Image,
                      teacher: Image.Image,
                      labels=("Bicubic LR (x1)", "Bilinear x2", "Teacher x2")):
    """
    Собирает 3 панели в одну картинку по горизонтали.

    Первая панель — оригинал, увеличенный бикубиком до HR-размера,
    чтобы у всех панелей была одинаковая высота и её можно было
    сравнить «в лоб» (это то, что мы подаём на вход сети,
    только показанное без пиксельных потерь).
    """
    bilinear = bilinear.convert("RGB")
    teacher = teacher.convert("RGB")

    HR_W, HR_H = bilinear.size

    # Приводим "оригинал" к тому же размеру, что и результаты
    # (визуально — это то, что апскейлер получает как LR, увеличенное).
    original_disp = original.convert("RGB").resize(
        (HR_W, HR_H), Image.Resampling.BICUBIC
    )

    pad = 8
    header_h = 34

    W = HR_W * 3 + pad * 4
    H = HR_H + header_h + pad * 2

    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    font = _try_load_font(20)

    panels = [original_disp, bilinear, teacher]

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = pad + i * (HR_W + pad)
        y = pad + header_h

        canvas.paste(panel, (x, y))

        # Подпись
        draw.text((x, pad + 4), label, fill=(230, 230, 230), font=font)

    return canvas


# ============================================================
# IO
# ============================================================

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp",
            "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP")


def find_images(directory):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(directory, "**", ext),
                           recursive=True)
    return sorted(set(paths))


def process_one(model, in_path, out_dir,
                tile, overlap,
                side_by_side=False,
                verbose=True):
    """
    Обрабатывает один файл:
      сохраняет bilinear и teacher результаты в out_dir,
      при желании — side-by-side превью.
    Возвращает (путь_bilinear, путь_teacher).
    """

    print(f"[+] {in_path}")

    img = Image.open(in_path).convert("RGB")
    W, H = img.size
    print(f"    input: {W}x{H} -> {W * SCALE_FACTOR}x{H * SCALE_FACTOR}")

    stem = Path(in_path).stem

    # --- 1. Bilinear baseline ---
    t0 = time.time()
    bilinear = upscale_bilinear(img)
    t_bilinear = time.time() - t0

    bilinear_path = os.path.join(out_dir, f"{stem}_bilinear_x2.png")
    bilinear.save(bilinear_path, "PNG")
    print(f"    bilinear: {bilinear_path} ({t_bilinear:.2f}s)")

    # --- 2. Teacher ---
    t0 = time.time()
    teacher = upscale_image_teacher(
        model, img,
        tile=tile, overlap=overlap,
        verbose=verbose
    )
    t_teacher = time.time() - t0

    teacher_path = os.path.join(out_dir, f"{stem}_teacher_x2.png")
    teacher.save(teacher_path, "PNG")
    print(f"    teacher:  {teacher_path} ({t_teacher:.2f}s)")

    # --- 3. Optional side-by-side ---
    if side_by_side:
        try:
            preview = make_side_by_side(img, bilinear, teacher)
            preview_path = os.path.join(
                out_dir, f"{stem}_compare.png"
            )
            preview.save(preview_path, "PNG")
            print(f"    compare:  {preview_path}")
        except Exception as e:
            print(f"    side-by-side failed: {e}")

    del img, bilinear, teacher
    gc.collect()

    return bilinear_path, teacher_path


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CPU x2 upscaler: Bilinear vs TeacherESPCN"
    )

    parser.add_argument(
        "input",
        help="Входной файл изображения или папка"
    )

    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Выходная папка (или файл, если вход — одиночный файл; "
             "в этом случае имя файла используется как префикс). "
             "Если не указано — рядом с входом."
    )

    parser.add_argument(
        "--weights",
        default=None,
        help=f"Путь к .pth с весами. "
             f"По умолчанию: {DEFAULT_WEIGHTS} или {FALLBACK_CKPT}"
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=DEFAULT_TILE,
        help=f"Размер тайла в LR-пикселях (по умолчанию {DEFAULT_TILE})"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Перекрытие тайлов в LR-пикселях "
             f"(по умолчанию {DEFAULT_OVERLAP})"
    )

    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Дополнительно сохранять картинку-сравнение "
             "(LR / bilinear / teacher) в одну строку"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Число CPU-потоков (0 = авто)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Меньше вывода"
    )

    args = parser.parse_args()

    global NUM_THREADS
    NUM_THREADS = args.threads

    setup_cpu()

    weights_path = resolve_weights(args.weights)
    model = load_model(weights_path)

    verbose = not args.quiet

    in_path = Path(args.input)

    # ---- Single file ----
    if in_path.is_file():
        if args.output:
            out = Path(args.output)
            # Если указан путь с расширением .png — трактуем как
            # директорию? Нет: у нас два файла на выходе, поэтому
            # всегда используем директорию.
            if out.suffix.lower() in (".png", ".jpg", ".jpeg",
                                      ".bmp", ".webp"):
                out_dir = out.parent
                out_dir.mkdir(parents=True, exist_ok=True)
            else:
                out_dir = out
                out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = in_path.parent

        process_one(
            model,
            str(in_path),
            str(out_dir),
            args.tile,
            args.overlap,
            side_by_side=args.side_by_side,
            verbose=verbose
        )

        print()
        print("[OK]")
        print(f"Выходная папка: {out_dir}")
        return

    # ---- Directory ----
    if in_path.is_dir():
        paths = find_images(str(in_path))

        if not paths:
            print(f"В папке нет изображений: {in_path}")
            return

        if args.output:
            out_dir = Path(args.output)
        else:
            out_dir = in_path / "upscaled_x2"

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Найдено изображений: {len(paths)}")
        print(f"Выходная папка: {out_dir}")
        print()

        t0 = time.time()

        for i, p in enumerate(paths, 1):
            print(f"--- [{i}/{len(paths)}] ---")
            try:
                process_one(
                    model, p, str(out_dir),
                    args.tile, args.overlap,
                    side_by_side=args.side_by_side,
                    verbose=verbose
                )
            except Exception as e:
                print(f"    ОШИБКА: {e}")

            print()

        print(f"Готово за {time.time() - t0:.1f}s")
        print("[OK]")
        return

    print(f"Не найден вход: {in_path}")


if __name__ == "__main__":
    main()
