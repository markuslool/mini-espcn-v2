# quantize_fp16.py
# Конвертирует .pth (float32) -> .pth (float16)
#
# Использование:
#   python quantize_fp16.py teacher2_best.pth teacher2_best_fp16.pth
#   python quantize_fp16.py                      # по умолчанию берёт teacher2_best.pth

import os
import sys
import torch


def quantize_state_dict(state):
    return {k: v.half() if v.is_floating_point() else v for k, v in state.items()}


def main():
    src = sys.argv[1] if len(sys.argv) >= 2 else "teacher_best_state_dict.pth"
    dst = sys.argv[2] if len(sys.argv) >= 3 else src.replace(".pth", "_fp16.pth")

    if not os.path.exists(src):
        raise FileNotFoundError(f"Не найден файл: {src}")

    print(f"Loading: {src}")
    ckpt = torch.load(src, map_location="cpu")

    # Поддерживаем три варианта:
    #   1) {"model": ..., "ema": ..., ...}  — чекпоинт Teacher #2
    #   2) {"model": ...}                    — чекпоинт студента
    #   3) чистый state_dict
    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        for key in ("model", "ema"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt[key] = quantize_state_dict(ckpt[key])
                print(f"  quantized '{key}' -> float16")

    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        ckpt = quantize_state_dict(ckpt)
        print("  quantized raw state_dict -> float16")

    else:
        raise ValueError(f"Неизвестный формат чекпоинта: {type(ckpt)}")

    torch.save(ckpt, dst)

    size_src = os.path.getsize(src) / 1024 / 1024
    size_dst = os.path.getsize(dst) / 1024 / 1024
    print(f"[OK] {dst}")
    print(f"     {size_src:.2f} MB -> {size_dst:.2f} MB "
          f"({size_dst / size_src * 100:.1f}%)")


if __name__ == "__main__":
    main()
