# converter.py
# Конвертирует .pth (TeacherESPCN из train_teacher2_distill.py) -> .hlsl
#
# Архитектура TeacherESPCN:
#   conv1:       3  -> 40  (3x3, pad 1) + ReLU
#   conv2:       40 -> 28  (5x5, pad 2) + ReLU
#   conv3:       28 -> 12  (3x3, pad 1)
#   PixelShuffle(2)  -> 3 канала x2 upscale
#
# Поддерживаются оба варианта чекпоинта:
#   - {"model": state_dict, ...}   (raw weights)
#   - {"ema":   state_dict, ...}   (EMA weights)  <- по умолчанию берём EMA
#   - чистый state_dict
#
# Использование:
#   python converter.py teacher2_best.pth teacher2_weights.hlsl
#   python converter.py teacher2_best.pth teacher2_weights.hlsl --raw   # взять model, не ema

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# АРХИТЕКТУРА (копия из train_teacher2_distill.py)
# ============================================================
TEACHER_C1 = 40
TEACHER_C2 = 28


class TeacherESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        assert scale_factor == 2
        self.conv1 = nn.Conv2d(3, TEACHER_C1, 3, padding=1)
        self.conv2 = nn.Conv2d(TEACHER_C1, TEACHER_C2, 5, padding=2)
        self.conv3 = nn.Conv2d(TEACHER_C2, 3 * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pixel_shuffle(self.conv3(x))


# ============================================================
# ЭКСПОРТ В HLSL
# ============================================================
def export_weights_to_hlsl(state_dict, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("// TeacherESPCN (Teacher #2) — ESPCN x2\n")
        f.write("// conv1: 3->40 (3x3, pad 1) + ReLU\n")
        f.write("// conv2: 40->28 (5x5, pad 2) + ReLU\n")
        f.write("// conv3: 28->12 (3x3, pad 1)\n")
        f.write("// PixelShuffle(2) -> 3 channels x2 upscale\n")
        f.write("//\n")
        f.write("// Forward (compute shader steps):\n")
        f.write("//   1. x1 = relu(conv3x3(lr,   conv1_weight, conv1_bias))   // 40 ch\n")
        f.write("//   2. x2 = relu(conv5x5(x1,   conv2_weight, conv2_bias))   // 28 ch\n")
        f.write("//   3. x3 =       conv3x3(x2,  conv3_weight, conv3_bias)    // 12 ch\n")
        f.write("//   4. out = pixel_shuffle_2x(x3)                            // 3 ch, 2x size\n")
        f.write("//   5. out = clamp(out, 0, 1)\n\n")

        for key, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy()
            name = key.replace(".", "_")

            if key.endswith("weight"):
                shape = "".join(f"[{d}]" for d in arr.shape)
                f.write(f"static const float {name}{shape} = {{\n")
                flat = arr.flatten()
                for i, v in enumerate(flat):
                    f.write(f"{v:+.8f}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")

            elif key.endswith("bias"):
                f.write(f"static const float {name}[{arr.shape[0]}] = {{\n")
                for i, v in enumerate(arr):
                    f.write(f"{v:+.8f}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")

    print(f"[OK] HLSL weights: {filename}")


# ============================================================
# MAIN
# ============================================================
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    pth_path = args[0] if len(args) >= 1 else "teacher_best_state_dict.pth"
    out_path = args[1] if len(args) >= 2 else "weights.hlsl"
    use_raw = "--raw" in flags   # по умолчанию берём EMA

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Не найден файл весов: {pth_path}")

    print(f"Loading: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")

    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        if use_raw:
            state = ckpt["model"]
            print("  using: 'model' (raw)")
        else:
            state = ckpt.get("ema", ckpt.get("model"))
            print("  using: 'ema'" if "ema" in ckpt else "  using: 'model' (no ema found)")
        if "psnr_ema" in ckpt:
            print(f"  saved PSNR EMA: {ckpt['psnr_ema']:.3f} dB")
        if "psnr_raw" in ckpt:
            print(f"  saved PSNR RAW: {ckpt['psnr_raw']:.3f} dB")
        if "psnr" in ckpt:
            print(f"  saved PSNR:     {ckpt['psnr']:.3f} dB")
    else:
        state = ckpt
        print("  using: raw state_dict")

    model = TeacherESPCN(scale_factor=2)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    with torch.no_grad():
        dummy = torch.rand(1, 3, 32, 32)
        out = model(dummy)
        assert out.shape == (1, 3, 64, 64), out.shape
    print("Sanity check passed: (1,3,32,32) -> (1,3,64,64)")

    export_weights_to_hlsl(model.state_dict(), out_path)


if __name__ == "__main__":
    main()
