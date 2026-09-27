# converter.py — универсальный под твои модели
# Поддерживает:
#   - TinyFSRStudent (conv_in / dw / mix / to_residual)
#   - MiniESPCN      (conv1: 3->32, conv2: 32->16, conv3: 16->12)
#   - TeacherESPCN   (conv1: 3->40, conv2: 40->28, conv3: 28->12)
# Автоматически определяет архитектуру по ключам и формам тензоров.
# Работает и с FP32, и с FP16 чекпоинтами.

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# АРХИТЕКТУРЫ
# ============================================================
class TinyFSRStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.mix = nn.Conv2d(8, 8, 1)
        self.to_residual = nn.Conv2d(8, 12, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, lr):
        x = F.relu(self.conv_in(lr))
        x = F.relu(self.dw(x))
        x = F.relu(self.mix(x))
        r = self.shuffle(self.to_residual(x))
        base = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        return (base + r).clamp(0.0, 1.0), r


class _ESPCN(nn.Module):
    """Общий ESPCN-каркас, параметры подставляются из чекпоинта."""
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 5, padding=2)
        self.conv3 = nn.Conv2d(c2, 12, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pixel_shuffle(self.conv3(x))


# ============================================================
# АВТООПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ
# ============================================================
def detect_architecture(state):
    keys = set(state.keys())

    if "conv_in.weight" in keys:
        return "TinyFSRStudent", TinyFSRStudent()

    if "conv1.weight" in keys and "conv2.weight" in keys:
        c1 = state["conv1.weight"].shape[0]
        c2 = state["conv2.weight"].shape[0]
        name = "TeacherESPCN" if c1 == 40 else "MiniESPCN"
        return f"{name} (c1={c1}, c2={c2})", _ESPCN(c1, c2)

    raise ValueError(f"Не удалось распознать архитектуру. Ключи: {sorted(keys)}")


# ============================================================
# ЭКСПОРТ В HLSL (FP16 или FP32 автоматически)
# ============================================================
def export_weights_to_hlsl(state_dict, filename, arch_name):
    first = next(iter(state_dict.values()))
    use_fp16 = first.dtype == torch.float16
    hlsl_type = "half" if use_fp16 else "float"
    suffix = "h" if use_fp16 else "f"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"// Architecture: {arch_name}\n")
        f.write(f"// Weight type: {hlsl_type}\n")
        f.write("// Auto-generated from .pth\n\n")

        for key, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy()
            name = key.replace(".", "_")

            if key.endswith("weight"):
                shape = "".join(f"[{d}]" for d in arr.shape)
                f.write(f"static const {hlsl_type} {name}{shape} = {{\n")
                flat = arr.flatten()
                for i, v in enumerate(flat):
                    f.write(f"{float(v):+.6f}{suffix}, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")

            elif key.endswith("bias"):
                f.write(f"static const {hlsl_type} {name}[{arr.shape[0]}] = {{\n")
                for i, v in enumerate(arr):
                    f.write(f"{float(v):+.6f}{suffix}, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")

    print(f"[OK] HLSL ({hlsl_type}): {filename}")


# ============================================================
# MAIN
# ============================================================
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    pth_path = args[0] if len(args) >= 1 else "teacher2_best_fp16.pth"
    out_path = args[1] if len(args) >= 2 else "weights.hlsl"
    use_raw = "--raw" in flags

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Не найден файл: {pth_path}")

    print(f"Loading: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")

    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        if use_raw:
            state = ckpt["model"]
            print("  using: 'model' (raw)")
        else:
            state = ckpt.get("ema", ckpt.get("model"))
            print("  using: 'ema'" if "ema" in ckpt else "  using: 'model'")
    else:
        state = ckpt
        print("  using: raw state_dict")

    state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    arch_name, model = detect_architecture(state)
    print(f"  architecture: {arch_name}")

    ckpt_dtype = next(iter(state.values())).dtype
    if ckpt_dtype == torch.float16:
        model = model.half()
    model.load_state_dict(state, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    with torch.no_grad():
        dummy = torch.rand(1, 3, 32, 32)
        if ckpt_dtype == torch.float16:
            dummy = dummy.half()
        out = model(dummy)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (1, 3, 64, 64), out.shape
    print("  sanity check: (1,3,32,32) -> (1,3,64,64)")

    export_weights_to_hlsl(model.state_dict(), out_path, arch_name)


if __name__ == "__main__":
    main()
