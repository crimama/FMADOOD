"""Convert pinned HF DINOv3-B weights for RAD's native DINOv3 loader.

Run only in an isolated RAD checkout. Conversion writes a new checkpoint and
requires a numerical final-feature check before it may be used by RAD.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

if not hasattr(torch.utils._pytree, "register_pytree_node"):
    def _register_pytree_node(typ, flatten_fn, unflatten_fn, **_kwargs):
        return torch.utils._pytree._register_pytree_node(typ, flatten_fn, unflatten_fn)

    torch.utils._pytree.register_pytree_node = _register_pytree_node

from transformers import AutoModel

from dinov3.hub.backbones import dinov3_vitb16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.hf_dir / "model.safetensors"
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        hf = {key: handle.get_tensor(key) for key in handle.keys()}
    model = dinov3_vitb16(pretrained=False)
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def take(dst: str, src: str) -> None:
        tensor = hf[src]
        if dst == "mask_token" and tensor.ndim == 3:
            tensor = tensor.squeeze(1)
        if tensor.shape != target[dst].shape:
            raise ValueError(f"Shape mismatch: {src} {tensor.shape} -> {dst} {target[dst].shape}")
        mapped[dst] = tensor
        used.add(src)

    for dst, src in (
        ("cls_token", "embeddings.cls_token"),
        ("storage_tokens", "embeddings.register_tokens"),
        ("mask_token", "embeddings.mask_token"),
        ("patch_embed.proj.weight", "embeddings.patch_embeddings.weight"),
        ("patch_embed.proj.bias", "embeddings.patch_embeddings.bias"),
        ("norm.weight", "norm.weight"),
        ("norm.bias", "norm.bias"),
    ):
        take(dst, src)

    for layer in range(12):
        src = f"layer.{layer}"
        dst = f"blocks.{layer}"
        for suffix in ("weight", "bias"):
            for norm in ("norm1", "norm2"):
                take(f"{dst}.{norm}.{suffix}", f"{src}.{norm}.{suffix}")
            take(f"{dst}.attn.proj.{suffix}", f"{src}.attention.o_proj.{suffix}")
            take(f"{dst}.mlp.fc1.{suffix}", f"{src}.mlp.up_proj.{suffix}")
            take(f"{dst}.mlp.fc2.{suffix}", f"{src}.mlp.down_proj.{suffix}")
        take(f"{dst}.ls1.gamma", f"{src}.layer_scale1.lambda1")
        take(f"{dst}.ls2.gamma", f"{src}.layer_scale2.lambda1")
        qkv = [hf[f"{src}.attention.{part}_proj.weight"] for part in ("q", "k", "v")]
        mapped[f"{dst}.attn.qkv.weight"] = torch.cat(qkv, dim=0)
        used.update(f"{src}.attention.{part}_proj.weight" for part in ("q", "k", "v"))
        qb = hf[f"{src}.attention.q_proj.bias"]
        vb = hf[f"{src}.attention.v_proj.bias"]
        mapped[f"{dst}.attn.qkv.bias"] = torch.cat((qb, torch.zeros_like(qb), vb))
        used.update((f"{src}.attention.q_proj.bias", f"{src}.attention.v_proj.bias"))

    # Native RoPE periods and the key-bias mask are deterministic nontrainable buffers.
    for key in target.keys() - mapped.keys():
        if key == "rope_embed.periods":
            mapped[key] = target[key]
        elif key.endswith(".attn.qkv.bias_mask"):
            width = target[key].numel() // 3
            mask = torch.ones_like(target[key])
            mask[width : 2 * width] = 0
            mapped[key] = mask

    missing = set(target) - set(mapped)
    unused = set(hf) - used
    if missing or unused:
        raise ValueError(f"Incomplete conversion: missing={sorted(missing)}, unused={sorted(unused)}")
    model.load_state_dict(mapped, strict=True)

    hf_model = AutoModel.from_pretrained(str(args.hf_dir), local_files_only=True).eval()
    model.eval()
    torch.manual_seed(20260930)
    image = torch.rand(1, 3, 224, 224)
    with torch.inference_mode():
        expected = hf_model(pixel_values=image).last_hidden_state[:, 5:, :]
        actual = model.get_intermediate_layers(image, n=[11], norm=True)[0]
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError(
            "Nonfinite features: "
            f"HF={int(torch.isfinite(expected).sum())}/{expected.numel()}, "
            f"native={int(torch.isfinite(actual).sum())}/{actual.numel()}"
        )
    max_error = float((expected - actual).abs().max())
    relative_error = float((expected - actual).norm() / expected.norm())
    print(f"Final patch tokens: max_abs_error={max_error:.8g}, relative_l2_error={relative_error:.8g}")
    if max_error > 1e-4 or relative_error > 1e-5:
        raise ValueError("Native RAD and HF DINOv3-B features are not numerically equivalent")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, args.output)
    print(f"Saved strict-loadable native checkpoint to {args.output}")


if __name__ == "__main__":
    main()
