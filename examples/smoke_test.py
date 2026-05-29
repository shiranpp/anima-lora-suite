"""
Smoke test: build a synthetic Anima-shaped LoRA, save it, load it through
the editor with a few different configs, verify the right tensors survive.
"""

import os
import sys
import tempfile

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    load_lora_state_dict,
    detect_architecture,
    analyze_lora,
    edit_lora,
    save_lora_state_dict,
    ANIMA_NUM_BLOCKS,
)
from core.detect import summarize_keys, extract_block_info
from core.editor import EditConfig
from core.presets import preset_block_set
from core.compress import (
    compress_state_dict,
    state_dict_nbytes,
    downcast_state_dict,
)


def make_fake_anima_lora(num_blocks=ANIMA_NUM_BLOCKS, rank=4, dim=64):
    """Build a state_dict shaped like a typical kohya/sd-scripts Anima LoRA."""
    sd = {}
    # 28 block layers, each with a few submodules
    for i in range(num_blocks):
        for sub in ("attn_q_proj", "attn_k_proj", "attn_v_proj", "attn_out_proj", "mlp_fc1", "mlp_fc2"):
            base = f"lora_unet_blocks_{i}_{sub}"
            # Vary magnitudes so we can test the impact analyzer
            scale = 1.0 + 0.2 * i
            sd[f"{base}.lora_down.weight"] = torch.randn(rank, dim) * 0.1
            sd[f"{base}.lora_up.weight"]   = torch.randn(dim, rank) * 0.1 * scale
            sd[f"{base}.alpha"]            = torch.tensor(float(rank))
    # LLMAdapter weights
    for sub in ("layer_0_q_proj", "layer_0_k_proj", "layer_0_mlp"):
        base = f"lora_unet_llm_adapter_{sub}"
        sd[f"{base}.lora_down.weight"] = torch.randn(rank, dim) * 0.1
        sd[f"{base}.lora_up.weight"]   = torch.randn(dim, rank) * 0.1
        sd[f"{base}.alpha"]            = torch.tensor(float(rank))
    # A couple of "other" tensors (final layer, time embed)
    for name in ("lora_unet_final_layer_linear", "lora_unet_time_embed_proj"):
        sd[f"{name}.lora_down.weight"] = torch.randn(rank, dim) * 0.1
        sd[f"{name}.lora_up.weight"]   = torch.randn(dim, rank) * 0.1
        sd[f"{name}.alpha"]            = torch.tensor(float(rank))
    return sd


def main():
    print("=" * 72)
    print(" Anima LoRA Editor — smoke test")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "fake_anima.safetensors")
        sd = make_fake_anima_lora()
        save_file({k: v.contiguous() for k, v in sd.items()}, in_path)
        print(f"\n[+] Wrote fake LoRA ({len(sd)} tensors) -> {in_path}")

        # Re-load through the public API
        loaded = load_lora_state_dict(in_path)
        keys = list(loaded.keys())

        # --- detect ---
        arch = detect_architecture(keys)
        print(f"[+] Detected architecture: {arch}")
        assert arch == "ANIMA", f"expected ANIMA, got {arch}"

        # --- summarize ---
        summ = summarize_keys(keys)
        print(f"[+] Summary: {summ['num_blocks']} blocks, "
              f"LLMAdapter={summ['has_llm_adapter']}, "
              f"other={summ['other_count']}")
        assert summ["num_blocks"] == 28
        assert summ["has_llm_adapter"] is True
        assert summ["other_count"] > 0  # the final/time tensors

        # --- analyze ---
        imp = analyze_lora(loaded)
        # Block 27 had the highest scale factor, so it should dominate
        strongest = max(imp["block_norm"].items(), key=lambda x: x[1])
        print(f"[+] Strongest block: {strongest[0]} ({strongest[1]}%)")
        assert strongest[0] == 27

        # --- edit: Late Only preset (blocks 19..27) ---
        late = preset_block_set("Late Only (19-27)")
        cfg = EditConfig(enabled_blocks=late, llm_adapter_enabled=True, other_enabled=True)
        new_sd, info = edit_lora(loaded, cfg)
        print(f"[+] Late-only preset: kept {len(info['blocks_kept'])} blocks, "
              f"dropped {len(info['blocks_dropped'])}")
        assert set(info["blocks_kept"]) == late
        # Confirm no key from a disabled block leaked through
        for k in new_sd:
            tag, n = extract_block_info(k)
            if tag == "block":
                assert n in late, f"block {n} should have been dropped"

        # --- edit: drop LLMAdapter, scale block 25 to 2.0 ---
        cfg = EditConfig(
            enabled_blocks=set(range(28)),
            block_strengths={25: 2.0},
            llm_adapter_enabled=False,
            other_enabled=True,
            global_strength=1.0,
        )
        new_sd, info = edit_lora(loaded, cfg)
        print(f"[+] Drop-LLM + scale block 25 x2: kept {info['output_tensor_count']} tensors")
        # No llm_adapter keys should remain
        for k in new_sd:
            assert "llm_adapter" not in k.lower(), f"LLMAdapter leaked: {k}"
        # block 25's "up" tensors should be scaled, "down" unchanged
        b25_up_key = "lora_unet_blocks_25_attn_q_proj.lora_up.weight"
        b25_down_key = "lora_unet_blocks_25_attn_q_proj.lora_down.weight"
        assert torch.allclose(new_sd[b25_up_key], loaded[b25_up_key] * 2.0)
        assert torch.allclose(new_sd[b25_down_key], loaded[b25_down_key])
        print("    block 25 up = original * 2.0  ✓")
        print("    block 25 down unchanged       ✓")

        # --- edit: write the file and reload to confirm ---
        out_path = os.path.join(td, "edited.safetensors")
        save_lora_state_dict(new_sd, out_path,
                             metadata={"anima_lora_editor": "1"})
        roundtrip = load_file(out_path)
        assert set(roundtrip.keys()) == set(new_sd.keys())
        for k in new_sd:
            assert torch.allclose(new_sd[k], roundtrip[k]), f"roundtrip mismatch on {k}"
        print(f"[+] Roundtrip save/load OK ({len(roundtrip)} tensors)")

        # --- edit: All Off preset ---
        cfg = EditConfig(
            enabled_blocks=preset_block_set("All Off"),
            llm_adapter_enabled=False,
            other_enabled=False,
        )
        new_sd, info = edit_lora(loaded, cfg)
        print(f"[+] All-Off preset: kept {info['output_tensor_count']} tensors")
        assert info["output_tensor_count"] == 0

        # --- detect non-Anima LoRA so the warning fires ---
        flux_keys = ["transformer.single_transformer_blocks.0.attn.q.lora_A.weight"]
        arch = detect_architecture(flux_keys)
        print(f"[+] Non-Anima detection: '{arch}' (should be FLUX)")
        assert arch == "FLUX"

        # --- compress: fp16 downcast halves the payload, near-lossless ---
        fp32 = make_fake_anima_lora()
        comp16, cinfo = compress_state_dict(fp32, dtype="fp16")
        print(f"[+] fp16 downcast: {state_dict_nbytes(fp32)} -> "
              f"{state_dict_nbytes(comp16)} bytes (ratio {cinfo['ratio']:.3f})")
        assert abs(cinfo["ratio"] - 0.5) < 0.01, "fp16 should ~halve the payload"
        assert cinfo["dtype_after"] == "float16"
        # .alpha scalars are left untouched (kept full precision)
        for k, v in comp16.items():
            if k.endswith(".alpha"):
                assert v.dtype == torch.float32, f"alpha was downcast: {k}"
            elif v.is_floating_point():
                assert v.dtype == torch.float16, f"weight not fp16: {k}"

        # --- compress: SVD reduction on a genuinely low-rank module ---
        # Build a rank-32 pair whose true update only needs rank 8, then ask for
        # rank 8 back — reconstruction should be near-exact.
        out_d, in_d, true_r, stored_r = 96, 96, 8, 32
        U = torch.randn(out_d, true_r)
        V = torch.randn(true_r, in_d)
        delta = U @ V  # rank-8 ground truth
        # express it as a rank-32 pair (pad with tiny noise rows)
        up = torch.cat([U, torch.randn(out_d, stored_r - true_r) * 1e-4], dim=1)
        down = torch.cat([V, torch.randn(stored_r - true_r, in_d) * 1e-4], dim=0)
        svd_sd = {
            "lora_unet_blocks_0_attn_q_proj.lora_down.weight": down,
            "lora_unet_blocks_0_attn_q_proj.lora_up.weight": up,
            "lora_unet_blocks_0_attn_q_proj.alpha": torch.tensor(float(stored_r)),
        }
        red, rinfo = compress_state_dict(svd_sd, svd_rank=true_r)
        d2 = red["lora_unet_blocks_0_attn_q_proj.lora_down.weight"]
        a2 = float(red["lora_unet_blocks_0_attn_q_proj.alpha"])
        recon = red["lora_unet_blocks_0_attn_q_proj.lora_up.weight"].float() @ (
            d2.float() * (a2 / d2.shape[0]))
        rel_err = (recon - delta).norm().item() / delta.norm().item()
        print(f"[+] SVD rank {stored_r}->{rinfo['svd']['rank_after']}, "
              f"recon rel-err {rel_err:.4f}")
        assert d2.shape[0] == true_r, "rank not reduced to target"
        assert rel_err < 0.02, f"SVD reconstruction too lossy: {rel_err}"
        assert rinfo["svd"]["pairs_reduced"] == 1

        # --- compress: fp8 if this torch build supports it ---
        if hasattr(torch, "float8_e4m3fn"):
            comp8, c8 = compress_state_dict(fp32, dtype="fp8_e4m3fn")
            print(f"[+] fp8 downcast: ratio {c8['ratio']:.3f}")
            assert abs(c8["ratio"] - 0.25) < 0.02, "fp8 should ~quarter the payload"
        else:
            print("[i] fp8 not available in this torch build — skipped")

    print("\n  All checks passed ✓\n")


if __name__ == "__main__":
    main()
