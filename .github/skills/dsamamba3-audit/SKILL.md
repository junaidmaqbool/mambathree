---
name: dsamamba3-audit
description: "Audit DSAmamba3.py (VSSM image classifier) against the official Mamba3 definition. Use when: verifying DSAmamba3 is architecturally correct, checking Mamba3 constructor kwargs, validating anemia binary classification pipeline, confirming SISO vs MIMO config, checking tensor shapes through PatchEmbed → VSSLayer → head, or debugging mismatches between DSAmamba3.py and mamba3.py."
argument-hint: "Describe what to check: constructor params, forward shapes, SISO/MIMO, classification head, or full end-to-end audit"
---

# DSA-Mamba3 Audit Skill

Systematic procedure for verifying that `DSAmamba3.py` correctly implements a VSSM-style image classifier on top of the official `Mamba3` SSM, suitable for binary anemia classification (anemic / non-anemic) from microscopy or peripheral-blood-smear images.

## When to Use
- Verifying DSAmamba3.py against official mamba3.py definitions
- Debugging shape mismatches or import errors in the model
- Checking that the Anemia classification head and loss are correct
- Confirming that SISO mode (`is_mimo=False`) is used so no TileLang is required
- Validating that all Mamba3 constructor kwargs are valid and compatible
- Pre-training sanity check on a Kaggle T4 / Colab GPU

## Source Files to Read

| File | Role |
|------|------|
| `mamba_ssm/models/DSAmamba3.py` | The model being audited |
| `mamba_ssm/modules/mamba3.py` | Official Mamba3 definition (ground truth) |
| `mamba_ssm/modules/mamba_simple.py` | Mamba1 reference (SSM baseline) |
| `mamba_ssm/modules/mamba2.py` | Mamba2 reference (multi-head SSM) |

---

## Audit Procedure

### Step 1 — Read Source Files in Parallel

Read all four files listed above. Focus on:
- `Mamba3.__init__` parameter list and their types/defaults
- `Mamba3.forward` expected input shape and return shape
- `VSSM.__init__` and `VSSM.forward_features` in DSAmamba3.py

### Step 2 — Check Mamba3 Constructor Kwargs

Verify every kwarg in `Mamba3ImageBlock` matches `Mamba3.__init__`:

| Kwarg in Mamba3ImageBlock | Correct Mamba3 param | Notes |
|--------------------------|----------------------|-------|
| `d_model=hidden_dim` | `d_model` | dim of the token embedding at that stage |
| `d_state=d_state` | `d_state` | SSM state size; 64 or 128; reduce to 32 on small GPU |
| `expand=expand` | `expand` | channel expansion; default 2 |
| `headdim=headdim` | `headdim` | must satisfy `(expand * d_model) % headdim == 0` |
| `ngroups=1` | `ngroups` | num_bc_heads; 1 is correct for standard SISO |
| `rope_fraction=rope_fraction` | `rope_fraction` | must be 0.5 or 1.0 (assertion in Mamba3) |
| `is_mimo=False` | `is_mimo` | **must be False** on Kaggle T4; True requires TileLang |
| `chunk_size=chunk_size` | `chunk_size` | 64 recommended for SISO |
| `layer_idx=layer_idx` | `layer_idx` | block index; used only for inference cache (no-op here) |

**headdim divisibility check** — for default config `dims=[96,192,384,768]`, `expand=2`, `headdim=64`:
- Stage 0: d_inner = 192, nheads = 3 ✓
- Stage 1: d_inner = 384, nheads = 6 ✓
- Stage 2: d_inner = 768, nheads = 12 ✓
- Stage 3: d_inner = 1536, nheads = 24 ✓

**rope_fraction / num_rope_angles check** — with `d_state=64`, `rope_fraction=0.5`:
- `split_tensor_size = int(64 * 0.5) = 32`
- `num_rope_angles = 32 // 2 = 16 > 0` ✓

### Step 3 — Verify Tensor Shapes Through the Pipeline

Trace a `(B, 3, 224, 224)` input (standard anemia microscopy size):

```
Input     (B, 3, 224, 224)
  ↓ PatchEmbed2D(patch_size=4)
         (B, 56, 56, 96)        # 224/4 = 56
  ↓ VSSLayer[0] depth=2, no downsample? NO — downsample=PatchMerging2D
         (B, 56, 56, 96)  → blocks  →  (B, 28, 28, 192)
  ↓ VSSLayer[1]
         (B, 28, 28, 192) → blocks  →  (B, 14, 14, 384)
  ↓ VSSLayer[2]
         (B, 14, 14, 384) → blocks  →  (B,  7,  7, 768)
  ↓ VSSLayer[3] (no downsample — last stage)
         (B,  7,  7, 768) → blocks  →  (B,  7,  7, 768)
  ↓ LayerNorm(768)
  ↓ permute(0,3,1,2) → (B, 768, 7, 7)
  ↓ AdaptiveAvgPool2d(1) → (B, 768, 1, 1)
  ↓ flatten(1) → (B, 768)
  ↓ Linear(768, num_classes) → (B, 2)
```

**Inside Mamba3ImageBlock.forward:**
```
x     (B, H, W, C)
  → norm → x_norm
  → view(B, H*W, C)     ← Mamba3 input: (batch, seqlen, hidden_dim) ✓
  → mamba_fwd(x_seq)    → (B, L, C)
  → if bidirectional: mamba_bwd(x_seq.flip(1)).flip(1) → (B, L, C)
  → average → view(B, H, W, C) → residual add
```

### Step 4 — Confirm Mamba3's Forward Contract

`Mamba3.forward(u)` where `u: (batch, seqlen, d_model)` returns `(batch, seqlen, d_model)`.
- Input: `x_seq = (B, H*W, hidden_dim)` ✓
- Output: same shape → reshapes back to `(B, H, W, C)` ✓

### Step 5 — Verify SISO Mode (No TileLang Required)

In `Mamba3.forward`, the SISO branch calls `mamba3_siso_combined` (Triton kernel).
The MIMO branch calls `mamba3_mimo_combined` (TileLang kernel, not available on Kaggle T4).

- `is_mimo=False` in DSAmamba3 → SISO branch only ✓
- TileLang import is guarded: `mamba3_mimo_combined = None` if TileLang missing ✓
- MIMO assertion fires only if `is_mimo=True`: `assert mamba3_mimo_combined is not None` ✓

### Step 6 — Check Anemia Classification Head

| Criterion | Check |
|-----------|-------|
| `num_classes=2` (default) | Binary: 0=non-anemic, 1=anemic |
| `head = nn.Linear(dims[-1], num_classes)` | Outputs 2 logits ✓ |
| Loss compatible: `nn.CrossEntropyLoss()` | Expects `(B, 2)` logits + `(B,)` long labels ✓ |
| Input dtype: float32 RGB images | Standard for microscopy/smear images ✓ |
| `in_chans=3` | RGB; change to 1 for grayscale stains |

For **Hb-level regression** (continuous hemoglobin estimation), change:
- `num_classes=1` → `head` outputs 1 scalar
- Loss: `nn.MSELoss()` or `nn.HuberLoss()`

### Step 7 — Verify `no_weight_decay` Coverage

`VSSM.no_weight_decay()` returns `{"dt_bias", "D", "B_bias", "C_bias"}`.

These correspond to Mamba3's special parameters — the optimizer wrapper must filter by parameter name suffix. Standard timm / train.py pattern:
```python
params_wd, params_no_wd = [], []
for name, p in model.named_parameters():
    if any(name.endswith(k) for k in model.no_weight_decay()):
        params_no_wd.append(p)
    else:
        params_wd.append(p)
```

### Step 8 — Layer-idx Offset Check (Known Limitation)

Each `VSSLayer` resets `layer_idx` from `0..depth-1`. Across 4 stages this means multiple Mamba3 modules share the same `layer_idx` values. This is **harmless for image classification** (inference cache is never used), but would be incorrect if you ever attach `inference_params` for sequential generation. If that use case arises, use a global counter:
```python
global_layer_idx = sum(depths[:i]) + j
```

### Step 9 — Verify Import in train.py

Make sure `train.py` (or the Kaggle notebook) imports:
```python
from mamba_ssm.models.DSAmamba3 import VSSM as dsamamba
```
not the older DSA-Mamba (VMamba) variant.

---

## Common Issues and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `AssertionError: Fails to import Mamba-3 MIMO kernels` | `is_mimo=True` without TileLang | Set `is_mimo=False` |
| `AssertionError: d_inner % headdim != 0` | Wrong `headdim` for the stage dim | Pick `headdim` that divides `expand * dims[0]` |
| `AssertionError: rope_fraction must be 0.5 or 1.0` | Non-standard value | Use `rope_fraction=0.5` |
| `assert num_rope_angles > 0` | `d_state` too small with `rope_fraction=0.5` | Use `d_state >= 4` |
| `RuntimeError: Expected 3D input` | Passing `(B, C, H, W)` to Mamba3 | Use `x.view(B, H*W, C)` first |
| OOM on Kaggle T4 | `d_state=128` or `dims=[128,256,512,1024]` | Reduce `d_state=32`, `dims=[64,128,256,512]` |
| Loss not decreasing | Wrong import (still using Mamba1 VMamba) | Check import in train.py |

---

## References

- [Audit Checklist (quick summary)](./references/checklist.md)
- Official Mamba3: `mamba_ssm/modules/mamba3.py`
- DSA-Mamba3 model: `mamba_ssm/models/DSAmamba3.py`
- SISO Triton kernel: `mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py`
