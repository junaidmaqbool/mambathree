# DSA-Mamba3 Audit — Quick Checklist

## 1. Imports
- [ ] `from mamba_ssm.modules.mamba3 import Mamba3` ✓
- [ ] No stale import of VMamba SS2D or mamba_simple.Mamba

## 2. SISO / MIMO Mode
- [ ] `is_mimo=False` in ALL Mamba3 instantiations (required for Kaggle T4)
- [ ] No TileLang import required when `is_mimo=False`

## 3. Mamba3 Constructor Args (per stage)
- [ ] `d_model = dims[i]` (correct channel width)
- [ ] `(expand * dims[i]) % headdim == 0` for every stage
- [ ] `rope_fraction in {0.5, 1.0}`
- [ ] `d_state >= 4` (ensures `num_rope_angles > 0`)
- [ ] `ngroups = 1` (standard SISO)
- [ ] `chunk_size = 64` (optimal for SISO Triton kernel)

## 4. Tensor Shape Contract
- [ ] PatchEmbed input: `(B, 3, H, W)` → output: `(B, H/P, W/P, dims[0])`
- [ ] Mamba3 input: `(B, seqlen, d_model)` — flatten spatial before calling
- [ ] Mamba3 output: same shape → reshape back to `(B, H', W', C)`
- [ ] PatchMerging halves H and W, doubles C
- [ ] Final spatial: `(B, H/(P*2^(N-1)), W/(P*2^(N-1)), dims[-1])`  
  Default: `(B, 7, 7, 768)` for 224×224 input, patch_size=4, 4 stages

## 5. Classification Head
- [ ] `num_classes=2` for binary anemia (0=non-anemic, 1=anemic)
- [ ] `head = nn.Linear(dims[-1], num_classes)` ← no bias issue
- [ ] Loss: `nn.CrossEntropyLoss()` with long-int labels

## 6. Regression Head (Hb estimation)
- [ ] `num_classes=1`
- [ ] Loss: `nn.MSELoss()` or `nn.HuberLoss()`

## 7. no_weight_decay
- [ ] Returns `{"dt_bias", "D", "B_bias", "C_bias"}`
- [ ] Optimizer wrapper filters by name suffix

## 8. train.py import
- [ ] `from mamba_ssm.models.DSAmamba3 import VSSM as dsamamba`

## 9. Known Limitation
- [ ] `layer_idx` resets per stage (harmless for classification; note if adding inference cache)
