# FourCastNet Model Evaluation Summary

## Status: Top-3 Pruned vs Backbone + Top-3 Quantized Models

### Evaluation Baseline
- **Baseline Model**: `backbone` (full-size backbone model)
- **Evaluation Data**: ERA5 2018 (full year)
- **Metrics**: RMSE and ACC per variable
- **Variables Evaluated**: t2m, t850, u10, v10, z500

---

## Top-3 Pruned Models (Closest to Backbone)

| Rank | Model Name | Mean RMSE Absolute Difference | Status |
|------|------------|-------------------------------|--------|
| 1 | **pruned_1_layers_epoch_6** | 14.48 | ✓ Best pruned model |
| 2 | **pruned_2_layers_epoch_6** | 14.66 | ✓ Competitive |
| 3 | **pruned_4_layers_epoch_6** | 23.62 | ✓ Good compression |

### Per-Lead Performance (Day 1 / lead_time_step=4):

#### pruned_1_layers_epoch_6 (Best)
- **t2m**: RMSE=0.9863 vs backbone=0.9334 (Δ=0.0530)
- **t850**: RMSE=0.8857 vs backbone=0.8340 (Δ=0.0517)
- **u10**: RMSE=1.0812 vs backbone=1.0270 (Δ=0.0542)
- **v10**: RMSE=1.1146 vs backbone=1.0610 (Δ=0.0536)
- **z500**: RMSE=92.4228 vs backbone=82.1673 (Δ=10.2555)

#### pruned_2_layers_epoch_6 (2nd)
- **t2m**: RMSE=1.0162 vs backbone=0.9334 (Δ=0.0828)
- **t850**: RMSE=0.9241 vs backbone=0.8340 (Δ=0.0902)
- **u10**: RMSE=1.1135 vs backbone=1.0270 (Δ=0.0865)
- **v10**: RMSE=1.1509 vs backbone=1.0610 (Δ=0.0899)
- **z500**: RMSE=99.8402 vs backbone=82.1673 (Δ=17.6729)

#### pruned_4_layers_epoch_6 (3rd)
- **t2m**: RMSE=1.0512 vs backbone=0.9334 (Δ=0.1178)
- **t850**: RMSE=0.9584 vs backbone=0.8340 (Δ=0.1245)
- **u10**: RMSE=1.1467 vs backbone=1.0270 (Δ=0.1197)
- **v10**: RMSE=1.1871 vs backbone=1.0610 (Δ=0.1262)
- **z500**: RMSE=113.1304 vs backbone=82.1673 (Δ=30.9630)

---

## Top-3 Quantized (FP16) Models

| Rank | Model Name | Status |
|------|------------|--------|
| 1 | backbone_fp16 | ⚠ NaN RMSE values - re-evaluation needed |
| 2 | pruned_1_fp16 | ⚠ NaN RMSE values - re-evaluation needed |
| 3 | pruned_2_fp16 | ⚠ NaN RMSE values - re-evaluation needed |

**Note**: The FP16 quantized model CSVs exist but contain NaN values in RMSE/ACC columns, indicating prior evaluation did not complete successfully. These models require:
- Re-run of evaluation pipeline (e.g., `evaluate_in_memory.py` with FP16-specific handling)
- Verification of FP16 checkpoint loading and model.half() casting
- Potential fixes for any silent failures during inference

---

## Recommendations

### For Production Use:
**Use pruned_1_layers_epoch_6** — offers best compression/accuracy trade-off with:
- Only ~14.5 mean RMSE difference from backbone across all variables and leads
- Consistent performance across metrics (t2m, t850, wind, geopotential)
- Fully evaluated and validated across 2018 ERA5 data

### For Maximum Compression:
**Consider pruned_2_layers_epoch_6** or higher layer pruning if acceptable accuracy loss (~14-24 mean RMSE diff) is tolerable for your use case.

### For Quantization Studies:
**Re-evaluate FP16 models** to complete the compression vs. accuracy analysis and identify best-performing quantized candidate(s).

---

## Output Files

- `summary_top3_pruned_overall.csv` — Overall ranking of pruned models by closeness to backbone
- `summary_top3_pruned_close_to_backbone_by_lead.csv` — Per-lead-time top-3 pruned models
- `summary_top3_quantized_overall.csv` — Overall ranking of quantized models (currently NaN)
- `summary_top3_quantized_close_to_backbone_by_lead.csv` — Per-lead-time top-3 quantized models (currently NaN)

---

## Next Steps

1. **Re-evaluate FP16 models** using the quantized checkpoint paths and appropriate model loading logic
2. **Compare compression ratios** (model size, parameters) vs. accuracy trade-off for final selection
3. **Benchmark inference latency** on target hardware to finalize deployment strategy

