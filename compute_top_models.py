#!/usr/bin/env python
import os
import pandas as pd
import numpy as np

results_dir = "/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/results"
merged_path = os.path.join(results_dir, "results_all_models.csv")
merged = pd.read_csv(merged_path)

print("Found models:")
print(sorted(merged['model_name'].unique()))

# Identify backbone model name
names = merged['model_name'].astype(str).unique()
backbone_candidates = [n for n in names if 'backbone' in n.lower() and 'prun' not in n.lower() and 'quant' not in n.lower() and 'fp16' not in n.lower()]
if not backbone_candidates:
    backbone_candidates = [n for n in names if 'backbone' in n.lower()]

if not backbone_candidates:
    raise SystemExit('No backbone model name found in results_all_models.csv')

# choose the simplest backbone name
backbone_name = sorted(backbone_candidates, key=lambda s: (len(s), s))[0]
print('\nUsing backbone reference:', backbone_name)

# compute backbone RMSE per variable & lead_time_step
backbone_df = merged[merged['model_name'] == backbone_name]
if backbone_df.empty:
    # fallback: choose any model that contains backbone
    backbone_df = merged[merged['model_name'].str.contains('backbone', case=False)]
    backbone_name = backbone_df['model_name'].unique()[0]

backbone_ref = backbone_df.groupby(['variable', 'lead_time_step'], as_index=False)['rmse'].mean().rename(columns={'rmse':'backbone_rmse'})

# merge backbone ref into merged
merged2 = merged.merge(backbone_ref, on=['variable','lead_time_step'], how='left')
merged2['rmse_abs_diff'] = (merged2['rmse'] - merged2['backbone_rmse']).abs()

# classify models
merged2['is_pruned'] = merged2['model_name'].str.contains('prun', case=False)
merged2['is_quant'] = merged2['model_name'].str.contains('quant|fp16|int', case=False, regex=True)

# per-lead top-3 pruned
pruned = merged2[merged2['is_pruned']]
pruned_top3 = pruned.sort_values(['variable','lead_time_step','rmse_abs_diff'], ascending=[True,True,True]).groupby(['variable','lead_time_step']).head(3)
pruned_out = os.path.join(results_dir, 'summary_top3_pruned_close_to_backbone_by_lead.csv')
pruned_top3.to_csv(pruned_out, index=False)

# per-lead top-3 quantized
quant = merged2[merged2['is_quant']]
quant_top3 = quant.sort_values(['variable','lead_time_step','rmse_abs_diff'], ascending=[True,True,True]).groupby(['variable','lead_time_step']).head(3)
quant_out = os.path.join(results_dir, 'summary_top3_quantized_close_to_backbone_by_lead.csv')
quant_top3.to_csv(quant_out, index=False)

# overall (mean across leads and variables) closeness per model
overall = merged2.groupby(['model_name','is_pruned','is_quant'], as_index=False)['rmse_abs_diff'].mean().rename(columns={'rmse_abs_diff':'mean_rmse_abs_diff'})
pruned_overall = overall[overall['is_pruned']].sort_values('mean_rmse_abs_diff').head(3)
quant_overall = overall[overall['is_quant']].sort_values('mean_rmse_abs_diff').head(3)
pruned_overall.to_csv(os.path.join(results_dir,'summary_top3_pruned_overall.csv'), index=False)
quant_overall.to_csv(os.path.join(results_dir,'summary_top3_quantized_overall.csv'), index=False)

print('\nWrote files:')
print(' ', pruned_out)
print(' ', quant_out)
print(' ', os.path.join(results_dir,'summary_top3_pruned_overall.csv'))
print(' ', os.path.join(results_dir,'summary_top3_quantized_overall.csv'))

# Print overall top 3 summaries
print('\n\n=== OVERALL TOP 3 PRUNED (closest to backbone) ===')
print(pruned_overall.to_string())

print('\n\n=== OVERALL TOP 3 QUANTIZED (closest to backbone) ===')
print(quant_overall.to_string())

# Print sample for lead_time_step=4
sample_step = 4
print('\n\n=== Sample (lead_time_step=4) PRUNED top-3 closest to backbone (RMSE diff) ===')
for var in sorted(pruned_top3['variable'].unique()):
    rows = pruned_top3[(pruned_top3['variable']==var) & (pruned_top3['lead_time_step']==sample_step)]
    if rows.empty:
        continue
    print('\n', var)
    for _,r in rows.iterrows():
        print(f" {r['model_name']}: rmse={r['rmse']:.4f}, backbone_rmse={r['backbone_rmse']:.4f}, abs_diff={r['rmse_abs_diff']:.4f}")

print('\n\n=== Sample (lead_time_step=4) QUANTIZED top-3 closest to backbone (RMSE diff) ===')
for var in sorted(quant_top3['variable'].unique()):
    rows = quant_top3[(quant_top3['variable']==var) & (quant_top3['lead_time_step']==sample_step)]
    if rows.empty:
        continue
    print('\n', var)
    for _,r in rows.iterrows():
        print(f" {r['model_name']}: rmse={r['rmse']:.4f}, backbone_rmse={r['backbone_rmse']:.4f}, abs_diff={r['rmse_abs_diff']:.4f}")
