import os
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

data_path = "FourCastNet/data/out_of_sample/2018.h5"
stats_root = "FourCastNet/data/stats"

device = "cuda" if torch.cuda.is_available() else "cpu"
time_means = np.load(os.path.join(stats_root, "time_means.npy"))[:1, :20, :720, :]
clim_t = torch.from_numpy(time_means).float().to(device)

try:
    lat_1d = np.load(os.path.join(stats_root, "latitude.npy"))
except FileNotFoundError:
    lat_1d = np.linspace(90, -90, 721)
lat_1d = lat_1d[:720]
weights = torch.cos(torch.deg2rad(torch.from_numpy(lat_1d).float()))
weights = (weights / weights.mean()).view(1, 720, 1).to(device)

variables = ["z500", "t850", "t2m", "u10", "v10"]
var_indices = {"z500": 14, "t850": 5, "t2m": 2, "u10": 0, "v10": 1}
leads = [1, 4, 12] # 6h, 24h, 72h
stride = 6
batch_size = 4

results = []

for baseline in ["persistence", "climatology"]:
    metrics = {lead: {var: {"rmse_sum":0.0, "acc_sum":0.0, "count":0} for var in variables} for lead in leads}
    
    with h5py.File(data_path, "r") as f:
        fields = f["fields"]
        n_samples = fields.shape[0]
        init_indices = list(range(0, n_samples - max(leads), stride))
        
        for batch_start in tqdm(range(0, len(init_indices), batch_size), desc=f"Eval {baseline}"):
            batch_indices = init_indices[batch_start:batch_start + batch_size]
            
            # For persistence, we need the init state
            if baseline == "persistence":
                inits = []
                for idx in batch_indices:
                    inits.append(fields[idx:idx+1, :20, :720, :])
                inits = np.concatenate(inits, axis=0)
                pred = torch.from_numpy(inits).float().to(device)
            else:
                # For climatology, the prediction is exactly the climatology for all samples
                pred = clim_t.expand(len(batch_indices), -1, -1, -1)
            
            for lead in leads:
                targets = []
                for idx in batch_indices:
                    targets.append(fields[idx+lead:idx+lead+1, :20, :720, :])
                targets = np.concatenate(targets, axis=0)
                target = torch.from_numpy(targets).float().to(device)
                
                for var in variables:
                    c = var_indices[var]
                    p = pred[:, c]
                    t = target[:, c]
                    clim = clim_t[0, c]
                    
                    diff = p - t
                    rmse_step = torch.sqrt(torch.mean((diff ** 2) * weights, dim=(1,2)))
                    
                    p_anom = p - clim
                    t_anom = t - clim
                    p_anom_mean = torch.mean(p_anom * weights, dim=(1,2), keepdim=True)
                    t_anom_mean = torch.mean(t_anom * weights, dim=(1,2), keepdim=True)
                    
                    p_var = p_anom - p_anom_mean
                    t_var = t_anom - t_anom_mean
                    
                    cov = torch.sum(p_var * t_var * weights, dim=(1,2))
                    p_std = torch.sqrt(torch.sum((p_var ** 2) * weights, dim=(1,2)))
                    t_std = torch.sqrt(torch.sum((t_var ** 2) * weights, dim=(1,2)))
                    
                    acc_step = cov / (p_std * t_std)
                    
                    metrics[lead][var]["rmse_sum"] += torch.sum(rmse_step).item()
                    
                    # For climatology, ACC is mathematically undefined or 0
                    if baseline == "climatology":
                        acc_step = torch.zeros_like(acc_step)
                        # Climatology anomaly is 0 exactly, so set denominator to 1 to avoid inf if it happens
                        
                    # Handle NaNs from zero standard deviation properly:
                    acc_step = torch.nan_to_num(acc_step, nan=0.0)
                        
                    metrics[lead][var]["acc_sum"] += torch.sum(acc_step).item()
                    metrics[lead][var]["count"] += len(batch_indices)
                    
    for lead in leads:
        for var in variables:
            avg_rmse = metrics[lead][var]["rmse_sum"] / metrics[lead][var]["count"]
            avg_acc = metrics[lead][var]["acc_sum"] / metrics[lead][var]["count"]
            results.append({
                "model_name": baseline,
                "lead_time_step": lead,
                "variable": var,
                "rmse": avg_rmse,
                "acc": avg_acc
            })

df = pd.DataFrame(results)
df.to_csv("FourCastNet/results/baseline_2018.csv", index=False)
print("Saved results/baseline_2018.csv")
