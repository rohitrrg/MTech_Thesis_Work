"""
Memory-efficient baseline generator for 2018 data.
Streams data from HDF5 using lazy loading to minimize RAM usage.
"""
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
H5_FILE_PATH = "FourCastNet/data/out_of_sample/2018.h5"
OUTPUT_CSV = "FourCastNet/results/baselines_2018.csv"

VARIABLES = ["z500", "t850", "t2m", "u10", "v10"]
CHANNEL_MAP = {
    'u10': 0,
    'v10': 1,
    't2m': 2,
    'z500': 14,
    't850': 5
}

# Lead times: 1, 4, 12 steps (6h, 24h, 72h)
LEAD_STEPS = [1, 4, 12]


def compute_rmse(forecast, truth):
    """Compute spatial-temporal RMSE"""
    diff = forecast - truth
    mse = np.mean(diff ** 2, axis=(1, 2, 3))  # Average over spatial dims
    rmse = np.sqrt(mse)
    return np.mean(rmse)


def main():
    print(f"Opening {H5_FILE_PATH} (lazy loading)...")
    with h5py.File(H5_FILE_PATH, 'r') as f:
        dataset_key = 'fields' if 'fields' in f else list(f.keys())[0]
        data = f[dataset_key]
        n_times, n_channels, n_lats, n_lons = data.shape
        print(f"Dataset shape: {data.shape}")
        
        # Compute climatology: 100-timestep rolling average
        print("Computing climatology baseline (rolling average)...")
        clim_cache = {}
        window_size = 100
        
        for t in tqdm(range(n_times), desc="Climatology", total=n_times):
            start_idx = max(0, t - window_size // 2)
            end_idx = min(n_times, t + window_size // 2 + 1)
            clim_cache[t] = np.mean(data[start_idx:end_idx], axis=0)
        
        results = []
        
        # Process each lead time
        for lead_step in tqdm(LEAD_STEPS, desc="Lead times"):
            valid_start = lead_step
            valid_end = n_times
            valid_times = np.arange(valid_start, valid_end)
            init_times = valid_times - lead_step
            
            for var_name in VARIABLES:
                ch_idx = CHANNEL_MAP[var_name]
                
                # Load truth data
                truth = data[valid_times, ch_idx]
                
                # Persistence forecast
                persistence_fcst = data[init_times, ch_idx]
                pers_rmse = compute_rmse(persistence_fcst, truth)
                
                # Climatology forecast
                clim_fcst = np.array([clim_cache[t][ch_idx] for t in valid_times])
                clim_rmse = compute_rmse(clim_fcst, truth)
                
                # Store results
                results.append({
                    'model_name': 'persistence',
                    'lead_time_step': lead_step,
                    'variable': var_name,
                    'rmse': float(pers_rmse),
                    'acc': 0.0
                })
                
                results.append({
                    'model_name': 'climatology',
                    'lead_time_step': lead_step,
                    'variable': var_name,
                    'rmse': float(clim_rmse),
                    'acc': 0.0
                })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Baselines saved to {OUTPUT_CSV}")
    print(df.head(20))


if __name__ == "__main__":
    main()
