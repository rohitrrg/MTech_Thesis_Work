import os
import h5py
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weatherbench2")))
from weatherbench2 import metrics

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
H5_FILE_PATH = "FourCastNet/data/out_of_sample/2018.h5"
OUTPUT_CSV = "FourCastNet/results/generate_baselines_2018.csv"

# The 4 variables you are evaluating
VARIABLES = ["z500", "t850", "t2m", "u10"]

# WARNING: Update these integers to match the exact channel indices in your .h5 file!
CHANNEL_MAP = {
    'u10': 0,
    'v10': 1,    # Included just in case, though not in your 4-var list
    't2m': 2,
    'z500': 14, 
    't850': 5  
}

# 28 steps at 6-hour intervals (1 to 28)
LEAD_STEPS = list(range(1, 29))
LEAD_HOURS = [step * 6 for step in LEAD_STEPS]

CLIM_PATH = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"


# ==========================================
# 2. HDF5 LAZY WRAPPER
# ==========================================
def load_h5_as_xarray(h5_path):
    print(f"Wrapping {h5_path} into lazy xarray dataset...")
    f = h5py.File(h5_path, 'r')
    
    # Check your actual dataset key name. Usually 'fields' in FourCastNet.
    dataset_key = 'fields' if 'fields' in f else list(f.keys())[0]
    data = f[dataset_key] 
    
    # Generate exact 2018 coordinates (1460 steps)
    times = pd.date_range(start='2018-01-01 00:00:00', end='2018-12-31 18:00:00', freq='6h')
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)
    
    ds_dict = {}
    for var_name in VARIABLES:
        channel_idx = CHANNEL_MAP[var_name]
        # Use Dask to load strictly in chunks so RAM stays low
        lazy_array = da.from_array(data[:, channel_idx, :, :], chunks=(100, 721, 1440))
        
        ds_dict[var_name] = xr.DataArray(
            data=lazy_array,
            dims=['time', 'lat', 'lon'],
            coords={'time': times, 'lat': lats, 'lon': lons}
        )
        
    return xr.Dataset(ds_dict)


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    # 1. Load local 2018 ground truth
    era5_2018 = load_h5_as_xarray(H5_FILE_PATH)
    
    # 2. Load WeatherBench Climatology from Cloud
    print("Connecting to WeatherBench 2 GCS bucket...")
    fs = gcsfs.GCSFileSystem(anon=True, skip_instance_cache=True)
    try:
        clim_store = fs.get_mapper(CLIM_PATH)
        clim_dataset = xr.open_zarr(clim_store, consolidated=True)
    except Exception as e:
        print(f"Consolidated zarr failed ({e}), trying non-consolidated...")
        try:
            clim_dataset = xr.open_zarr(clim_store, consolidated=False)
        except Exception as e2:
            print(f"ERROR: Could not load climatology: {e2}")
            raise

    WB2_VARS = {
        "z500": ("geopotential", 500),
        "t850": ("temperature", 850),
        "t2m": ("2m_temperature", None),
        "u10": ("10m_u_component_of_wind", None),
        "v10": ("10m_v_component_of_wind", None),
    }

    clim_vars = {}
    for out_name, (src_name, level) in WB2_VARS.items():
        if out_name not in VARIABLES: continue
        var = clim_dataset[src_name]
        if level is not None:
            var = var.sel(level=level)
        clim_vars[out_name] = var
    clim_dataset = xr.Dataset(clim_vars)
    
    # 3. Initialize Metric
    rmse_metric = metrics.RMSE()
    
    # Data storage for final CSV
    results_list = []

    # ==========================================
    # A. CALCULATE PERSISTENCE & CLIMATOLOGY
    # ==========================================
    for hours in LEAD_HOURS:
        print(f"--- Processing Lead Time: {hours}h (Step {int(hours/6)}) ---")
        delta = np.timedelta64(hours, 'h')
        
        # Determine valid times for this specific step
        start_time = era5_2018.time.values[0] + delta
        valid_times = era5_2018.time.sel(time=slice(start_time, None))
        
        # --- Persistence ---
        init_times = valid_times - delta
        persistence_forecast = era5_2018.sel(time=init_times)
        persistence_forecast['time'] = valid_times # Align coordinates
        
        truth = era5_2018.sel(time=valid_times)
        
        print("  Computing Persistence...")
        pers_rmse = rmse_metric.compute_chunk(persistence_forecast, truth).compute()
        
        # --- Climatology ---
        print("  Computing Climatology...")
        # Match time via dayofyear and hour matching WeatherBench2 climatology specs
        target_times = pd.to_datetime(valid_times.values)
        
        # Some climatology stores use time, some use dayofyear/hour
        if "dayofyear" in clim_dataset.coords:
            dayofyear = xr.DataArray(target_times.dayofyear, dims=["time"], coords={"time": valid_times})
            hour = xr.DataArray(target_times.hour, dims=["time"], coords={"time": valid_times})
            clim_sel = clim_dataset.sel(dayofyear=dayofyear, hour=hour)
            if "dayofyear" in clim_sel.dims:
                clim_step = clim_sel.swap_dims({"dayofyear": "time"})
            else:
                clim_step = clim_sel
        else:
            clim_time = pd.to_datetime(clim_dataset.time.values)
            clim_time_2018 = pd.to_datetime([f"2018-{t.strftime('%m-%d %H:%M:%S')}" for t in clim_time])
            clim_aligned = clim_dataset.assign_coords(time=clim_time_2018)
            clim_step = clim_aligned.sel(time=valid_times, method="nearest")
        
        clim_rmse = rmse_metric.compute_chunk(clim_step, truth).compute()
        
        # --- Store Results ---
        step_number = int(hours / 6)
        
        # Append Persistence row
        row_pers = {'model_name': 'Persistence', 'lead_time_step': step_number}
        for var in VARIABLES:
            row_pers[var] = float(pers_rmse[var].values)
        results_list.append(row_pers)
        
        # Append Climatology row
        row_clim = {'model_name': 'Climatology', 'lead_time_step': step_number}
        for var in VARIABLES:
            row_clim[var] = float(clim_rmse[var].values)
        results_list.append(row_clim)

    # ==========================================
    # B. SAVE TO CSV
    # ==========================================
    print(f"\nFormatting and saving results to {OUTPUT_CSV}...")
    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Execution complete. Baseline generation successful.")

if __name__ == "__main__":
    main()