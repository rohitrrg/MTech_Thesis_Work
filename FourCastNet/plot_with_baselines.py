import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import gcsfs
from tqdm import tqdm

# Allow importing the local WeatherBench2 package.
LOCAL_WB2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weatherbench2"))
if LOCAL_WB2 not in sys.path:
    sys.path.insert(0, LOCAL_WB2)

from weatherbench2 import metrics


VARIABLES = ["z500", "t850", "t2m", "u10", "v10"]
WB2_VARS = {
    "z500": ("geopotential", 500),
    "t850": ("temperature", 850),
    "t2m": ("2m_temperature", None),
    "u10": ("10m_u_component_of_wind", None),
    "v10": ("10m_v_component_of_wind", None),
}

ERA5_PATH = "gs://weatherbench2/datasets/era5/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
IFS_PATH = "gs://weatherbench2/datasets/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr"
CLIM_PATH = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "variable" in df.columns and "rmse" in df.columns:
        return df
    required = {"model_name", "lead_time_step"}.union(VARIABLES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df_long = df.melt(
        id_vars=["model_name", "lead_time_step"],
        value_vars=VARIABLES,
        var_name="variable",
        value_name="rmse",
    )
    return df_long


def _find_lead_dim(ds):
    for dim in ds.dims:
        if "prediction_timedelta" in dim:
            return dim
        if "lead" in dim or "step" in dim or "prediction" in dim:
            return dim
    return None


def _lead_value(lead_coord, hours):
    if np.issubdtype(lead_coord.dtype, np.timedelta64):
        return np.timedelta64(hours, "h")
    return hours


def _align_forecast_time(forecast, lead_value, lead_dim):
    if "init_time" in forecast.coords:
        if np.issubdtype(forecast[lead_dim].dtype, np.timedelta64):
            valid_time = forecast["init_time"] + lead_value
        else:
            valid_time = forecast["init_time"] + np.timedelta64(int(lead_value), "h")
        forecast = forecast.assign_coords(time=valid_time).swap_dims({"init_time": "time"})
        forecast = forecast.drop_vars("init_time")
    elif "time" in forecast.coords:
        if np.issubdtype(forecast[lead_dim].dtype, np.timedelta64):
            valid_time = forecast["time"] + lead_value
        else:
            valid_time = forecast["time"] + np.timedelta64(int(lead_value), "h")
        forecast = forecast.assign_coords(time=valid_time)
    return forecast


def _select_variables(ds):
    out = {}
    for out_name, (src_name, level) in WB2_VARS.items():
        if src_name not in ds:
            raise KeyError(f"{src_name} not found in dataset")
        var = ds[src_name]
        if level is not None:
            if "level" not in var.dims:
                raise KeyError(f"{src_name} missing level dimension")
            var = var.sel(level=level)
        out[out_name] = var
    return xr.Dataset(out)


def _climatology_for_times(clim, times):
    times = pd.to_datetime(times.values)
    dayofyear = xr.DataArray(times.dayofyear, dims=["time"], coords={"time": times})
    hour = xr.DataArray(times.hour, dims=["time"], coords={"time": times})
    clim_sel = clim.sel(dayofyear=dayofyear, hour=hour)
    return clim_sel.assign_coords(time=times).swap_dims({"dayofyear": "time"})


def compute_baseline_rmse(era5, ifs, clim, lead_hours):
    rmse_metric = metrics.RMSESqrtBeforeTimeAvg()
    results = {var: {"ifs": [], "clim": []} for var in VARIABLES}

    lead_dim = _find_lead_dim(ifs)
    if lead_dim is None:
        raise ValueError("Could not find lead_time dimension in IFS dataset.")

    for hours in tqdm(lead_hours, desc="Baseline lead times"):
        lead_value = _lead_value(ifs[lead_dim], hours)
        ifs_step = ifs.sel({lead_dim: lead_value})
        ifs_step = _align_forecast_time(ifs_step, lead_value, lead_dim)

        truth = era5.sel(time=ifs_step.time)

        ifs_rmse = rmse_metric.compute_chunk(ifs_step, truth)
        for var in VARIABLES:
            results[var]["ifs"].append(float(ifs_rmse[var].values))

        clim_step = _climatology_for_times(clim, truth.time)
        clim_rmse = rmse_metric.compute_chunk(clim_step, truth)
        for var in VARIABLES:
            results[var]["clim"].append(float(clim_rmse[var].values))

    return results


def plot_results(df_long, baselines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for var in tqdm(VARIABLES, desc="Plotting variables"):
        plt.figure(figsize=(12, 8))
        df_var = df_long[df_long["variable"] == var].copy()
        df_var["lead_time_days"] = df_var["lead_time_step"] * 6.0 / 24.0

        for model_name, group in df_var.groupby("model_name"):
            plt.plot(
                group["lead_time_days"],
                group["rmse"],
                label=model_name,
                linewidth=1.2,
            )

        plt.plot(
            df_var["lead_time_days"].unique(),
            baselines[var]["ifs"],
            label="ECMWF IFS (Operational)",
            color="black",
            linewidth=3.0,
        )
        plt.plot(
            df_var["lead_time_days"].unique(),
            baselines[var]["clim"],
            label="Climatology",
            color="black",
            linestyle="--",
            linewidth=3.0,
        )

        plt.title(f"Latitude-Weighted RMSE for {var.upper()}")
        plt.xlabel("Lead Time (Days)")
        plt.ylabel("Latitude-Weighted RMSE")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(ncol=2, fontsize=8, frameon=False)

        out_path = os.path.join(output_dir, f"{var}_rmse_baselines.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")


def _parse_models_arg(models_arg):
    if not models_arg:
        return None
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def main():
    parser = argparse.ArgumentParser(description="Plot FourCastNet RMSE with WB2 baselines")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing results_*.csv files",
    )
    parser.add_argument("--output", default="plots_baselines")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model_name values to include",
    )
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.results_dir, "results_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No results_*.csv files found in {args.results_dir}")

    frames = [load_csv(path) for path in csv_paths]
    df_long = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df_long)} rows from {len(csv_paths)} CSV files")
    selected_models = _parse_models_arg(args.models)
    if selected_models:
        df_long = df_long[df_long["model_name"].isin(selected_models)]
        if df_long.empty:
            raise ValueError("No rows match the provided --models filter.")

    gcs = gcsfs.GCSFileSystem(token="anon")
    def _open_zarr(path):
        mapper = gcs.get_mapper(path)
        try:
            return xr.open_zarr(mapper, consolidated=True)
        except KeyError:
            return xr.open_zarr(mapper, consolidated=False)

    era5 = _select_variables(_open_zarr(ERA5_PATH))
    ifs = _select_variables(_open_zarr(IFS_PATH))
    clim = _select_variables(_open_zarr(CLIM_PATH))

    era5 = era5.sel(time=slice("2018-01-01", "2018-12-31T18:00"))
    if "init_time" in ifs.coords:
        ifs = ifs.sel(init_time=slice("2018-01-01", "2018-12-31T18:00"))
    else:
        ifs = ifs.sel(time=slice("2018-01-01", "2018-12-31T18:00"))
    # Climatology uses dayofyear/hour coordinates

    lead_hours = [int(step) * 6 for step in sorted(df_long["lead_time_step"].unique())]
    print("Computing baseline RMSE for IFS and climatology...")
    baselines = compute_baseline_rmse(era5, ifs, clim, lead_hours)
    print("Baseline RMSE computation complete")

    plot_results(df_long, baselines, args.output)


if __name__ == "__main__":
    main()
