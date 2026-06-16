
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import h5py

# It seems there is an error with the weatherbench2 library, so we will implement the metrics manually.
# from weatherbench2.metrics import rmse, acc

def latitude_weighted_rmse(pred, true, lat):
    weights = np.cos(np.deg2rad(lat))
    weights /= weights.mean()
    
    error = pred - true
    
    # Ensure weights are broadcastable to the error shape
    if error.ndim == 3: # (time, lat, lon)
        weights = weights.reshape(1, -1, 1)
    elif error.ndim == 2: # (lat, lon)
        weights = weights.reshape(-1, 1)

    return np.sqrt(np.mean((error)**2 * weights))

def latitude_weighted_acc(pred, true, lat):
    weights = np.cos(np.deg2rad(lat))
    weights /= weights.mean()

    # Ensure weights are broadcastable
    if pred.ndim == 3:
        weights = weights.reshape(1, -1, 1)
    elif pred.ndim == 2:
        weights = weights.reshape(-1, 1)

    pred_mean = np.mean(pred * weights)
    true_mean = np.mean(true * weights)
    
    pred_anomaly = pred - pred_mean
    true_anomaly = true - true_mean
    
    covariance = np.sum(pred_anomaly * true_anomaly * weights)
    pred_variance = np.sum(pred_anomaly**2 * weights)
    true_variance = np.sum(true_anomaly**2 * weights)
    
    return covariance / (np.sqrt(pred_variance) * np.sqrt(true_variance))


def calculate_metrics():
    """
    Calculates RMSE and ACC for all model predictions.
    """
    prediction_files = glob.glob('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/results/*_predictions.zarr')
    
    # Load ground truth data
    gt_path = '/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/data/out_of_sample/2018.h5'
    with h5py.File(gt_path, 'r') as f:
        gt_data = f['fields'][1:29, :20, :720, :] # 28 steps for 7 days, 20 channels, 720 lat

    variables = ['z500', 't850', 't2m', 'u10', 'v10']
    var_indices = {'z500': 14, 't850': 5, 't2m': 2, 'u10': 0, 'v10': 1}

    results = []

    for pred_file in prediction_files:
        model_name = os.path.basename(pred_file).replace('_predictions.zarr', '')
        pred_ds = xr.open_zarr(pred_file)

        latency_avg = pred_ds.attrs.get('latency_avg_ms', 'N/A')
        latency_std = pred_ds.attrs.get('latency_std_ms', 'N/A')

        for var in variables:
            pred_var = pred_ds[var].values
            true_var = gt_data[:, var_indices[var], :, :]
            
            lat = pred_ds['lat'].values

            for lead_time_idx, lead_time_hours in enumerate(pred_ds['time'].values):
                lead_time_days = (lead_time_hours / 24) + 0.25

                pred_step = pred_var[lead_time_idx]
                true_step = true_var[lead_time_idx]

                rmse_val = latitude_weighted_rmse(pred_step, true_step, lat)
                acc_val = latitude_weighted_acc(pred_step, true_step, lat)

                results.append({
                    'model': model_name,
                    'variable': var,
                    'lead_time_days': lead_time_days,
                    'rmse': rmse_val,
                    'acc': acc_val,
                    'latency_avg_ms': latency_avg,
                    'latency_std_ms': latency_std
                })

    df = pd.DataFrame(results)
    df.to_csv('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/results/evaluation_metrics.csv', index=False)
    print("Saved metrics to /home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/results/evaluation_metrics.csv")

if __name__ == '__main__':
    calculate_metrics()
