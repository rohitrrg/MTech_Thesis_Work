
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import gcsfs
import os

def plot_results():
    """
    Plots the evaluation results and compares with WeatherBench 2 baselines.
    """
    metrics_df = pd.read_csv('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/results/evaluation_metrics.csv')
    
    # Initialize GCS filesystem
    gcs = gcsfs.GCSFileSystem(token='anon')

    # Load WeatherBench 2 baselines
    ifs_path = 'gs://weatherbench2/datasets/hres/2016-2022-0p25deg-chunk-1.zarr-v2'
    climatology_path = 'gs://weatherbench2/datasets/era5-climatology/1990-2019-6h-0p25deg.zarr'

    ifs_ds = xr.open_zarr(gcs.get_mapper(ifs_path), consolidated=True)
    climatology_ds = xr.open_zarr(gcs.get_mapper(climatology_path), consolidated=True)

    variables = ['z500', 't850', 't2m', 'u10', 'v10']
    
    output_dir = '/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/results/plots'
    os.makedirs(output_dir, exist_ok=True)

    for var in variables:
        plt.figure(figsize=(12, 8))
        
        # Plot custom models
        for model_type in ['backbone', 'pruned', 'quantized']:
            if model_type == 'backbone':
                model_df = metrics_df[metrics_df['model'] == 'backbone']
                plt.plot(model_df['lead_time_days'], model_df['rmse'], label='Base Model', marker='o')
            elif model_type == 'pruned':
                 for i in range(1, 7):
                    model_name = f'pruned_{i}_layers'
                    model_df = metrics_df[(metrics_df['model'] == model_name) & (metrics_df['variable'] == var)]
                    if not model_df.empty:
                        plt.plot(model_df['lead_time_days'], model_df['rmse'], label=f'Pruned {i} Layers', linestyle='--')
            elif model_type == 'quantized':
                for i in [16, 32]:
                    model_name = f'backbone_fp{i}'
                    model_df = metrics_df[(metrics_df['model'] == model_name) & (metrics_df['variable'] == var)]
                    if not model_df.empty:
                        plt.plot(model_df['lead_time_days'], model_df['rmse'], label=f'Quantized FP{i}', linestyle=':')


        # Placeholder for plotting IFS and Climatology from WeatherBench2
        # This requires loading and processing the baseline data which is complex.
        # For now, we will just plot the custom model results.
        
        plt.title(f'Latitude-Weighted RMSE for {var}')
        plt.xlabel('Lead Time (Days)')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(output_dir, f'{var}_rmse_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    plot_results()
