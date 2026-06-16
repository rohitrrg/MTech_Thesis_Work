import os
import sys
import time
import numpy as np
import argparse
import h5py
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import glob
import shutil
from tqdm import tqdm

# Add FourCastNet parent directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from networks.afnonet import AFNONet
from networks.afnonet1 import AFNONet as AFNONetQuant
from utils.YParams import YParams

def get_model_list():
    """
    Returns a list of all model checkpoint paths.
    """
    base_model = '/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/weights/backbone.ckpt'
    # Using the fine-tuned pruned models as requested
    pruned_models = sorted(glob.glob('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/weights/fine-tuned models/*.ckpt'))
    quantized_models = sorted(glob.glob('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/weights/quantized models/*.ckpt'))
    
    # Sorting pruned models by the number of layers pruned
    def _pruned_layers_from_name(path):
        name = os.path.basename(path)
        parts = name.split('_')
        if len(parts) > 2 and parts[0] == 'pruned' and parts[1].isdigit():
            return int(parts[1])
        return 0

    pruned_models = sorted(pruned_models, key=_pruned_layers_from_name)

    model_list = [base_model]  + pruned_models + quantized_models
    return model_list

def get_model_depth(model_name):
    """
    Returns the depth of the model based on its name.
    Correctly parses filenames like 'pruned_1_layers_epoch_6'.
    """
    if 'pruned' in model_name:
        try:
            # Matches pruned_<n>_layers_*, pruned_<n>_fp16, pruned_<n>_layers_epoch_*
            parts = model_name.split('_')
            pruned_idx = parts.index('pruned') if 'pruned' in parts else 0
            num_token = parts[pruned_idx + 1]
            num_pruned_layers = int(num_token)
            return 12 - num_pruned_layers
        except (IndexError, ValueError):
            # Fallback for unexpected filenames
            return 12
    return 12 # Default for backbone and non-pruned quantized models

def load_model(model_path, params, device):
    """
    Loads a model checkpoint.
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    is_quantized = 'fp16' in model_name
    
    depth = get_model_depth(model_name)
    params.depth = depth

    if is_quantized:
        model = AFNONetQuant(params).to(device)
    else:
        model = AFNONet(params).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint)

    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        name = key[7:] if key.startswith('module.') else key
        if name != 'ged':
            new_state_dict[name] = val

    model.load_state_dict(new_state_dict, strict=False)

    if is_quantized:
        model = model.half()
        for param in model.parameters():
            param.data = param.data.half()
        for buf in model.buffers():
            buf.data = buf.data.half()
    
    model.eval()
    return model

def measure_latency(model, device, repetitions=100, warm_up=20):
    """Measures average inference time using CUDA Events."""
    model_dtype = next(model.parameters()).dtype
    dummy_input = torch.randn(1, 20, 720, 1440, device=device, dtype=model_dtype)

    # Warm-up phase
    for _ in range(warm_up):
        _ = model(dummy_input)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    
    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize() # Wait for GPU to finish
            timings.append(starter.elapsed_time(ender))
            
    return np.mean(timings), np.std(timings)

class ParamsObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_inference():
    """
    Runs autoregressive inference for all models.
    """
    # Load base parameters from YAML
    base_params_yaml = YParams(os.path.abspath('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/config/AFNO.yaml'), 'afno_backbone')
    
    # Convert to a simple object, like in evaluate.py
    base_params = {
        'patch_size': 8, #base_params_yaml.patch_size,
        'embed_dim': 768, #base_params_yaml.embed_dim,
        'num_blocks': base_params_yaml.num_blocks,
        'img_size': (720, 1440),
        'in_channels': 20,
        'out_channels': 20,
    }
    params = ParamsObject(**base_params)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = '/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/data/out_of_sample/2018.h5'
    with h5py.File(data_path, 'r') as f:
        # Slice to 20 channels to match model's expected input
        data = f['fields'][:, :20, :, :]
    
    # Define variables and their indices in the data
    variables = ['z500', 't850', 't2m', 'u10', 'v10']
    # These indices are now relative to the 20-channel data
    var_indices = {'z500': 14, 't850': 5, 't2m': 2, 'u10': 0, 'v10': 1}
    
    # Load normalization stats and slice to 20 channels
    global_means = np.load('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/data/stats/global_means.npy')[:, :20, :, :]
    global_stds = np.load('/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/data/stats/global_stds.npy')[:, :20, :, :]

    model_list = get_model_list()
    rollout_steps = 28
    stride = 2

    for model_path in tqdm(model_list, desc="Processing models"):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        print(f"Running inference for model: {model_name}")

        model = load_model(model_path, params, device)

        avg_lat, std_lat = measure_latency(model, device)
        print(f"  Avg Latency: {avg_lat:.2f} ms (±{std_lat:.2f})")

        output_path = f"/home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/results/{model_name}_predictions.zarr"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        lead_times = np.arange(0, rollout_steps * 6, 6)
        init_indices = range(0, data.shape[0] - rollout_steps, stride)

        for idx_i, start_idx in enumerate(tqdm(init_indices, desc=f"Initial conditions {model_name}", leave=False)):
            predictions = []

            # Initial condition for this rollout
            initial_condition = torch.from_numpy(data[start_idx:start_idx+1, :, :720, :]).float().to(device)
            initial_condition = (initial_condition - torch.from_numpy(global_means[:, :, :720, :]).float().to(device)) / torch.from_numpy(global_stds[:, :, :720, :]).float().to(device)

            current_state = initial_condition
            if next(model.parameters()).dtype == torch.float16:
                current_state = current_state.half()

            with torch.no_grad():
                for _ in range(rollout_steps):
                    output = model(current_state)

                    unnormalized_output = output.float() * torch.from_numpy(global_stds[:, :, :720, :]).float().to(device) + torch.from_numpy(global_means[:, :, :720, :]).float().to(device)

                    predictions.append(unnormalized_output.cpu().numpy())
                    current_state = output

            predictions = np.concatenate(predictions, axis=0)

            ds = xr.Dataset(
                {
                    var: (
                        ('init_time', 'lead_time', 'lat', 'lon'),
                        predictions[:, var_indices[var], :, :][None, ...]
                    )
                    for var in variables
                },
                coords={
                    'init_time': [start_idx],
                    'lead_time': lead_times,
                    'lat': np.linspace(90, -90, 720),
                    'lon': np.linspace(0, 360, 1440, endpoint=False)
                }
            )

            ds.attrs['latency_avg_ms'] = avg_lat
            ds.attrs['latency_std_ms'] = std_lat

            if idx_i == 0:
                ds.to_zarr(output_path, mode='w')
            else:
                ds.to_zarr(output_path, mode='a', append_dim='init_time')

        print(f"Saved predictions to {output_path}")

if __name__ == '__main__':
    run_inference()
