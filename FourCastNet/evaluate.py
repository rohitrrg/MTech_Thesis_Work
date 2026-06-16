import torch
import h5py
import numpy as np
import time
from collections import OrderedDict
from networks.afnonet import AFNONet
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels

def load_custom_model(model, checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def measure_latency(model, device, repetitions=100, warm_up=20):
    """Measures average inference time using CUDA Events."""
    dummy_input = torch.randn(1, 20, 720, 1440).to(device)
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

def run_full_year_benchmark(params_obj, weight_path, data_path, num_steps=1, stride=36):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFNONet(params_obj).to(device)
    model = load_custom_model(model, weight_path, device)

    # 1. Latency Measurement
    avg_lat, std_lat = measure_latency(model, device)

    # 2. Load Stats
    means = np.load("data/stats/global_means.npy")[0, :20].reshape(20, 1, 1)
    stds = np.load("data/stats/global_stds.npy")[0, :20].reshape(20, 1, 1)
    std_tensor = torch.as_tensor(stds).to(device).view(1, 20, 1, 1)
    
    # 3. Load Climatology (Time Means) for valid ACC
    time_means_raw = np.load("data/stats/time_means.npy")[0, :20].reshape(20, 721, 1440)
    time_means = time_means_raw[:, 0:720, :] # Slice to 720
    clim_std = torch.as_tensor((time_means - means) / stds).to(device).view(1, 20, 720, 1440).float()

    all_z500_rmse, all_t850_rmse, all_accs = [], [], []

    with h5py.File(data_path, 'r') as f:
        n_samples = f['fields'].shape[0]
        for start_idx in range(0, n_samples - num_steps, stride):
            raw_ic = f['fields'][start_idx : start_idx+1, :20, :720, :1440]
            current_input = torch.from_numpy((raw_ic - means) / stds).to(device).float()
            raw_targets = f['fields'][start_idx+1 : start_idx+num_steps+1, :20, :720, :1440]
            targets = torch.from_numpy((raw_targets - means) / stds).to(device).float()

            with torch.no_grad():
                for step in range(num_steps):
                    prediction = model(current_input)
                    phys_pred, phys_target = prediction * std_tensor, targets[step:step+1] * std_tensor
                    
                    rmse_per_channel = weighted_rmse_torch_channels(phys_pred, phys_target)
                    # ACC with climatology subtraction
                    acc_per_channel = weighted_acc_torch_channels(prediction - clim_std, targets[step:step+1] - clim_std)
                    
                    all_z500_rmse.append(rmse_per_channel[0, 14].item())
                    all_t850_rmse.append(rmse_per_channel[0, 5].item())
                    all_accs.append(acc_per_channel[0, 14].item()) # Z500 ACC
                    current_input = prediction 

    print(f"\n--- Results for {weight_path} ---")
    print(f"Avg Latency: {avg_lat:.2f} ms (±{std_lat:.2f})")
    print(f"Avg Z500 RMSE: {np.mean(all_z500_rmse):.2f} m^2/s^2")
    print(f"Avg T850 RMSE: {np.mean(all_t850_rmse):.2f} K")
    print(f"Avg Z500 ACC:  {np.mean(all_accs):.4f}")

if __name__ == "__main__":
    class ParamsObject:
        def __init__(self, depth, **entries):
            self.__dict__.update(entries)
            self.depth = depth
            self.in_channels, self.out_channels = 20, 20

    base_params = {'patch_size': 8, 'embed_dim': 768, 'num_blocks': 8, 'img_size': (720, 1440)}

    # Evaluate Backbone (Depth 12)
    backbone_params = ParamsObject(depth=6, **base_params)
    run_full_year_benchmark(backbone_params, "weights/pruned models/pruned_6_layers.ckpt", "data/out_of_sample/2018.h5")
