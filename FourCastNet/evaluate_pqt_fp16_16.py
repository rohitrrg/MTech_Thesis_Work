import torch
import h5py
import numpy as np
from collections import OrderedDict
from networks.afnonet1 import AFNONet
from utils.weighted_acc_rmse import (
    weighted_rmse_torch_channels,
    weighted_acc_torch_channels
)


# Load Quantized Model
# --------------------------------------------------
def load_quantized_model(model, checkpoint_file, device):
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False
    )

    state_dict = checkpoint.get('model_state', checkpoint)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    model = model.half()  # Convert model to FP16

    # Explicitly convert parameters and buffers to FP16
    for name, param in model.named_parameters():
        param.data = param.data.half()
        print(f"Converted {name} to FP16")  # Debugging
    for name, buf in model.named_buffers():
        buf.data = buf.data.half()
        print(f"Converted buffer {name} to FP16")  # Debugging

    model.eval()

    return model


# Latency Measurement (works for quantized too)
# --------------------------------------------------
def measure_latency(model, device, repetitions=100, warm_up=20):

    dummy_input = torch.randn(1, 20, 720, 1440).to(device).half()
    print("Dummy Dtype:", dummy_input.dtype)
    # Warm-up
    for _ in range(warm_up):
        _ = model(dummy_input)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    timings = []

    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    return np.mean(timings), np.std(timings)


# Full-Year Evaluation (same as baseline)
def run_full_year_quantized_benchmark(
    params_obj,
    weight_path,
    data_path,
    num_steps=1,
    stride=36
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AFNONet(params_obj).to(device)
    model = load_quantized_model(model, weight_path, device)

    # Verification check
    first_param_dtype = next(model.parameters()).dtype
    print(f"Model internal precision: {first_param_dtype}")
    # Latency
    # ------------------------------
    avg_lat, std_lat = measure_latency(model, device)

    # Load normalization stats
    # ------------------------------
    means = np.load("data/stats/global_means.npy")[0, :20].reshape(20, 1, 1)
    stds = np.load("data/stats/global_stds.npy")[0, :20].reshape(20, 1, 1)

    std_tensor = torch.as_tensor(stds).to(device).view(1, 20, 1, 1)

    # Climatology for ACC
    time_means_raw = np.load("data/stats/time_means.npy")[0, :20].reshape(20, 721, 1440)
    time_means = time_means_raw[:, 0:720, :]

    clim_std = torch.as_tensor(
        (time_means - means) / stds
    ).to(device).view(1, 20, 720, 1440).float()

    all_z500_rmse, all_t850_rmse, all_accs = [], [], []

    # Loop over full 2018 dataset
    # ------------------------------
    with h5py.File(data_path, 'r') as f:

        n_samples = f['fields'].shape[0]

        for start_idx in range(0, n_samples - num_steps, stride):

            raw_ic = f['fields'][start_idx:start_idx+1, :20, :720, :1440]
            current_input = torch.from_numpy(
                (raw_ic - means) / stds
            ).to(device).half()

            

            raw_targets = f['fields'][
                start_idx+1:start_idx+num_steps+1,
                :20,
                :720,
                :1440
            ]

            targets = torch.from_numpy(
                (raw_targets - means) / stds
            ).to(device).half()
            

            with torch.no_grad():

                for step in range(num_steps):

                    prediction = model(current_input)

                    phys_pred = prediction * std_tensor
                    phys_target = targets[step:step+1] * std_tensor

                    rmse_per_channel = weighted_rmse_torch_channels(
                        phys_pred,
                        phys_target
                    )

                    acc_per_channel = weighted_acc_torch_channels(
                        prediction - clim_std,
                        targets[step:step+1] - clim_std
                    )

                    # Z500 index = 14
                    # T850 index = 5
                    all_z500_rmse.append(rmse_per_channel[0, 14].item())
                    all_t850_rmse.append(rmse_per_channel[0, 5].item())
                    all_accs.append(acc_per_channel[0, 14].item())

                    current_input = prediction

    # Report
    # ------------------------------
    print(f"\n--- Quantized Model Results ---")
    print(f"Checkpoint: {weight_path}")
    print(f"Avg Latency: {avg_lat:.2f} ms (±{std_lat:.2f})")
    print(f"Avg Z500 RMSE: {np.mean(all_z500_rmse):.2f}")
    print(f"Avg T850 RMSE: {np.mean(all_t850_rmse):.2f}")
    print(f"Avg Z500 ACC:  {np.mean(all_accs):.4f}")
    print("-" * 50)


# Run
# --------------------------------------------------
if __name__ == "__main__":

    class ParamsObject:
        def __init__(self, depth, **entries):
            self.__dict__.update(entries)
            self.depth = depth
            self.in_channels = 20
            self.out_channels = 20

    base_params = {
        'patch_size': 8,
        'embed_dim': 768,
        'num_blocks': 8,
        'img_size': (720, 1440)
    }

    quantized_params = ParamsObject(depth=6, **base_params)

    run_full_year_quantized_benchmark(
        quantized_params,
        "weights/quantized models/pruned_6_fp16.ckpt",
        "data/out_of_sample/2018.h5"
    )
