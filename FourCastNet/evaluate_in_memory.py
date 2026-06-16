import os
import glob
import argparse
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from networks.afnonet import AFNONet
from networks.afnonet1 import AFNONet as AFNONetQuant
from utils.YParams import YParams


def get_model_list(weights_root):
    base_model = os.path.join(weights_root, "backbone.ckpt")
    pruned_models = sorted(
        glob.glob(os.path.join(weights_root, "fine-tuned models", "*.ckpt"))
    )
    quantized_models = sorted(
        glob.glob(os.path.join(weights_root, "quantized models", "*.ckpt"))
    )

    def _pruned_layers_from_name(path):
        name = os.path.basename(path)
        parts = name.split("_")
        if len(parts) > 2 and parts[0] == "pruned" and parts[1].isdigit():
            return int(parts[1])
        return 0

    pruned_models = sorted(pruned_models, key=_pruned_layers_from_name)
    return [base_model] + pruned_models + quantized_models


def get_model_depth(model_name):
    if "pruned" in model_name:
        try:
            parts = model_name.split("_")
            pruned_idx = parts.index("pruned") if "pruned" in parts else 0
            num_token = parts[pruned_idx + 1]
            num_pruned_layers = int(num_token)
            return 12 - num_pruned_layers
        except (IndexError, ValueError):
            return 12
    return 12


def load_model(model_path, params, device):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    is_quantized = "fp16" in model_name

    params.depth = get_model_depth(model_name)
    if is_quantized:
        model = AFNONetQuant(params).to(device)
    else:
        model = AFNONet(params).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)

    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        if name != "ged":
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


def weighted_rmse(pred, target, weights):
    diff = pred - target
    mse = (diff * diff * weights).mean(dim=(-2, -1))
    rmse = torch.sqrt(mse)
    return rmse


def weighted_acc(pred, target, clim, weights, eps=1e-6):
    pred_anom = pred - clim
    target_anom = target - clim

    cov = (pred_anom * target_anom * weights).mean(dim=(-2, -1))
    var_pred = (pred_anom * pred_anom * weights).mean(dim=(-2, -1))
    var_target = (target_anom * target_anom * weights).mean(dim=(-2, -1))

    denom = torch.sqrt(var_pred * var_target + eps)
    acc = cov / denom
    return acc


def evaluate_model(
    model_path,
    params,
    device,
    data_path,
    stats_root,
    batch_size,
    stride,
    rollout_steps,
    results_dir,
):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"Evaluating {model_name}")

    model = load_model(model_path, params, device)
    model_dtype = next(model.parameters()).dtype

    means = np.load(os.path.join(stats_root, "global_means.npy"))[:, :20, :, :]
    stds = np.load(os.path.join(stats_root, "global_stds.npy"))[:, :20, :, :]
    time_means = np.load(os.path.join(stats_root, "time_means.npy"))[0, :20, :720, :]

    try:
        lat_1d = np.load(os.path.join(stats_root, "latitude.npy"))
    except FileNotFoundError:
        lat_1d = np.linspace(90, -90, 721)

    lat_1d = lat_1d[:720]
    weights = torch.cos(torch.deg2rad(torch.from_numpy(lat_1d).float()))
    weights = (weights / weights.mean()).view(1, 720, 1).to(device)

    means_t = torch.from_numpy(means[:, :, :720, :]).float().to(device)
    stds_t = torch.from_numpy(stds[:, :, :720, :]).float().to(device)
    clim_t = torch.from_numpy(time_means).float().to(device)

    if model_dtype == torch.float16:
        means_t = means_t.half()
        stds_t = stds_t.half()

    variables = ["z500", "t850", "t2m", "u10", "v10"]
    var_indices = {"z500": 14, "t850": 5, "t2m": 2, "u10": 0, "v10": 1}

    metrics = {
        step: {
            var: {"rmse_sum": 0.0, "acc_sum": 0.0, "count": 0}
            for var in variables
        }
        for step in range(rollout_steps)
    }

    with h5py.File(data_path, "r") as f:
        fields = f["fields"]
        n_samples = fields.shape[0]
        init_indices = list(range(0, n_samples - rollout_steps, stride))

        for batch_start in tqdm(
            range(0, len(init_indices), batch_size),
            desc=f"Batches {model_name}",
        ):
            batch_indices = init_indices[batch_start:batch_start + batch_size]

            batch_inputs = []
            for idx in batch_indices:
                batch_inputs.append(fields[idx:idx + 1, :20, :720, :])
            batch_inputs = np.concatenate(batch_inputs, axis=0)

            current_state = torch.from_numpy(batch_inputs).float().to(device)
            current_state = (current_state - means_t) / stds_t
            if model_dtype == torch.float16:
                current_state = current_state.half()

            with torch.no_grad():
                for step in range(rollout_steps):
                    output = model(current_state)

                    pred_phys = output.float() * stds_t.float() + means_t.float()

                    batch_targets = []
                    for idx in batch_indices:
                        target_idx = idx + step + 1
                        batch_targets.append(fields[target_idx:target_idx + 1, :20, :720, :])
                    batch_targets = np.concatenate(batch_targets, axis=0)
                    target_phys = torch.from_numpy(batch_targets).float().to(device)

                    for var in variables:
                        v_idx = var_indices[var]
                        pred_v = pred_phys[:, v_idx, :, :]
                        target_v = target_phys[:, v_idx, :, :]
                        clim_v = clim_t[v_idx, :, :].unsqueeze(0)

                        rmse = weighted_rmse(pred_v, target_v, weights)
                        acc = weighted_acc(pred_v, target_v, clim_v, weights)

                        metrics[step][var]["rmse_sum"] += rmse.mean().item()
                        metrics[step][var]["acc_sum"] += acc.mean().item()
                        metrics[step][var]["count"] += 1

                    current_state = output

                    if step == 0 or (step + 1) % 7 == 0 or (step + 1) == rollout_steps:
                        print(
                            f"{model_name} batch {batch_start // batch_size + 1}: "
                            f"step {step + 1}/{rollout_steps}"
                        )

                    del pred_phys, target_phys, batch_targets

    rows = []
    for step in range(rollout_steps):
        for var in variables:
            count = max(metrics[step][var]["count"], 1)
            rows.append(
                {
                    "model_name": model_name,
                    "lead_time_step": step + 1,
                    "variable": var,
                    "rmse": metrics[step][var]["rmse_sum"] / count,
                    "acc": metrics[step][var]["acc_sum"] / count,
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"results_{model_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="In-memory FourCastNet evaluation")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--data", default=os.path.join(base_dir, "data", "out_of_sample", "2018.h5"))
    parser.add_argument("--stats", default=os.path.join(base_dir, "data", "stats"))
    parser.add_argument("--weights", default=os.path.join(base_dir, "weights"))
    parser.add_argument("--results", default=os.path.join(base_dir, "results"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=28)
    args = parser.parse_args()

    params_yaml = YParams(os.path.join(base_dir, "config", "AFNO.yaml"), "afno_backbone")
    params = {
        "patch_size": 8,
        "embed_dim": 768,
        "num_blocks": params_yaml.num_blocks,
        "img_size": (720, 1440),
        "in_channels": 20,
        "out_channels": 20,
    }

    class ParamsObject:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    params_obj = ParamsObject(**params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = get_model_list(args.weights)
    for model_path in model_list:
        evaluate_model(
            model_path=model_path,
            params=params_obj,
            device=device,
            data_path=args.data,
            stats_root=args.stats,
            batch_size=args.batch_size,
            stride=args.stride,
            rollout_steps=args.rollout_steps,
            results_dir=args.results,
        )


if __name__ == "__main__":
    main()
