import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def categorize_model(model_name):
    if model_name.startswith("pruned"):
        return "Pruned"
    if "fp16" in model_name or "fp32" in model_name:
        return "Quantized"
    return "Base"


def var_safe_name(name):
    return name.replace("/", "_").replace(" ", "_")


def plot_variable_for_model(df, backbone_name, model_name, var, output_dir):
    df_var = df[(df["model_name"] == model_name) & (df["variable"] == var)].copy()
    df_var["lead_time_days"] = df_var["lead_time_step"] * 6.0 / 24.0

    group = categorize_model(model_name)
    color = "black"
    linestyle = "-"
    if group == "Pruned":
        color = sns.color_palette("Blues", 6)[2]
        linestyle = "--"
    elif group == "Quantized":
        color = sns.color_palette("Oranges", 6)[2]
        linestyle = ":"

    plt.figure(figsize=(10, 6))

    backbone_df = df[(df["model_name"] == backbone_name) & (df["variable"] == var)].copy()
    backbone_df["lead_time_days"] = backbone_df["lead_time_step"] * 6.0 / 24.0

    plt.plot(
        backbone_df["lead_time_days"],
        backbone_df["rmse"],
        label=backbone_name,
        color="black",
        linewidth=2.0,
    )

    plt.plot(
        df_var["lead_time_days"],
        df_var["rmse"],
        label=model_name,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
    )

    plt.title(f"RMSE vs Lead Time: {var.upper()} ({model_name} vs {backbone_name})")
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Latitude-Weighted RMSE")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, frameon=False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"rmse_{var}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE curves from per-model CSVs")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing results_*.csv files",
    )
    parser.add_argument(
        "--output",
        default="plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.results_dir, "results_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No results_*.csv files found in {args.results_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    required_cols = {"model_name", "lead_time_step", "variable", "rmse", "acc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSVs missing required columns: {sorted(missing)}")

    model_names = sorted(df["model_name"].unique())

    base_models = [m for m in model_names if categorize_model(m) == "Base"]
    backbone_name = "backbone" if "backbone" in base_models else (base_models[0] if base_models else None)
    if backbone_name is None:
        raise ValueError("No backbone model found in CSVs.")

    for model_name in model_names:
        if model_name == backbone_name:
            continue
        model_dir = os.path.join(args.output, var_safe_name(model_name))
        for var in ["z500", "t850", "t2m", "u10", "v10"]:
            plot_variable_for_model(df, backbone_name, model_name, var, model_dir)


if __name__ == "__main__":
    main()
