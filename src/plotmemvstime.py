"""
plot_memory_vs_time.py

Plots GPU memory usage (MB) against training time (s) for two runs on the
same axes -- e.g. LogCL baseline vs. LogCL + STRIDE curriculum -- to visualize
compute overhead directly.

Usage:
    python plot_memory_vs_time.py \
        --baseline logcl_baseline_memory_log.json \
        --framework logcl_curriculum_memory_log.json \
        --output memory_vs_time.png \
        --baseline-label "LogCL (baseline)" \
        --framework-label "LogCL + STRIDE"
"""

import argparse
import json
import matplotlib.pyplot as plt


def load_log(path):
    with open(path, "r") as f:
        data = json.load(f)
    times = [pt["elapsed_time_s"] for pt in data["memory_log"]]
    mems = [pt["gpu_memory_mb"] for pt in data["memory_log"]]
    return times, mems, data["total_train_time_s"], data["peak_gpu_memory_mb"]


def plot_memory_vs_time(baseline_path, framework_path, output_path,
                         baseline_label="Baseline", framework_label="Framework",
                         figsize=(10, 6), time_unit="min"):

    base_t, base_m, base_total_t, base_peak_m = load_log(baseline_path)
    fw_t, fw_m, fw_total_t, fw_peak_m = load_log(framework_path)

    # convert seconds to minutes if requested, for a more readable x-axis
    if time_unit == "min":
        base_t = [t / 60 for t in base_t]
        fw_t = [t / 60 for t in fw_t]
        base_total_t_disp = base_total_t / 60
        fw_total_t_disp = fw_total_t / 60
        x_label = "Training Time (minutes)"
    else:
        base_total_t_disp = base_total_t
        fw_total_t_disp = fw_total_t
        x_label = "Training Time (seconds)"

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(base_t, base_m, color="#4C72B0", linewidth=2.5, marker="o",
             markersize=4, alpha=0.85, label=f"{baseline_label}")
    ax.plot(fw_t, fw_m, color="#C44E52", linewidth=2.5, marker="s",
             markersize=4, alpha=0.85, label=f"{framework_label}")

    # mark peak memory for each with a horizontal reference line
    ax.axhline(base_peak_m, color="#4C72B0", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.axhline(fw_peak_m, color="#C44E52", linestyle=":", linewidth=1.2, alpha=0.6)

    ax.set_xlabel(x_label, fontsize=13, fontweight="bold")
    ax.set_ylabel("GPU Memory Allocated (MB)", fontsize=13, fontweight="bold")
    ax.set_title(
        "GPU Memory Usage over Training Time\n"
        f"{baseline_label}: {base_total_t_disp:.1f} {time_unit}, peak {base_peak_m:.0f} MB   |   "
        f"{framework_label}: {fw_total_t_disp:.1f} {time_unit}, peak {fw_peak_m:.0f} MB",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")
    print(f"{baseline_label}: total={base_total_t_disp:.1f}{time_unit}, peak_mem={base_peak_m:.0f}MB")
    print(f"{framework_label}: total={fw_total_t_disp:.1f}{time_unit}, peak_mem={fw_peak_m:.0f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline *_memory_log.json")
    parser.add_argument("--framework", required=True, help="Path to framework *_memory_log.json")
    parser.add_argument("--output", default="memory_vs_time.png")
    parser.add_argument("--baseline-label", default="LogCL (baseline)")
    parser.add_argument("--framework-label", default="LogCL + STRIDE")
    parser.add_argument("--time-unit", choices=["s", "min"], default="min")
    args = parser.parse_args()

    plot_memory_vs_time(
        args.baseline, args.framework, args.output,
        baseline_label=args.baseline_label,
        framework_label=args.framework_label,
        time_unit=args.time_unit,
    )