#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate default session plots from CSV files created by export_session_csv.py."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session directory that contains exports/csv.",
    )
    parser.add_argument(
        "--csv-subdir",
        type=str,
        default="exports/csv",
        help="CSV subdirectory under the session directory.",
    )
    parser.add_argument(
        "--plot-subdir",
        type=str,
        default="exports/plots",
        help="Plot output subdirectory under the session directory.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window size used for smoothed request-level curves.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def ensure_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column not in df.columns:
        return df.iloc[0:0]
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    out = out.dropna(subset=[column]).sort_values(column)
    return out


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_request_details(csv_dir: Path, plot_dir: Path, rolling_window: int) -> list[Path]:
    path = csv_dir / "request_details.csv"
    df = load_csv(path)
    if df is None:
        return []
    df = ensure_timestamp(df)
    if df.empty:
        return []

    numeric_cols = [
        "cache_hit_ratio",
        "e2e_latency",
        "queue_time",
        "decode_throughput",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    specs = [
        ("cache_hit_ratio", "Cache Hit Ratio", "#2E6F40"),
        ("e2e_latency", "E2E Latency (s)", "#B04A5A"),
        ("queue_time", "Queue Time (s)", "#5D4E9A"),
        ("decode_throughput", "Decode Throughput", "#D97706"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = df[["timestamp"]].copy()
        valid[col] = series
        valid = valid.dropna(subset=[col])
        if valid.empty:
            ax.set_visible(False)
            continue
        valid[f"{col}_rolling"] = valid[col].rolling(window=rolling_window, min_periods=1).mean()
        ax.scatter(valid["timestamp"], valid[col], s=10, alpha=0.28, color=color)
        ax.plot(valid["timestamp"], valid[f"{col}_rolling"], linewidth=2.0, color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle("Request Overview", fontsize=14)
    return [save_figure(fig, plot_dir / "request_overview.png")]


def plot_runtime_metrics(csv_dir: Path, plot_dir: Path) -> list[Path]:
    created: list[Path] = []
    metric_files = sorted(csv_dir.glob("runtime_metrics_*.csv"))
    for path in metric_files:
        role = path.stem.replace("runtime_metrics_", "")
        df = load_csv(path)
        if df is None:
            continue
        df = ensure_timestamp(df)
        if df.empty:
            continue

        specs = [
            ("sglang_gen_throughput", "Gen Throughput"),
            ("sglang_num_queue_reqs", "Queued Requests"),
            ("sglang_num_running_reqs", "Running Requests"),
            ("sglang_token_usage", "Token Usage"),
            ("sglang_cache_hit_rate", "Cache Hit Rate"),
        ]
        available = [spec for spec in specs if spec[0] in df.columns]
        if not available:
            continue

        fig, axes = plt.subplots(
            len(available),
            1,
            figsize=(13, max(3.4 * len(available), 5.0)),
            sharex=True,
            constrained_layout=True,
        )
        if len(available) == 1:
            axes = [axes]
        for ax, (col, title) in zip(axes, available):
            series = pd.to_numeric(df[col], errors="coerce")
            valid = df[["timestamp"]].copy()
            valid[col] = series
            valid = valid.dropna(subset=[col])
            if valid.empty:
                ax.set_visible(False)
                continue
            ax.plot(valid["timestamp"], valid[col], linewidth=1.8)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.suptitle(f"Runtime Metrics: {role}", fontsize=14)
        created.append(save_figure(fig, plot_dir / f"runtime_overview_{role}.png"))
    return created


def plot_gpu_metrics(csv_dir: Path, plot_dir: Path) -> list[Path]:
    path = csv_dir / "gpu_metrics.csv"
    df = load_csv(path)
    if df is None:
        return []
    df = ensure_timestamp(df)
    if df.empty or "gpu_index" not in df.columns:
        return []

    df["gpu_index"] = pd.to_numeric(df["gpu_index"], errors="coerce")
    df["util_gpu_pct"] = pd.to_numeric(df.get("util_gpu_pct"), errors="coerce")
    df["mem_used_mb"] = pd.to_numeric(df.get("mem_used_mb"), errors="coerce")
    df["mem_total_mb"] = pd.to_numeric(df.get("mem_total_mb"), errors="coerce")
    df = df.dropna(subset=["gpu_index"])
    if df.empty:
        return []

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for gpu_index, group in df.groupby("gpu_index"):
        group = group.sort_values("timestamp")
        axes[0].plot(group["timestamp"], group["util_gpu_pct"], label=f"GPU {int(gpu_index)}")
        if group["mem_total_mb"].notna().any():
            mem_pct = (group["mem_used_mb"] / group["mem_total_mb"]) * 100.0
            axes[1].plot(group["timestamp"], mem_pct, label=f"GPU {int(gpu_index)}")
    axes[0].set_title("GPU Utilization (%)")
    axes[1].set_title("GPU Memory Used (%)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle("GPU Overview", fontsize=14)
    return [save_figure(fig, plot_dir / "gpu_overview.png")]


def plot_system_metrics(csv_dir: Path, plot_dir: Path) -> list[Path]:
    path = csv_dir / "system_metrics.csv"
    df = load_csv(path)
    if df is None:
        return []
    df = ensure_timestamp(df)
    if df.empty:
        return []

    for col in ["cpu_util_pct", "mem_util_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    if "cpu_util_pct" in df.columns:
        valid = df.dropna(subset=["cpu_util_pct"])
        axes[0].plot(valid["timestamp"], valid["cpu_util_pct"], linewidth=1.8, color="#0F766E")
    axes[0].set_title("CPU Utilization (%)")
    axes[0].grid(True, alpha=0.25)

    if "mem_util_pct" in df.columns:
        valid = df.dropna(subset=["mem_util_pct"])
        axes[1].plot(valid["timestamp"], valid["mem_util_pct"], linewidth=1.8, color="#7C3AED")
    axes[1].set_title("System Memory Utilization (%)")
    axes[1].grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle("System Overview", fontsize=14)
    return [save_figure(fig, plot_dir / "system_overview.png")]


def plot_process_batches(csv_dir: Path, plot_dir: Path) -> list[Path]:
    path = csv_dir / "process_batch_stats.csv"
    df = load_csv(path)
    if df is None:
        return []
    df = ensure_timestamp(df)
    if df.empty or "batch_phase" not in df.columns:
        return []

    created: list[Path] = []
    df["throughput_tokens_per_sec"] = pd.to_numeric(df.get("throughput_tokens_per_sec"), errors="coerce")
    df["queue_reqs"] = pd.to_numeric(df.get("queue_reqs"), errors="coerce")
    df["running_reqs"] = pd.to_numeric(df.get("running_reqs"), errors="coerce")
    df["token_usage"] = pd.to_numeric(df.get("token_usage"), errors="coerce")
    df["new_tokens"] = pd.to_numeric(df.get("new_tokens"), errors="coerce")
    df["cached_tokens"] = pd.to_numeric(df.get("cached_tokens"), errors="coerce")

    for phase in sorted(df["batch_phase"].dropna().unique()):
        phase_df = df[df["batch_phase"] == phase].copy()
        if phase_df.empty:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
        axes[0].plot(phase_df["timestamp"], phase_df["throughput_tokens_per_sec"], linewidth=1.6)
        axes[0].set_title(f"{phase.title()} Throughput")
        axes[1].plot(phase_df["timestamp"], phase_df["running_reqs"], linewidth=1.6, label="running")
        axes[1].plot(phase_df["timestamp"], phase_df["queue_reqs"], linewidth=1.6, label="queue")
        axes[1].set_title(f"{phase.title()} Running vs Queue")
        axes[1].legend(frameon=False)
        if phase == "prefill":
            axes[2].plot(phase_df["timestamp"], phase_df["new_tokens"], linewidth=1.6, label="new")
            axes[2].plot(phase_df["timestamp"], phase_df["cached_tokens"], linewidth=1.6, label="cached")
            axes[2].set_title("Prefill Tokens")
            axes[2].legend(frameon=False)
        else:
            axes[2].plot(phase_df["timestamp"], phase_df["token_usage"], linewidth=1.6)
            axes[2].set_title("Decode Token Usage")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.suptitle(f"Process Batch Stats: {phase}", fontsize=14)
        created.append(save_figure(fig, plot_dir / f"process_batches_{phase}.png"))
    return created


def main() -> None:
    args = parse_args()
    csv_dir = args.session_dir / args.csv_subdir
    plot_dir = args.session_dir / args.plot_subdir

    created: list[Path] = []
    created.extend(plot_request_details(csv_dir, plot_dir, args.rolling_window))
    created.extend(plot_runtime_metrics(csv_dir, plot_dir))
    created.extend(plot_gpu_metrics(csv_dir, plot_dir))
    created.extend(plot_system_metrics(csv_dir, plot_dir))
    created.extend(plot_process_batches(csv_dir, plot_dir))

    print(f"Generated {len(created)} plot files")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
