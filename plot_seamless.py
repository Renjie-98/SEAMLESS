#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SEAMLESS dismantling curves")
    parser.add_argument("--dir", default="test_output", help="Output directory from SEAMLESS run")
    parser.add_argument("--protocols", nargs="+", default=None, help="Protocols to plot")
    parser.add_argument("--m-values", type=int, nargs="+", default=None, help="SEAMLESS m values to plot")
    parser.add_argument("--outfile", default=None, help="Output PDF file")
    parser.add_argument("--width", type=float, default=10, help="Plot width in inches")
    parser.add_argument("--height", type=float, default=6, help="Plot height in inches")
    parser.add_argument("--alpha", type=float, default=0.3, help="Shaded region alpha")
    return parser.parse_args()


def make_label(row):
    if row["protocol"] == "seamless" and pd.notna(row["m"]):
        return f"seamless (m={int(row['m'])})"
    return row["protocol"]


def filter_data(data, protocols=None, m_values=None):
    if protocols is not None:
        data = data[data["protocol"].isin(protocols)]

    if m_values is not None:
        data = data[data["m"].isna() | data["m"].isin(m_values)]

    return data


def plot_component(data, mean_col, sd_col, ylabel, title, args):
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    for label, group in data.groupby("protocol_label"):
        group = group.sort_values("p")

        x = group["p"]
        y = group[mean_col]
        sd = group[sd_col].fillna(0)

        ax.plot(x, y, label=label)
        ax.fill_between(
            x,
            (y - sd).clip(lower=0),
            (y + sd).clip(upper=1),
            alpha=args.alpha,
        )

    ax.set_title(title)
    ax.set_xlabel("p = removed / N")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=8,
        frameon=False,
    )

    fig.subplots_adjust(right=0.75)
    return fig


def plot_combined(data, args):
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    protocol_handles = {}

    for label, group in data.groupby("protocol_label"):
        group = group.sort_values("p")

        line_s1, = ax.plot(
            group["p"],
            group["mean_S1_over_N"],
            linestyle="-",
            linewidth=1.5,
            label=label,
        )

        ax.plot(
            group["p"],
            group["mean_S2_over_N"],
            linestyle="--",
            linewidth=1.5,
            color=line_s1.get_color(),
        )

        protocol_handles[label] = line_s1

    ax.set_title("Connected Components vs Fraction Removed (p)")
    ax.set_xlabel("p = removed / N")
    ax.set_ylabel("Component Size / N")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    protocol_legend = ax.legend(
        handles=list(protocol_handles.values()),
        labels=list(protocol_handles.keys()),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=8,
        frameon=False,
    )

    ax.add_artist(protocol_legend)

    component_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2),
    ]

    component_legend = ax.legend(
        handles=component_handles,
        labels=["S1 (LCC)", "S2"],
        loc="upper right",
        fontsize=9,
        frameon=True,
        title="Component",
    )

    ax.add_artist(component_legend)

    fig.subplots_adjust(right=0.75)

    return fig


def print_summary_statistics(args):
    metrics_file = Path(args.dir) / "metrics.csv"

    if not metrics_file.exists():
        return

    metrics = pd.read_csv(metrics_file)
    metrics = filter_data(metrics, args.protocols, args.m_values)

    print("\n=== Summary Statistics ===")

    print("\nAUC S1 (mean ± sd):")
    auc_s1 = (
        metrics.groupby(["protocol", "m"], dropna=False)["AUC_S1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print(auc_s1)

    print("\np at which S1/N <= 0.5 (mean):")
    p05 = (
        metrics.groupby(["protocol", "m"], dropna=False)["p_LCC_0.5"]
        .mean()
        .reset_index(name="mean")
    )
    print(p05)


def main():
    args = parse_args()

    summary_file = Path(args.dir) / "summary.csv"

    if not summary_file.exists():
        raise FileNotFoundError(f"summary.csv not found in {args.dir}")

    print(f"Loading {summary_file}")
    data = pd.read_csv(summary_file)

    print(f"Loaded {len(data)} rows")
    print("Protocols found:", ", ".join(data["protocol"].unique()))

    data = filter_data(data, args.protocols, args.m_values)
    data["protocol_label"] = data.apply(make_label, axis=1)

    outfile = Path(args.outfile) if args.outfile else Path(args.dir) / "plots.pdf"

    print(f"Saving plots to {outfile}")

    fig1 = plot_component(
        data,
        mean_col="mean_S1_over_N",
        sd_col="sd_S1_over_N",
        ylabel="S1 / N",
        title="Largest Connected Component (S1/N) vs Fraction Removed (p)",
        args=args,
    )

    fig2 = plot_component(
        data,
        mean_col="mean_S2_over_N",
        sd_col="sd_S2_over_N",
        ylabel="S2 / N",
        title="Second Largest Connected Component (S2/N) vs Fraction Removed (p)",
        args=args,
    )

    fig3 = plot_combined(data, args)

    with PdfPages(outfile) as pdf:
       pdf.savefig(fig1, bbox_inches="tight")
       pdf.savefig(fig2, bbox_inches="tight")
       pdf.savefig(fig3)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    print(f"Done! Plots saved to {outfile}")

    print_summary_statistics(args)

if __name__ == "__main__":
    main()