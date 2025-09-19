#!/usr/bin/env python3
"""
ICRA-ready plots for LiDAR benchmarks.

- Figure 1: FPS (Hz) grouped by n_faces, bars for each LiDAR type per group.
- Figure 2: Total points per second grouped by n_faces, bars for each LiDAR type per group.

Input data are hard-coded from speed-test prints in run_single_scan.py comments.
You can adapt/extend easily to new measurements.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------
# Data definition
# ----------------------
# Groups by mesh face count (n_faces); within each group we have four LiDARs.
# Values are taken from the comment block in examples/run_single_scan.py
# Format per line: '{lidar}': n_faces, avg_ms, rays_count, rays_per_second

@dataclass
class Record:
    n_faces: int
    avg_ms: float
    rays_count: int
    rays_per_sec: float

# Raw text block copied from the comment for clarity
_raw_blocks = [
    {
        'n_faces': 6873,
        'data': {
            'mid360': Record(6873, 0.57, 24000, 42309724.28),
            'vlp32':  Record(6873, 1.57, 120000, 76233506.51),
            'HDL64':  Record(6873, 1.49, 110016, 73942881.00),
            'os128':  Record(6873, 2.25, 260096, 115806638.20),
        }
    },
    {
        'n_faces': 241119,
        'data': {
            'mid360': Record(241119, 1.07, 24000, 22406467.52),
            'vlp32':  Record(241119, 1.47, 120000, 81707220.78),
            'HDL64':  Record(241119, 1.80, 110016, 60996767.86),
            'os128':  Record(241119, 3.11, 260096, 83551995.37),
        }
    },
    {
        'n_faces': 977968,
        'data': {
            'mid360': Record(977968, 0.92, 24000, 26071819.74),
            'vlp32':  Record(977968, 2.01, 120000, 59746501.74),
            'HDL64':  Record(977968, 1.93, 110016, 57075778.80),
            'os128':  Record(977968, 3.07, 260096, 84694286.27),
        }
    },
    {
        'n_faces': 1938266,
        'data': {
            'mid360': Record(1938266, 1.40, 24000, 17183315.01),
            'vlp32':  Record(1938266, 1.96, 120000, 61116957.49),
            'HDL64':  Record(1938266, 2.07, 110016, 53225125.60),
            'os128':  Record(1938266, 4.33, 260096, 60111508.69),
        }
    },
    {
        'n_faces': 2160099,
        'data': {
            'mid360': Record(2160099, 1.59, 24000, 15116878.81),
            'vlp32':  Record(2160099, 2.54, 120000, 47255769.93),
            'HDL64':  Record(2160099, 2.51, 110016, 43806538.02),
            'os128':  Record(2160099, 3.80, 260096, 68381767.72),
        }
    },
    {
        'n_faces': 6625274,
        'data': {
            'mid360': Record(6625274, 1.32, 24000, 18152576.19),
            'vlp32':  Record(6625274, 1.58, 120000, 75979180.00),
            'HDL64':  Record(6625274, 2.00, 110016, 55118439.15),
            'os128':  Record(6625274, 2.89, 260096, 90107434.04),
        }
    },
]

LIDARS = ['mid360', 'vlp32', 'HDL64', 'os128']
COLORS = {
    # Low-saturation, high-lightness pastel colors (paper-friendly)
    'mid360': "#B6E0F7",  # light blue
    'vlp32':  "#C1E1A4",  # light green
    'HDL64':  "#EEDAC0",  # light orange
    'os128':  "#E3C8F0",  # light purple
}

# Matplotlib rcParams tuned for paper-quality figures
mpl.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'legend.fontsize':10,
    'xtick.labelsize':11,
    'ytick.labelsize':11,
    'axes.grid': False,  # disable global grid; we control per-axis grids below
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
})


def _compute_fps(rec: Record) -> float:
    # avg_ms is average time per update in milliseconds
    if rec.avg_ms <= 0:
        return np.nan
    return 1000.0 / rec.avg_ms


def _compute_pts_per_sec(rec: Record) -> float:
    # Prefer reported rays_per_sec; fall back to rays_count * fps to be robust
    if rec.rays_per_sec and rec.rays_per_sec > 0:
        return float(rec.rays_per_sec)
    fps = _compute_fps(rec)
    return rec.rays_count * fps


def _prepare_arrays():
    groups = [blk['n_faces'] for blk in _raw_blocks]
    fps = {lid: [] for lid in LIDARS}
    throughput = {lid: [] for lid in LIDARS}
    for blk in _raw_blocks:
        for lid in LIDARS:
            rec = blk['data'][lid]
            fps[lid].append(_compute_fps(rec))
            throughput[lid].append(_compute_pts_per_sec(rec))
    return np.array(groups), {k: np.array(v) for k, v in fps.items()}, {k: np.array(v) for k, v in throughput.items()}


def _bar_group(ax, x, groups, values_by_lidar, ylabel):
    width = 0.18
    offsets = np.linspace(-1.5*width, 1.5*width, len(LIDARS))
    for i, lid in enumerate(LIDARS):
        ax.bar(x + offsets[i], values_by_lidar[lid], width=width,
               color=COLORS[lid], label=lid if i == 0 else None,
               edgecolor='none', linewidth=0)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v):,}" for v in groups], rotation=0)
    ax.set_xlabel("n_faces (mesh triangle count)")
    # Enable only horizontal (y-axis) dashed grid lines
    ax.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.grid(False)


def make_figures(out_dir: str = None):
    groups, fps, throughput = _prepare_arrays()
    x = np.arange(len(groups), dtype=float)

    fig1, ax1 = plt.subplots(figsize=(5.5, 2.8))
    _bar_group(ax1, x, groups, fps, ylabel='Frame rate (Hz)')
    # Construct a single shared legend (outside, upper center)
    handles = [mpl.patches.Patch(facecolor=COLORS[l], edgecolor='none', label=l) for l in LIDARS]
    ax1.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.25), frameon=False)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5.5, 2.8))
    _bar_group(ax2, x, groups, throughput, ylabel='Total points per second')
    ax2.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.25), frameon=False)
    fig2.tight_layout()

    # Save
    out_dir = out_dir or os.path.dirname(__file__) or '.'
    os.makedirs(out_dir, exist_ok=True)
    f1_base = os.path.join(out_dir, 'icra_fig1_fps')
    f2_base = os.path.join(out_dir, 'icra_fig2_throughput')
    fig1.savefig(f1_base + '.pdf')
    fig1.savefig(f1_base + '.png')
    fig2.savefig(f2_base + '.pdf')
    fig2.savefig(f2_base + '.png')
    print(f"Saved figures to: {out_dir}\n - {f1_base}.pdf/png\n - {f2_base}.pdf/png")

    # 计算并输出平均帧率和平均每秒点数
    all_fps = np.concatenate([fps[lid] for lid in LIDARS])
    tmp_lidars = LIDARS.copy()
    tmp_lidars.remove("mid360")
    print(LIDARS)
    print(tmp_lidars)
    all_throughput = np.concatenate([throughput[lid] for lid in tmp_lidars])
    print(f"平均帧率: {np.mean(all_fps):.2f} Hz")
    print(f"平均每秒点数: {np.mean(all_throughput):.2f}")


if __name__ == '__main__':
    make_figures()
