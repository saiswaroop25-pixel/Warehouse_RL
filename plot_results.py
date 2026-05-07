#!/usr/bin/env python3
"""
TA-RWARE Pro v2  -  Plot Training Results
==========================================
Run this at ANY time during or after training.
Reads logs/metrics.json and saves all plots to logs/plots/

Usage:
    python plot_results.py
    python plot_results.py --log_dir logs --out_dir logs/plots
"""

import json
import argparse
import os
from pathlib import Path

import numpy as np
from utils.experiment import resolve_run_dir

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    print("ERROR: matplotlib not installed. Run:  pip install matplotlib")
    _MPL = False


# ── Colour palette (dark theme) ──────────────────────────────────────────────
BG      = '#0f0f1a'
PANEL   = '#1a1a2e'
GRID_C  = '#2a2a4a'
COLORS  = {
    'reward':     '#4488ff',
    'delivery':   '#44dd88',
    'completion': '#ffaa33',
    'loss':       '#ff5566',
    'q_value':    '#66aaff',
    'epsilon':    '#aa66ff',
    'smooth':     '#ffffff',
}
AGENT_COLORS = ['#4f8cff', '#ff8c42', '#43c985', '#d66bff', '#00b7c2', '#e85d75', '#a0c25b', '#7a88ff']


def smooth(data, window):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def setup_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    ax.set_title(title,  color='#ccccee', fontsize=12, pad=8)
    ax.set_xlabel(xlabel, color='#aaaacc', fontsize=9)
    ax.set_ylabel(ylabel, color='#aaaacc', fontsize=9)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.grid(True, color=GRID_C, alpha=0.6, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color('#333355')


def plot_single(ax, data, title, xlabel, ylabel, color, w=20):
    setup_ax(ax, title, xlabel, ylabel)
    if not data:
        ax.text(0.5, 0.5, 'No data yet', transform=ax.transAxes,
                color='#666688', ha='center', va='center', fontsize=10)
        return
    x = np.arange(len(data))
    # Raw (faint)
    ax.plot(x, data, color=color, alpha=0.25, linewidth=0.7, label='raw')
    # Smoothed
    sm = smooth(data, min(w, max(1, len(data)//10)))
    xs = np.arange(len(sm))
    ax.plot(xs, sm, color=COLORS['smooth'], linewidth=1.8, label=f'smoothed (w={min(w, len(data)//10)})')
    # Stats annotation
    mn, mx, last = np.mean(data), np.max(data), data[-1]
    ax.axhline(mn, color=color, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(0.98, 0.04,
            f"mean={mn:.2f}  max={mx:.2f}  last={last:.2f}",
            transform=ax.transAxes, color='#aaaacc', fontsize=7.5,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111122',
                      edgecolor='#333355', alpha=0.8))
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor='#333355',
              labelcolor='#aaaacc', loc='upper left')


def plot_agent_lines(ax, series, title, ylabel, w=30):
    setup_ax(ax, title, 'Episode', ylabel)
    if not series:
        ax.text(0.5, 0.5, 'No agent comparison data yet', transform=ax.transAxes,
                color='#666688', ha='center', va='center', fontsize=10)
        return

    for idx, (label, data) in enumerate(series.items()):
        if not data:
            continue
        sm = smooth(data, min(w, max(1, len(data)//10)))
        xs = np.arange(len(sm))
        ax.plot(xs, sm, color=AGENT_COLORS[idx % len(AGENT_COLORS)],
                linewidth=1.8, label=label)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor='#333355',
              labelcolor='#aaaacc', loc='upper left')


def plot_agent_bar(ax, series, title, ylabel, tail=100):
    setup_ax(ax, title, 'Agent', ylabel)
    if not series:
        ax.text(0.5, 0.5, 'No agent comparison data yet', transform=ax.transAxes,
                color='#666688', ha='center', va='center', fontsize=10)
        return

    labels, values = [], []
    for label, data in series.items():
        if not data:
            continue
        labels.append(label)
        values.append(float(np.mean(data[-tail:])))

    if not labels:
        ax.text(0.5, 0.5, 'No agent comparison data yet', transform=ax.transAxes,
                color='#666688', ha='center', va='center', fontsize=10)
        return

    ax.bar(labels, values, color=AGENT_COLORS[:len(labels)], alpha=0.9)
    ax.tick_params(axis='x', rotation=20)


def make_plots(metrics, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_eps = len(metrics.get('episode_rewards', []))
    print(f"  Episodes recorded : {n_eps}")
    print(f"  Training steps    : {len(metrics.get('losses', []))}")

    # ── Figure 1: Episode Performance (3 panels) ─────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig1.patch.set_facecolor(BG)
    fig1.suptitle('Episode Performance', color='#ddddff',
                  fontsize=14, fontweight='bold', y=1.01)

    plot_single(axes[0],
                metrics.get('episode_rewards', []),
                'Episode Reward (total per episode)',
                'Episode', 'Reward',
                COLORS['reward'])

    plot_single(axes[1],
                metrics.get('deliveries', []),
                'Deliveries per Episode',
                'Episode', 'Deliveries',
                COLORS['delivery'])

    plot_single(axes[2],
                metrics.get('completion_rates', []),
                'Task Completion Rate (%)',
                'Episode', 'Completion %',
                COLORS['completion'])

    plt.tight_layout(pad=2.0)
    p1 = out_dir / 'episode_performance.png'
    plt.savefig(p1, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved -> {p1}")

    # ── Figure 2: Training Diagnostics (3 panels) ────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.patch.set_facecolor(BG)
    fig2.suptitle('Training Diagnostics', color='#ddddff',
                  fontsize=14, fontweight='bold', y=1.01)

    plot_single(axes2[0],
                metrics.get('losses', []),
                'Training Loss (Huber)',
                'Update Step', 'Loss',
                COLORS['loss'], w=50)

    plot_single(axes2[1],
                metrics.get('q_values', []),
                'Mean Q-Value',
                'Update Step', 'Q-Value',
                COLORS['q_value'], w=50)

    plot_single(axes2[2],
                metrics.get('epsilons', []),
                'Exploration Rate (Epsilon)',
                'Episode', 'Epsilon',
                COLORS['epsilon'])

    plt.tight_layout(pad=2.0)
    p2 = out_dir / 'training_diagnostics.png'
    plt.savefig(p2, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved -> {p2}")

    # ── Figure 3: Combined summary (6 panels) ────────────────────────────────
    fig3 = plt.figure(figsize=(20, 11))
    fig3.patch.set_facecolor(BG)
    fig3.suptitle('TA-RWARE Pro v2  —  Full Training Summary',
                  color='#ddddff', fontsize=15, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35)
    panel_data = [
        (metrics.get('episode_rewards',  []), 'Episode Reward',      'Episode', 'Reward',     COLORS['reward'],     20),
        (metrics.get('deliveries',       []), 'Deliveries/Episode',  'Episode', 'Count',      COLORS['delivery'],   20),
        (metrics.get('completion_rates', []), 'Completion Rate (%)', 'Episode', '%',           COLORS['completion'], 20),
        (metrics.get('losses',           []), 'Training Loss',       'Step',    'Loss',        COLORS['loss'],       50),
        (metrics.get('q_values',         []), 'Mean Q-Value',        'Step',    'Q',           COLORS['q_value'],    50),
        (metrics.get('epsilons',         []), 'Epsilon',             'Episode', 'Epsilon',     COLORS['epsilon'],    20),
    ]
    for i, (data, title, xl, yl, col, w) in enumerate(panel_data):
        ax = fig3.add_subplot(gs[i // 3, i % 3])
        plot_single(ax, data, title, xl, yl, col, w)

    p3 = out_dir / 'full_summary.png'
    plt.savefig(p3, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved -> {p3}")

    # ── Figure 4: Reward distribution (histogram) ────────────────────────────
    rewards = metrics.get('episode_rewards', [])
    if len(rewards) > 10:
        fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
        fig4.patch.set_facecolor(BG)
        fig4.suptitle('Reward Distribution Analysis',
                      color='#ddddff', fontsize=13, fontweight='bold')

        # Histogram
        ax = axes4[0]
        setup_ax(ax, 'Reward Histogram', 'Reward', 'Count')
        ax.hist(rewards, bins=30, color=COLORS['reward'],
                alpha=0.75, edgecolor='#2233aa')
        ax.axvline(np.mean(rewards), color='white', linewidth=1.5,
                   linestyle='--', label=f'Mean={np.mean(rewards):.1f}')
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor='#333355',
                  labelcolor='#aaaacc')

        # Running mean with std band
        ax2 = axes4[1]
        setup_ax(ax2, 'Reward Rolling Mean ± Std', 'Episode', 'Reward')
        w2 = max(10, len(rewards) // 20)
        rm = smooth(rewards, w2)
        x2 = np.arange(len(rm))
        # Std band
        std_vals = [np.std(rewards[max(0,i-w2):i+w2]) for i in range(len(rm))]
        rm_arr   = np.array(rm)
        std_arr  = np.array(std_vals)
        ax2.fill_between(x2, rm_arr - std_arr, rm_arr + std_arr,
                         color=COLORS['reward'], alpha=0.2, label='±1 std')
        ax2.plot(x2, rm_arr, color=COLORS['reward'], linewidth=2,
                 label=f'Rolling mean (w={w2})')
        ax2.legend(fontsize=8, facecolor=PANEL, edgecolor='#333355',
                   labelcolor='#aaaacc')

        plt.tight_layout(pad=2.0)
        p4 = out_dir / 'reward_analysis.png'
        plt.savefig(p4, dpi=130, facecolor=BG, bbox_inches='tight')
        plt.close(fig4)
        print(f"  Saved -> {p4}")

    # â”€â”€ Figure 5: Agent-to-agent comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    agent_metrics = metrics.get('agent_episode_metrics', {})
    if agent_metrics and agent_metrics.get('reward'):
        fig5, axes5 = plt.subplots(2, 3, figsize=(20, 11))
        fig5.patch.set_facecolor(BG)
        fig5.suptitle('Per-Agent Comparison',
                      color='#ddddff', fontsize=15, fontweight='bold')

        plot_agent_lines(axes5[0, 0], agent_metrics.get('reward', {}),
                         'Agent Reward', 'Reward')
        plot_agent_lines(axes5[0, 1], agent_metrics.get('deliveries', {}),
                         'Agent Deliveries', 'Deliveries')
        plot_agent_lines(axes5[0, 2], agent_metrics.get('assists', {}),
                         'Agent Assists', 'Assists')
        plot_agent_lines(axes5[1, 0], agent_metrics.get('distance', {}),
                         'Agent Distance Travelled', 'Distance')
        plot_agent_lines(axes5[1, 1], agent_metrics.get('collisions', {}),
                         'Agent Collisions', 'Collisions')
        plot_agent_bar(axes5[1, 2], agent_metrics.get('reward', {}),
                       'Recent Avg Reward (last 100 eps)', 'Reward')

        plt.tight_layout(pad=2.0)
        p5 = out_dir / 'agent_comparison.png'
        plt.savefig(p5, dpi=130, facecolor=BG, bbox_inches='tight')
        plt.close(fig5)
        print(f"  Saved -> {p5}")

    print(f"\n  All plots saved to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot TA-RWARE Pro training results')
    parser.add_argument('--log_dir', default='logs',
                        help='Directory containing metrics.json')
    parser.add_argument('--out_dir', default='logs/plots',
                        help='Output directory for plot images')
    args = parser.parse_args()

    if not _MPL:
        return

    log_dir = resolve_run_dir(args.log_dir)
    metrics_path = Path(log_dir) / 'metrics.json'

    print(f"\n{'='*52}")
    print(f"  TA-RWARE Pro v2  -  Plot Results")
    print(f"{'='*52}")
    print(f"  Reading: {metrics_path}")

    if not metrics_path.exists():
        print(f"\n  ERROR: {metrics_path} not found.")
        print("  Training must run for at least 10 episodes first.")
        print("  If training is in progress, wait a bit and try again.")
        return

    with open(metrics_path, encoding='utf-8') as f:
        metrics = json.load(f)

    out_dir = Path(args.out_dir)
    if args.out_dir == 'logs/plots':
        out_dir = Path(log_dir) / 'plots'
    make_plots(metrics, out_dir)

    print(f"\n  Open the PNG files in {Path(out_dir).resolve()}")
    print(f"  to see all training plots.\n")


if __name__ == '__main__':
    main()
