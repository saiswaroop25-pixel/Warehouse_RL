#!/usr/bin/env python3
"""Streamlit frontend for the TA-RWARE warehouse model."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from agents.baseline_policies import greedy_actions, random_actions
from agents.dqn_agent import DQNAgent
from envs.warehouse_env import AgentType, Cell, Phase, WarehouseEnv
from utils.experiment import resolve_run_dir


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_MODEL_DIR = ROOT / "models"
ACTION_NAMES = ["North", "South", "East", "West", "Wait"]
POLICIES = ["DQN", "Greedy", "Random"]
THEME = {
    "bg": "#f7f8fb",
    "surface": "#ffffff",
    "ink": "#172033",
    "muted": "#667085",
    "line": "#d8dee9",
    "empty": "#f4f6f8",
    "wall": "#1f2937",
    "rack": "#697586",
    "rack_active": "#c48a2c",
    "goal": "#314f7d",
    "charge": "#b89b2e",
    "agv": "#243b63",
    "picker": "#506174",
    "path": "#6b7fa4",
    "good": "#2f7d5a",
    "warn": "#b7791f",
    "bad": "#a94442",
}


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {THEME["bg"]};
            color: {THEME["ink"]};
        }}
        [data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid {THEME["line"]};
        }}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
            color: {THEME["ink"]};
            letter-spacing: 0;
        }}
        .block-container {{
            padding-top: 1.5rem;
            max-width: 1380px;
        }}
        div[data-testid="stMetric"] {{
            background: {THEME["surface"]};
            border: 1px solid {THEME["line"]};
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {THEME["muted"]};
            font-size: 0.84rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: {THEME["ink"]};
            font-size: 1.45rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.25rem;
            border-bottom: 1px solid {THEME["line"]};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {THEME["muted"]};
            border-radius: 6px 6px 0 0;
            padding: 0.65rem 0.9rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {THEME["ink"]};
            background: #ffffff;
            border: 1px solid {THEME["line"]};
            border-bottom: 1px solid #ffffff;
        }}
        .stButton > button {{
            border-radius: 6px;
            border: 1px solid {THEME["agv"]};
            background: {THEME["agv"]};
            color: white;
            font-weight: 600;
        }}
        .stProgress > div > div > div > div {{
            background-color: {THEME["agv"]};
        }}
        .section-note {{
            color: {THEME["muted"]};
            font-size: 0.94rem;
            margin-top: -0.45rem;
            margin-bottom: 1rem;
        }}
        .dashboard-title {{
            color: {THEME["ink"]};
            font-size: 1.85rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            letter-spacing: 0;
        }}
        .dashboard-subtitle {{
            color: {THEME["muted"]};
            font-size: 0.98rem;
            margin-bottom: 1.1rem;
        }}
        .legend-row {{
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
            color: {THEME["muted"]};
            font-size: 0.9rem;
            margin: 0.25rem 0 1rem 0;
        }}
        .legend-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }}
        .legend-dot {{
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 3px;
            display: inline-block;
            border: 1px solid {THEME["line"]};
        }}
        .policy-card {{
            background: #ffffff;
            border: 1px solid {THEME["line"]};
            border-radius: 8px;
            padding: 1rem;
            min-height: 9.5rem;
        }}
        .policy-name {{
            color: {THEME["ink"]};
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .policy-rank {{
            color: {THEME["muted"]};
            font-size: 0.82rem;
            margin-bottom: 0.75rem;
        }}
        .policy-score {{
            color: {THEME["ink"]};
            font-size: 1.85rem;
            font-weight: 750;
            line-height: 1;
        }}
        .policy-meta {{
            color: {THEME["muted"]};
            font-size: 0.86rem;
            margin-top: 0.6rem;
            line-height: 1.55;
        }}
        .callout {{
            background: #eef3f8;
            border: 1px solid #ced8e5;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            color: {THEME["ink"]};
            margin: 0.8rem 0 1rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_model_dirs() -> list[Path]:
    dirs = [DEFAULT_MODEL_DIR]
    run_root = DEFAULT_MODEL_DIR / "runs"
    if run_root.exists():
        dirs.extend(sorted([p for p in run_root.iterdir() if p.is_dir()], reverse=True))
    return dirs


def list_config_files() -> list[Path]:
    files = sorted((ROOT / "configs").glob("*.yaml"))
    return files or [DEFAULT_CONFIG]


def resolve_eval_config(model_dir: Path, fallback_cfg: Path) -> tuple[dict, Path]:
    candidates = [
        model_dir / "config_snapshot.yaml",
        ROOT / "logs" / "runs" / model_dir.name / "config_snapshot.yaml",
        fallback_cfg,
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_cfg(candidate), candidate
    return load_cfg(DEFAULT_CONFIG), DEFAULT_CONFIG


def parameter_slider(container, label, value, min_value, max_value, step, help_text=None):
    value = container.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        help=help_text,
    )
    ratio = (float(value) - float(min_value)) / max(float(max_value - min_value), 1e-9)
    container.progress(min(max(ratio, 0.0), 1.0))
    return value


def build_sidebar_config(base_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    env = cfg["environment"]
    rewards = cfg["rewards"]

    with st.sidebar.expander("Environment", expanded=True) as section:
        env["grid_rows"] = parameter_slider(section, "Rack rows", int(env["grid_rows"]), 2, 8, 1)
        env["grid_cols"] = parameter_slider(section, "Rack columns", int(env["grid_cols"]), 3, 10, 1)
        env["n_agvs"] = parameter_slider(section, "AGVs", int(env["n_agvs"]), 1, 6, 1)
        env["n_pickers"] = parameter_slider(section, "Pickers", int(env["n_pickers"]), 0, 4, 1)
        env["max_steps"] = parameter_slider(section, "Max episode steps", int(env["max_steps"]), 20, 1000, 10)
        env["request_queue_size"] = parameter_slider(section, "Order queue size", int(env["request_queue_size"]), 1, 12, 1)
        env["task_injection_interval"] = parameter_slider(
            section,
            "Task injection interval",
            int(env["task_injection_interval"]),
            10,
            300,
            5,
        )
        env["picker_service_radius"] = parameter_slider(
            section,
            "Picker service radius",
            int(env.get("picker_service_radius", 2)),
            1,
            5,
            1,
        )

    with st.sidebar.expander("Battery", expanded=False) as section:
        env["battery_capacity"] = parameter_slider(
            section,
            "Battery capacity",
            float(env["battery_capacity"]),
            50.0,
            500.0,
            10.0,
        )
        env["battery_drain_move"] = parameter_slider(
            section,
            "Move drain",
            float(env["battery_drain_move"]),
            0.05,
            2.0,
            0.05,
        )
        env["battery_drain_idle"] = parameter_slider(
            section,
            "Idle drain",
            float(env["battery_drain_idle"]),
            0.01,
            1.0,
            0.01,
        )
        env["battery_recharge_rate"] = parameter_slider(
            section,
            "Recharge rate",
            float(env["battery_recharge_rate"]),
            1.0,
            40.0,
            1.0,
        )
        env["battery_low_threshold"] = parameter_slider(
            section,
            "Low battery threshold",
            float(env.get("battery_low_threshold", 0.2)),
            0.05,
            0.8,
            0.05,
        )

    with st.sidebar.expander("Rewards", expanded=False) as section:
        rewards["delivery_complete"] = parameter_slider(
            section,
            "Delivery reward",
            float(rewards["delivery_complete"]),
            1.0,
            80.0,
            1.0,
        )
        rewards["pickup_item"] = parameter_slider(
            section,
            "Pickup reward",
            float(rewards["pickup_item"]),
            0.0,
            30.0,
            0.5,
        )
        rewards["progress"] = parameter_slider(
            section,
            "Progress reward",
            float(rewards["progress"]),
            0.0,
            3.0,
            0.1,
        )
        rewards["move_away"] = parameter_slider(
            section,
            "Move-away penalty",
            float(rewards["move_away"]),
            -5.0,
            0.0,
            0.1,
        )
        rewards["collision"] = parameter_slider(
            section,
            "Collision penalty",
            float(rewards["collision"]),
            -5.0,
            0.0,
            0.1,
        )
        rewards["time_step"] = parameter_slider(
            section,
            "Step penalty",
            float(rewards["time_step"]),
            -0.2,
            0.0,
            0.001,
        )
        rewards["team_bonus"] = parameter_slider(
            section,
            "Team bonus",
            float(rewards["team_bonus"]),
            0.0,
            120.0,
            1.0,
        )
    return cfg


def load_dqn_agents(env: WarehouseEnv, cfg: dict, model_dir: Path, device: str | None = None):
    agents = []
    errors = []
    for agent in env.agents:
        dqn = DQNAgent(env.state_dim, cfg["agent"]["action_dim"], cfg, device=device)
        dqn.eps = 0.0
        candidates = (
            [model_dir / f"agent{agent.id}_best.pt", model_dir / f"agent{agent.id}_final.pt"]
            + sorted(model_dir.glob(f"agent{agent.id}_ep*.pt"), reverse=True)
        )
        checkpoint = next((p for p in candidates if p.exists()), None)
        if checkpoint is None:
            errors.append(f"agent{agent.id}: no checkpoint found")
            agents.append(None)
            continue
        try:
            dqn.load(str(checkpoint))
            agents.append(dqn)
        except Exception as exc:
            errors.append(f"agent{agent.id}: {checkpoint.name} could not load ({exc})")
            agents.append(None)
    usable = all(agent is not None for agent in agents)
    return agents, usable, errors


def dqn_actions(dqns, obs, env):
    if not dqns or any(agent is None for agent in dqns):
        return greedy_actions(env)
    return [int(dqns[i].act(obs[i], train=False)) for i in range(env.n_agents)]


def action_for_policy(policy: str, env: WarehouseEnv, obs, dqns, rng):
    if policy == "DQN":
        return dqn_actions(dqns, obs, env)
    if policy == "Greedy":
        return greedy_actions(env)
    return random_actions(env, rng=rng)


def draw_environment(env: WarehouseEnv, info: dict, title: str = ""):
    color_map = {
        Cell.EMPTY: THEME["empty"],
        Cell.WALL: THEME["wall"],
        Cell.RACK: THEME["rack"],
        Cell.GOAL: THEME["goal"],
        Cell.CHARGE: THEME["charge"],
    }
    fig_w = min(12, max(6, env.W * 0.55))
    fig_h = min(10, max(5, env.H * 0.55))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=THEME["surface"])
    ax.set_facecolor(THEME["surface"])

    for r in range(env.H):
        for c in range(env.W):
            cell = Cell(int(env.grid[r, c]))
            ax.add_patch(
                plt.Rectangle(
                    (c, env.H - r - 1),
                    1,
                    1,
                    color=color_map[cell],
                    ec="#e3e8ef",
                    lw=0.35,
                )
            )

    requested = {tuple(o.rack_pos) for o in env.orders if not o.done}
    for c, r in requested:
        ax.add_patch(plt.Rectangle((c + 0.08, env.H - r - 0.92), 0.84, 0.84, color=THEME["rack_active"]))
        ax.text(c + 0.5, env.H - r - 0.5, "!", ha="center", va="center", color="white", weight="bold")

    for order in env.orders:
        if order.done or order.agv_id < 0:
            continue
        agv = next((a for a in env.agents if a.id == order.agv_id), None)
        if agv is None:
            continue
        target = order.goal_pos if agv.phase == Phase.TO_GOAL else order.rack_pos
        ax.plot(
            [agv.pos[0] + 0.5, target[0] + 0.5],
            [env.H - agv.pos[1] - 0.5, env.H - target[1] - 0.5],
            color=THEME["path"],
            lw=1.2,
            alpha=0.35,
        )

    for agent in env.agents:
        x = agent.pos[0] + 0.5
        y = env.H - agent.pos[1] - 0.5
        if agent.type == AgentType.AGV:
            color = THEME["agv"]
            marker = "h"
            label = f"A{agent.id}"
        else:
            color = THEME["picker"]
            marker = "D"
            label = f"P{agent.id - env.n_agvs}"
        ax.scatter(x, y, s=430, marker=marker, color=color, edgecolor="#ffffff", linewidth=1.4, zorder=4)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=5)
        battery_ratio = max(0.0, min(1.0, agent.battery / max(agent.batt_cap, 1e-9)))
        ax.add_patch(plt.Rectangle((agent.pos[0] + 0.12, env.H - agent.pos[1] - 0.06), 0.76, 0.06, color="#1f2937"))
        ax.add_patch(
            plt.Rectangle(
                (agent.pos[0] + 0.12, env.H - agent.pos[1] - 0.06),
                0.76 * battery_ratio,
                0.06,
                color=THEME["good"] if battery_ratio > 0.5 else THEME["warn"] if battery_ratio > 0.2 else THEME["bad"],
            )
        )

    ax.set_xlim(0, env.W)
    ax.set_ylim(0, env.H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"Step {info['steps']} / {env.max_steps}", fontsize=12, pad=8, color=THEME["ink"], weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def run_episode(cfg: dict, model_dir: Path, policy: str, seed: int, animate: bool, render_every: int, delay: float):
    env = WarehouseEnv(cfg)
    dqns, usable_dqn, load_errors = load_dqn_agents(env, cfg, model_dir) if policy == "DQN" else ([], False, [])
    if policy == "DQN" and not usable_dqn:
        st.warning("DQN checkpoints do not match this setup, so actions fall back to the greedy policy.")
        with st.expander("Checkpoint details"):
            for error in load_errors:
                st.write(error)

    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step_rows = []
    agent_rows = []
    env_placeholder = st.empty()
    progress_placeholder = st.empty()

    while not done:
        actions = action_for_policy(policy, env, obs, dqns, rng)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        total_reward += float(sum(rewards))

        step_rows.append(
            {
                "step": info["steps"],
                "reward": float(sum(rewards)),
                "total_reward": total_reward,
                "deliveries": int(info["deliveries"]),
                "pending_orders": int(info["pending_orders"]),
                "collisions": int(sum(am["collisions"] for am in info["agent_metrics"])),
                "charge_visits": int(sum(am["charge_visits"] for am in info["agent_metrics"])),
            }
        )
        for idx, agent in enumerate(env.agents):
            agent_rows.append(
                {
                    "step": info["steps"],
                    "agent": f"Agent {idx}",
                    "type": "AGV" if idx < env.n_agvs else "Picker",
                    "x": agent.pos[0],
                    "y": agent.pos[1],
                    "battery": agent.battery,
                    "reward": float(rewards[idx]),
                    "action": ACTION_NAMES[int(actions[idx])] if int(actions[idx]) < len(ACTION_NAMES) else str(actions[idx]),
                }
            )

        if animate and (info["steps"] % render_every == 0 or done):
            fig = draw_environment(env, info, f"{policy} policy - step {info['steps']}")
            env_placeholder.pyplot(fig, clear_figure=True)
            plt.close(fig)
            progress_placeholder.progress(min(info["steps"] / max(env.max_steps, 1), 1.0))
            time.sleep(delay)

    if not animate:
        fig = draw_environment(env, info, f"{policy} policy - final state")
        env_placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)
        progress_placeholder.progress(1.0)

    return env, info, pd.DataFrame(step_rows), pd.DataFrame(agent_rows)


def run_policy_summary(cfg: dict, model_dir: Path, policy: str, seed: int):
    env = WarehouseEnv(cfg)
    dqns, usable_dqn, errors = load_dqn_agents(env, cfg, model_dir) if policy == "DQN" else ([], True, [])
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    agent_rewards = [0.0] * env.n_agents

    while not done:
        actions = action_for_policy(policy, env, obs, dqns, rng)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        steps += 1
        total_reward += float(sum(rewards))
        for idx, reward in enumerate(rewards):
            agent_rewards[idx] += float(reward)

    completion = info["deliveries"] / max(info["total_orders"], 1) * 100.0
    collisions = sum(am["collisions"] for am in info["agent_metrics"])
    charge_visits = sum(am["charge_visits"] for am in info["agent_metrics"])
    cycle_times = info.get("completed_order_times", [])
    return {
        "policy": policy,
        "seed": seed,
        "reward": total_reward,
        "deliveries": int(info["deliveries"]),
        "total_orders": int(info["total_orders"]),
        "completion_rate": completion,
        "steps": int(steps),
        "throughput_per_100_steps": info["deliveries"] / max(steps, 1) * 100.0,
        "collisions": int(collisions),
        "charge_visits": int(charge_visits),
        "avg_order_cycle_time": float(np.mean(cycle_times)) if cycle_times else 0.0,
        "dqn_loaded": bool(usable_dqn),
        "load_errors": "; ".join(errors),
    }


def evaluate_policy_suite(cfg: dict, model_dir: Path, episodes: int, base_seed: int):
    rows = []
    for policy in POLICIES:
        for ep_idx in range(episodes):
            rows.append(run_policy_summary(cfg, model_dir, policy, base_seed + ep_idx))
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("policy", as_index=False)
        .agg(
            episodes=("seed", "count"),
            avg_reward=("reward", "mean"),
            reward_std=("reward", "std"),
            avg_deliveries=("deliveries", "mean"),
            avg_completion_rate=("completion_rate", "mean"),
            avg_steps=("steps", "mean"),
            avg_throughput_per_100_steps=("throughput_per_100_steps", "mean"),
            avg_collisions=("collisions", "mean"),
            avg_charge_visits=("charge_visits", "mean"),
            avg_order_cycle_time=("avg_order_cycle_time", "mean"),
        )
        .fillna(0.0)
    )
    policy_order = {policy: idx for idx, policy in enumerate(POLICIES)}
    summary["_order"] = summary["policy"].map(policy_order)
    summary = summary.sort_values("_order").drop(columns="_order")
    summary["rank"] = summary["avg_reward"].rank(method="dense", ascending=False).astype(int)
    return raw, summary


def plot_policy_comparison(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=THEME["surface"])
    metrics = [
        ("avg_reward", "Average Reward"),
        ("avg_deliveries", "Average Deliveries"),
        ("avg_completion_rate", "Completion Rate (%)"),
        ("avg_throughput_per_100_steps", "Throughput / 100 Steps"),
        ("avg_collisions", "Average Collisions"),
        ("avg_order_cycle_time", "Order Cycle Time"),
    ]
    colors = [THEME["agv"], THEME["picker"], THEME["rack_active"]]
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.bar(summary_df["policy"], summary_df[key], color=colors, width=0.55)
        ax.set_title(title, color=THEME["ink"])
        ax.grid(True, axis="y", color=THEME["line"], alpha=0.55)
        ax.set_facecolor(THEME["surface"])
        ax.tick_params(colors=THEME["muted"])
        for spine in ax.spines.values():
            spine.set_color(THEME["line"])
    fig.tight_layout()
    return fig


def render_policy_cards(summary_df: pd.DataFrame):
    ranked = summary_df.sort_values(["rank", "policy"])
    cols = st.columns(len(ranked))
    for col, (_, row) in zip(cols, ranked.iterrows()):
        completion = row["avg_completion_rate"]
        throughput = row["avg_throughput_per_100_steps"]
        collisions = row["avg_collisions"]
        col.markdown(
            f"""
            <div class="policy-card">
              <div class="policy-name">{row["policy"]}</div>
              <div class="policy-rank">Rank #{int(row["rank"])} by average reward</div>
              <div class="policy-score">{row["avg_reward"]:+.1f}</div>
              <div class="policy-meta">
                Deliveries: {row["avg_deliveries"]:.2f}<br/>
                Completion: {completion:.1f}%<br/>
                Throughput: {throughput:.2f} / 100 steps<br/>
                Collisions: {collisions:.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_leaderboard(summary_df: pd.DataFrame):
    leaderboard = summary_df.sort_values(["rank", "policy"]).copy()
    leaderboard = leaderboard[
        [
            "rank",
            "policy",
            "avg_reward",
            "reward_std",
            "avg_deliveries",
            "avg_completion_rate",
            "avg_throughput_per_100_steps",
            "avg_collisions",
            "avg_charge_visits",
            "avg_order_cycle_time",
        ]
    ]
    leaderboard = leaderboard.rename(
        columns={
            "rank": "Rank",
            "policy": "Policy",
            "avg_reward": "Avg Reward",
            "reward_std": "Reward Std",
            "avg_deliveries": "Avg Deliveries",
            "avg_completion_rate": "Completion %",
            "avg_throughput_per_100_steps": "Throughput / 100",
            "avg_collisions": "Avg Collisions",
            "avg_charge_visits": "Charge Visits",
            "avg_order_cycle_time": "Order Cycle Time",
        }
    )
    st.dataframe(leaderboard.round(3), width="stretch", hide_index=True)


def plot_episode(step_df: pd.DataFrame, agent_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=THEME["surface"])
    axes = axes.flat
    axes[0].plot(step_df["step"], step_df["total_reward"], color=THEME["agv"], lw=2)
    axes[0].set_title("Cumulative Reward")
    axes[1].step(step_df["step"], step_df["deliveries"], color=THEME["good"], lw=2)
    axes[1].set_title("Deliveries")
    axes[2].plot(step_df["step"], step_df["collisions"], color=THEME["bad"], lw=2)
    axes[2].set_title("Total Collisions")

    line_styles = ["-", "--", "-.", ":"]
    for label, group in agent_df.groupby("agent"):
        idx = int(label.split()[-1])
        axes[3].plot(
            group["step"],
            group["battery"],
            label=label,
            lw=1.8,
            color=THEME["agv"] if group["type"].iloc[0] == "AGV" else THEME["picker"],
            linestyle=line_styles[idx % len(line_styles)],
        )
    axes[3].set_title("Battery Levels")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.set_facecolor(THEME["surface"])
        ax.grid(True, color=THEME["line"], alpha=0.55)
        ax.set_xlabel("Step")
        ax.tick_params(colors=THEME["muted"])
        ax.title.set_color(THEME["ink"])
        for spine in ax.spines.values():
            spine.set_color(THEME["line"])
    fig.tight_layout()
    return fig


def show_saved_plots(model_dir: Path):
    log_dir = ROOT / "logs" / "runs" / model_dir.name if model_dir.name != "models" else ROOT / "logs"
    plot_dirs = [log_dir / "plots", log_dir, ROOT / "logs" / "plots"]
    images = []
    for plot_dir in plot_dirs:
        if plot_dir.exists():
            images.extend(sorted(plot_dir.glob("*.png")))
    unique = []
    seen = set()
    for image in images:
        if image.resolve() not in seen:
            unique.append(image)
            seen.add(image.resolve())
    if not unique:
        st.info("No saved plot artifacts found yet.")
        return
    cols = st.columns(2)
    for i, image in enumerate(unique[:8]):
        cols[i % 2].image(str(image), caption=str(image.relative_to(ROOT)), width="stretch")


def show_saved_metrics(model_dir: Path):
    candidates = [
        ROOT / "logs" / "runs" / model_dir.name / "metrics.json",
        ROOT / "logs" / "metrics.json",
    ]
    metrics_path = next((p for p in candidates if p.exists()), None)
    if metrics_path is None:
        st.info("No metrics.json found for the selected run.")
        return
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    rewards = metrics.get("episode_rewards", [])
    deliveries = metrics.get("deliveries", [])
    epsilons = metrics.get("epsilons", [])
    if not rewards:
        st.info("The metrics file does not contain episode rewards yet.")
        return
    df = pd.DataFrame(
        {
            "episode": np.arange(1, len(rewards) + 1),
            "reward": rewards,
            "deliveries": deliveries[: len(rewards)] if deliveries else [0] * len(rewards),
            "epsilon": epsilons[: len(rewards)] if epsilons else [0] * len(rewards),
        }
    )
    st.line_chart(df.set_index("episode")[["reward", "deliveries", "epsilon"]])


def main():
    st.set_page_config(page_title="TA-RWARE Warehouse Dashboard", layout="wide")
    inject_css()
    st.markdown('<div class="dashboard-title">TA-RWARE Warehouse Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Evaluate trained dispatch policies, tune simulation parameters, and watch warehouse agents move through the grid.</div>',
        unsafe_allow_html=True,
    )

    model_dirs = list_model_dirs()
    config_files = list_config_files()
    selected_model = st.sidebar.selectbox(
        "Model directory",
        model_dirs,
        index=0,
        format_func=lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
    )
    selected_model = resolve_run_dir(selected_model)
    selected_config = st.sidebar.selectbox(
        "Fallback config",
        config_files,
        index=config_files.index(DEFAULT_CONFIG) if DEFAULT_CONFIG in config_files else 0,
        format_func=lambda p: str(p.relative_to(ROOT)),
    )
    base_cfg, used_cfg_path = resolve_eval_config(selected_model, selected_config)
    st.sidebar.caption(f"Using config: {used_cfg_path.relative_to(ROOT) if used_cfg_path.is_relative_to(ROOT) else used_cfg_path}")

    cfg = build_sidebar_config(base_cfg)

    st.sidebar.header("Simulation")
    policy = st.sidebar.radio("Policy", POLICIES, horizontal=True)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=999999, value=1234, step=1)
    eval_episodes = st.sidebar.slider("Comparison episodes", 1, 25, 5, 1)
    animate = st.sidebar.checkbox("Animate agents", value=True)
    render_every = st.sidebar.slider("Render every N steps", 1, 25, 3)
    delay = st.sidebar.slider("Frame delay (seconds)", 0.0, 0.5, 0.05, 0.01)

    env_preview = WarehouseEnv(cfg)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Warehouse", f"{env_preview.W} x {env_preview.H}")
    c2.metric("Agents", f"{env_preview.n_agvs} AGV + {env_preview.n_pick} picker")
    c3.metric("Shelves", len(env_preview.rack_cells))
    c4.metric("Goals", len(env_preview.goal_cells))

    tab_run, tab_compare, tab_plots, tab_config = st.tabs(["Run", "Compare policies", "Saved plots", "Config"])

    with tab_run:
        st.markdown('<div class="section-note">The map uses a restrained palette so the simulation is easier to scan while agents move.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="legend-row">
              <span class="legend-chip"><span class="legend-dot" style="background:{THEME["wall"]}"></span>Wall</span>
              <span class="legend-chip"><span class="legend-dot" style="background:{THEME["rack"]}"></span>Rack</span>
              <span class="legend-chip"><span class="legend-dot" style="background:{THEME["rack_active"]}"></span>Active order</span>
              <span class="legend-chip"><span class="legend-dot" style="background:{THEME["goal"]}"></span>Goal</span>
              <span class="legend-chip"><span class="legend-dot" style="background:{THEME["charge"]}"></span>Charger</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run simulation", type="primary", width="stretch"):
            env, info, step_df, agent_df = run_episode(
                cfg,
                Path(selected_model),
                policy,
                int(seed),
                animate,
                int(render_every),
                float(delay),
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Deliveries", f"{info['deliveries']} / {info['total_orders']}")
            m2.metric("Total reward", f"{info['total_reward']:+.2f}")
            m3.metric("Steps", info["steps"])
            m4.metric("Pending", info["pending_orders"])

            fig = plot_episode(step_df, agent_df)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            st.subheader("Agent state")
            last_agents = agent_df.sort_values("step").groupby("agent").tail(1)
            st.dataframe(last_agents, width="stretch", hide_index=True)

            with st.expander("Raw step history"):
                st.dataframe(step_df, width="stretch", hide_index=True)
        else:
            obs, info = env_preview.reset(seed=int(seed))
            fig = draw_environment(env_preview, info, "Initial warehouse state")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    with tab_compare:
        st.markdown(
            '<div class="section-note">Runs DQN, greedy, and random policies on the same deterministic seeds so their performance can be compared fairly.</div>',
            unsafe_allow_html=True,
        )
        setup_cols = st.columns(4)
        setup_cols[0].metric("Policies", len(POLICIES))
        setup_cols[1].metric("Episodes / policy", int(eval_episodes))
        setup_cols[2].metric("Base seed", int(seed))
        setup_cols[3].metric("Total evaluations", len(POLICIES) * int(eval_episodes))

        if st.button("Evaluate DQN vs baselines", width="stretch"):
            with st.spinner("Running deterministic policy comparison..."):
                raw_df, summary_df = evaluate_policy_suite(cfg, Path(selected_model), int(eval_episodes), int(seed))
                st.session_state["policy_raw_df"] = raw_df
                st.session_state["policy_summary_df"] = summary_df

        raw_df = st.session_state.get("policy_raw_df")
        summary_df = st.session_state.get("policy_summary_df")

        if summary_df is not None and raw_df is not None:
            best = summary_df.sort_values(["rank", "policy"]).iloc[0]
            st.markdown(
                f"""
                <div class="callout">
                  <strong>Best policy:</strong> {best["policy"]} with average reward
                  <strong>{best["avg_reward"]:+.2f}</strong>, average deliveries
                  <strong>{best["avg_deliveries"]:.2f}</strong>, and completion rate
                  <strong>{best["avg_completion_rate"]:.1f}%</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_policy_cards(summary_df)
            st.subheader("Comparison Charts")
            fig = plot_policy_comparison(summary_df)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            st.subheader("Evaluation Leaderboard")
            render_leaderboard(summary_df)

            if not raw_df[raw_df["policy"] == "DQN"]["dqn_loaded"].all():
                st.warning("Some DQN checkpoints could not be loaded for the selected parameter shape. Those runs used the greedy fallback.")

            csv = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download summary CSV",
                data=csv,
                file_name="policy_comparison_summary.csv",
                mime="text/csv",
                width="stretch",
            )

            with st.expander("Per-episode results"):
                st.dataframe(raw_df.drop(columns=["load_errors"]).round(3), width="stretch", hide_index=True)
        else:
            st.info("Choose comparison episodes in the sidebar, then run the DQN vs baseline evaluation.")

    with tab_plots:
        st.subheader("Training metrics")
        show_saved_metrics(Path(selected_model))
        st.subheader("Saved plot artifacts")
        show_saved_plots(Path(selected_model))

    with tab_config:
        st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")


if __name__ == "__main__":
    main()
