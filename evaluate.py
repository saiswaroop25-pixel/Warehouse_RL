#!/usr/bin/env python3
"""TA-RWARE Pro v2 - Evaluation and baseline comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from envs.warehouse_env import WarehouseEnv
from agents.baseline_policies import greedy_actions, random_actions
from agents.dqn_agent import DQNAgent
from utils.experiment import resolve_run_dir
from utils.visualization import Renderer

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL = True
except Exception:
    _MPL = False


POLICY_CHOICES = ("dqn", "greedy", "random")


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_eval_config(model_dir, cfg_path):
    model_dir = Path(model_dir)
    snapshot_candidates = [
        model_dir / "config_snapshot.yaml",
        Path("logs") / "runs" / model_dir.name / "config_snapshot.yaml",
    ]
    for candidate in snapshot_candidates:
        if candidate.exists():
            if str(candidate) != str(Path(cfg_path)):
                print(f"  Using saved run config: {candidate}")
            return load_cfg(candidate), candidate
    return load_cfg(cfg_path), Path(cfg_path)


def load_dqns(env, cfg, model_dir, device=None):
    dqns = []
    mdir = Path(model_dir)
    for a in env.agents:
        dqn = DQNAgent(env.state_dim, cfg["agent"]["action_dim"], cfg, device=device)
        dqn.eps = 0.0
        cands = ([mdir / f"agent{a.id}_best.pt",
                  mdir / f"agent{a.id}_final.pt"]
                + sorted(mdir.glob(f"agent{a.id}_ep*.pt"), reverse=True))
        for cp in cands:
            if cp.exists():
                dqn.load(str(cp))
                break
        else:
            print(f"  WARNING: no checkpoint for agent {a.id}")
        dqns.append(dqn)
    return dqns


def policy_actions(policy_name, env, dqns=None, rng=None):
    if policy_name == "dqn":
        return [dqns[i].act(env._observe(i), train=False) for i in range(env.n_agents)]
    if policy_name == "greedy":
        return greedy_actions(env)
    if policy_name == "random":
        return random_actions(env, rng=rng)
    raise ValueError(f"Unsupported policy: {policy_name}")


def run_episode(env, policy_name, dqns=None, seed=None, renderer=None, ep_label=0, rng=None):
    obs, info = env.reset(seed=seed)
    ep_reward, done, steps = 0.0, False, 0
    agent_rewards = [0.0] * env.n_agents
    running = True

    while not done and running:
        if policy_name == "dqn":
            acts = [dqns[i].act(obs[i], train=False) for i in range(env.n_agents)]
        else:
            acts = policy_actions(policy_name, env, dqns=dqns, rng=rng)
        obs, rews, term, trunc, info = env.step(acts)
        done = term or trunc
        ep_reward += sum(rews)
        steps += 1
        for i, r in enumerate(rews):
            agent_rewards[i] += float(r)
        if renderer and not renderer.render(env, info, ep_label):
            running = False

    if not running:
        return None

    completion = info["deliveries"] / max(info["total_orders"], 1) * 100
    per_agent = []
    for i, am in enumerate(info["agent_metrics"]):
        label = f"Agent{i}-{'AGV' if i < env.n_agvs else 'Picker'}"
        idle_ratio = (am.get("wait_steps", 0) + am.get("charging_steps", 0)) / max(steps, 1)
        per_agent.append({
            "label": label,
            "reward": agent_rewards[i],
            "deliveries": am.get("deliveries", 0),
            "assists": am.get("assists", 0),
            "distance": am.get("distance", 0),
            "collisions": am.get("collisions", 0),
            "wait_steps": am.get("wait_steps", 0),
            "charging_steps": am.get("charging_steps", 0),
            "charge_visits": am.get("charge_visits", 0),
            "empty_events": am.get("empty_events", 0),
            "idle_ratio": idle_ratio,
        })

    cycle_times = info.get("completed_order_times", [])
    total_charge_visits = sum(am.get("charge_visits", 0) for am in info["agent_metrics"])
    total_collisions = sum(am.get("collisions", 0) for am in info["agent_metrics"])
    total_empty_events = sum(am.get("empty_events", 0) for am in info["agent_metrics"])
    avg_idle_ratio = float(np.mean([a["idle_ratio"] for a in per_agent])) if per_agent else 0.0

    return {
        "reward": ep_reward,
        "deliveries": info["deliveries"],
        "total_orders": info["total_orders"],
        "completion_rate": completion,
        "steps": steps,
        "throughput_per_100_steps": info["deliveries"] / max(steps, 1) * 100.0,
        "avg_order_cycle_time": float(np.mean(cycle_times)) if cycle_times else 0.0,
        "avg_idle_ratio": avg_idle_ratio,
        "charge_visits": total_charge_visits,
        "total_collisions": total_collisions,
        "empty_events": total_empty_events,
        "per_agent": per_agent,
    }


def summarize_policy(episodes):
    rewards = [ep["reward"] for ep in episodes]
    deliveries = [ep["deliveries"] for ep in episodes]
    completion = [ep["completion_rate"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]
    throughput = [ep["throughput_per_100_steps"] for ep in episodes]
    cycle = [ep["avg_order_cycle_time"] for ep in episodes]
    idle = [ep["avg_idle_ratio"] for ep in episodes]
    collisions = [ep["total_collisions"] for ep in episodes]
    charge_visits = [ep["charge_visits"] for ep in episodes]
    empty_events = [ep["empty_events"] for ep in episodes]

    agent_rollup = {}
    for ep in episodes:
        for agent in ep["per_agent"]:
            store = agent_rollup.setdefault(agent["label"], {
                "reward": [], "deliveries": [], "assists": [], "distance": [],
                "collisions": [], "wait_steps": [], "charging_steps": [],
                "charge_visits": [], "idle_ratio": [],
            })
            for key in store:
                store[key].append(agent[key])

    return {
        "episodes": len(episodes),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "best_reward": float(np.max(rewards)) if rewards else 0.0,
        "avg_deliveries": float(np.mean(deliveries)) if deliveries else 0.0,
        "avg_completion_rate": float(np.mean(completion)) if completion else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "avg_throughput_per_100_steps": float(np.mean(throughput)) if throughput else 0.0,
        "avg_order_cycle_time": float(np.mean(cycle)) if cycle else 0.0,
        "avg_idle_ratio": float(np.mean(idle)) if idle else 0.0,
        "avg_collisions": float(np.mean(collisions)) if collisions else 0.0,
        "avg_charge_visits": float(np.mean(charge_visits)) if charge_visits else 0.0,
        "avg_empty_events": float(np.mean(empty_events)) if empty_events else 0.0,
        "per_agent": {
            label: {f"avg_{k}": float(np.mean(v)) if v else 0.0 for k, v in data.items()}
            for label, data in agent_rollup.items()
        }
    }


def print_summary(name, summary):
    print(f"\n{'='*64}")
    print(f"  POLICY: {name.upper()}")
    print(f"{'='*64}")
    print(f"  Avg Reward             : {summary['avg_reward']:+.2f} +/- {summary['reward_std']:.2f}")
    print(f"  Avg Deliveries         : {summary['avg_deliveries']:.2f}")
    print(f"  Avg Completion         : {summary['avg_completion_rate']:.1f}%")
    print(f"  Avg Steps              : {summary['avg_steps']:.1f}")
    print(f"  Throughput / 100 steps : {summary['avg_throughput_per_100_steps']:.2f}")
    print(f"  Avg Order Cycle Time   : {summary['avg_order_cycle_time']:.2f}")
    print(f"  Avg Idle Ratio         : {summary['avg_idle_ratio']:.2f}")
    print(f"  Avg Collisions         : {summary['avg_collisions']:.2f}")
    print(f"  Avg Charge Visits      : {summary['avg_charge_visits']:.2f}")
    print(f"  Avg Empty Events       : {summary['avg_empty_events']:.2f}")
    print("  Per-Agent Averages")
    for label, vals in summary["per_agent"].items():
        print(f"  {label:<14} "
              f"reward={vals['avg_reward']:+.2f}  "
              f"del={vals['avg_deliveries']:.2f}  "
              f"assist={vals['avg_assists']:.2f}  "
              f"dist={vals['avg_distance']:.1f}  "
              f"idle={vals['avg_idle_ratio']:.2f}  "
              f"coll={vals['avg_collisions']:.2f}")


def save_comparison_artifacts(summaries, out_dir, tag):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{tag}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Saved -> {json_path}")

    if not _MPL:
        return

    metrics = [
        ("avg_reward", "Avg Reward"),
        ("avg_deliveries", "Avg Deliveries"),
        ("avg_completion_rate", "Completion %"),
        ("avg_throughput_per_100_steps", "Throughput / 100"),
        ("avg_order_cycle_time", "Order Cycle Time"),
        ("avg_collisions", "Collisions"),
    ]
    labels = list(summaries.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.patch.set_facecolor("#0f0f1a")

    for ax, (key, title) in zip(axes.flat, metrics):
        values = [summaries[label][key] for label in labels]
        ax.bar(labels, values, color=["#4488ff", "#44dd88", "#ffaa33"][:len(labels)])
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="#ccccee", fontsize=11)
        ax.tick_params(colors="#aaaacc", axis="x", rotation=20)
        ax.tick_params(colors="#aaaacc", axis="y")
        ax.grid(True, axis="y", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_color("#333355")

    plt.tight_layout(pad=2)
    png_path = out_dir / f"{tag}_comparison.png"
    plt.savefig(png_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {png_path}")


def evaluate_policy(model_dir, cfg_path, num_episodes=10, render=True, device=None, policy="dqn", seeds=None):
    cfg, used_cfg_path = resolve_eval_config(model_dir, cfg_path)
    env = WarehouseEnv(cfg)
    dqns = load_dqns(env, cfg, model_dir, device=device) if policy == "dqn" else None
    renderer = Renderer(env) if render else None
    rng = np.random.default_rng(12345)

    episodes = []
    seeds = seeds or [1000 + ep for ep in range(num_episodes)]
    print(f"\nEvaluating {policy.upper()} over {num_episodes} episodes")
    print(f"  Config: {used_cfg_path}")
    for ep, seed in enumerate(seeds, start=1):
        result = run_episode(env, policy, dqns=dqns, seed=seed, renderer=renderer, ep_label=ep, rng=rng)
        if result is None:
            break
        episodes.append(result)
        print(f"  Episode {ep:>2}: reward={result['reward']:+.2f}  "
              f"del={result['deliveries']}/{result['total_orders']}  "
              f"steps={result['steps']}")

    if renderer:
        try:
            import pygame as pg
            print("\nClose the window to exit.")
            while True:
                for ev in pg.event.get():
                    if ev.type == pg.QUIT:
                        renderer.close()
                        break
                else:
                    renderer.clk.tick(10)
                    continue
                break
        except Exception:
            pass

    summary = summarize_policy(episodes)
    print_summary(policy, summary)
    return summary


def compare_policies(model_dir, cfg_path, num_episodes=10, device=None, policies=None):
    model_dir = resolve_run_dir(model_dir)
    cfg, _ = resolve_eval_config(model_dir, cfg_path)
    log_dir = resolve_run_dir(cfg["logging"]["log_dir"], run_name=model_dir.name)
    eval_dir = Path(log_dir) / "evaluations"
    seeds = [1000 + ep for ep in range(num_episodes)]

    policies = policies or list(POLICY_CHOICES)
    summaries = {}
    for policy in policies:
        summaries[policy] = evaluate_policy(
            model_dir=model_dir,
            cfg_path=cfg_path,
            num_episodes=num_episodes,
            render=False,
            device=device,
            policy=policy,
            seeds=seeds,
        )

    save_comparison_artifacts(summaries, eval_dir, f"policy_compare_{num_episodes}ep")
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--policy", choices=POLICY_CHOICES, default="dqn")
    p.add_argument("--compare", action="store_true", help="Compare DQN against built-in baselines")
    p.add_argument("--policies", default="dqn,greedy,random",
                   help="Comma-separated policies to compare when --compare is used")
    args = p.parse_args()

    model_dir = resolve_run_dir(args.model_dir)
    if args.compare:
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
        compare_policies(model_dir, args.config, args.episodes, device=args.device, policies=policies)
    else:
        evaluate_policy(model_dir, args.config, args.episodes, args.render, args.device, args.policy)


if __name__ == "__main__":
    main()
