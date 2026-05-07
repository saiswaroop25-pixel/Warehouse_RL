#!/usr/bin/env python3
"""TA-RWARE Pro v2 - Training Script"""
import yaml, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from envs.warehouse_env import WarehouseEnv
from agents.dqn_agent   import DQNAgent
from utils.logger        import Logger
from utils.experiment    import prepare_run_dirs, save_run_snapshot
from utils.visualization import plot_metrics


def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_agents(env, cfg):
    agents = []
    for a in env.agents:
        sd  = env.state_dim
        act = cfg['agent']['action_dim']
        agents.append(DQNAgent(sd, act, cfg))
    return agents


def eval_run(env, dqns, n=3):
    totals = []
    for _ in range(n):
        obs, _ = env.reset()
        ep_r, done = 0.0, False
        while not done:
            acts = [dqns[i].act(obs[i], train=False) for i in range(env.n_agents)]
            obs, rews, term, trunc, _ = env.step(acts)
            ep_r += sum(rews); done = term or trunc
        totals.append(ep_r)
    return float(np.mean(totals))


def build_agent_episode_metrics(env, agent_rewards, info):
    metrics = []
    for i, am in enumerate(info['agent_metrics']):
        atype = 'AGV' if i < env.n_agvs else 'Picker'
        metrics.append({
            'label':      f"Agent{i}-{atype}",
            'reward':     float(agent_rewards[i]),
            'deliveries': int(am.get('deliveries', 0)),
            'assists':    int(am.get('assists', 0)),
            'distance':   int(am.get('distance', 0)),
            'collisions': int(am.get('collisions', 0)),
            'wait_steps': int(am.get('wait_steps', 0)),
            'charging_steps': int(am.get('charging_steps', 0)),
            'charge_visits': int(am.get('charge_visits', 0)),
        })
    return metrics


def train(cfg_path, device=None, resume=False, run_name=None):
    cfg = load_cfg(cfg_path)
    run_name, log_root, model_root, log_dir, model_dir = prepare_run_dirs(
        cfg, resume=resume, run_name=run_name
    )
    cfg['logging']['run_name'] = run_name
    cfg['logging']['log_root'] = str(log_root)
    cfg['logging']['model_root'] = str(model_root)
    cfg['logging']['log_dir'] = str(log_dir)
    cfg['logging']['model_dir'] = str(model_dir)
    save_run_snapshot(cfg, run_name, log_dir, model_dir)

    env = WarehouseEnv(cfg)

    n_ep       = cfg['training']['num_episodes']
    save_freq  = cfg['training']['save_freq']
    eval_freq  = cfg['training']['eval_freq']
    eval_eps   = cfg['training']['eval_episodes']

    dqns     = make_agents(env, cfg)
    logger   = Logger(str(log_dir), use_tb=cfg['logging']['tensorboard'])
    best_val = -float('inf')
    start_ep = 1
    step_ctr = 0

    print("=" * 58)
    print(f"  TA-RWARE Pro v2  | AGVs={env.n_agvs} Pickers={env.n_pick}")
    print(f"  Run Name: {run_name}")
    print(f"  Grid={env.W}x{env.H}  Racks={len(env.rack_cells)}  Goals={len(env.goal_cells)}")
    print(f"  Episodes={n_ep}  MaxSteps={env.max_steps}")
    print(f"  Device: {dqns[0].device}")
    print("=" * 58)

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume:
        loaded_ep = 0
        for i, dqn in enumerate(dqns):
            cands = ([model_dir/f"agent{i}_best.pt"]
                    + sorted(model_dir.glob(f"agent{i}_ep*.pt"), reverse=True))
            for cp in cands:
                if cp.exists():
                    dqn.load(str(cp))
                    if '_ep' in cp.stem:
                        loaded_ep = max(loaded_ep, int(cp.stem.split('_ep')[-1]))
                    break
        if loaded_ep:
            start_ep = loaded_ep + 1
            step_ctr = loaded_ep * env.max_steps
            print(f"\nResumed from ep {loaded_ep} -> continuing from ep {start_ep}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in tqdm(range(start_ep, n_ep+1), desc="Training"):
        obs, info = env.reset()
        ep_r  = 0.0
        done  = False
        ep_st = 0
        agent_ep_rewards = [0.0] * env.n_agents

        while not done:
            acts = [dqns[i].act(obs[i], train=True) for i in range(env.n_agents)]
            nobs, rews, term, trunc, info = env.step(acts)
            done = term or trunc

            for i in range(env.n_agents):
                dqns[i].store(obs[i], acts[i], rews[i], nobs[i], float(term))
                if ep_st % dqns[i].upd_frq == 0:
                    m = dqns[i].learn()
                    if m:
                        logger.log_train(step_ctr, m['loss'], m['q'])

            obs    = nobs
            ep_r  += sum(rews)
            for i, r in enumerate(rews):
                agent_ep_rewards[i] += float(r)
            ep_st += 1
            step_ctr += 1

        for dqn in dqns:
            dqn.decay_eps()

        agent_metrics = build_agent_episode_metrics(env, agent_ep_rewards, info)
        logger.log_ep(ep, {
            'reward':     ep_r,
            'deliveries': info['deliveries'],
            'orders':     info['total_orders'],
            'epsilon':    dqns[0].eps,
            'agent_metrics': agent_metrics,
        })

        # Print summary every 50 episodes
        if ep % 50 == 0:
            logger.summary(ep, w=50)

        # Save checkpoint
        if ep % save_freq == 0:
            for i, dqn in enumerate(dqns):
                dqn.save(str(model_dir/f"agent{i}_ep{ep}.pt"))
            logger.save()   # ← also save metrics to JSON every checkpoint

        # Evaluation
        if ep % eval_freq == 0:
            val = eval_run(env, dqns, eval_eps)
            print(f"  [Eval ep={ep:4d}] avg_reward={val:+.2f}  "
                  f"eps={dqns[0].eps:.3f}  "
                  f"deliveries={info['deliveries']}/{info['total_orders']}")
            if val > best_val:
                best_val = val
                for i, dqn in enumerate(dqns):
                    dqn.save(str(model_dir/f"agent{i}_best.pt"))
                print(f"  ** New best: {best_val:+.2f} - saved best models **")

    # Final save
    for i, dqn in enumerate(dqns):
        dqn.save(str(model_dir/f"agent{i}_final.pt"))
    logger.close()
    plot_metrics({
        'episode_rewards':  logger.rewards,
        'deliveries':       logger.deliveries,
        'completion_rates': logger.completions,
        'losses':           logger.losses,
        'q_values':         logger.q_vals,
        'epsilons':         logger.epsilons,
        'agent_labels':     logger.agent_labels,
        'agent_episode_metrics': logger.agent_episode_metrics,
    }, str(log_dir/'training_metrics.png'))
    print(f"\nTraining done. Best eval reward: {best_val:+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--device', default=None)
    p.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    p.add_argument('--run_name', default=None, help='Optional run name for this experiment')
    args = p.parse_args()
    train(args.config, args.device, args.resume, args.run_name)


if __name__ == '__main__':
    main()
