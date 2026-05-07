import numpy as np
import json
from pathlib import Path

_TB = False
try:
    from tensorboardX import SummaryWriter
    _TB = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        _TB = True
    except Exception:
        pass


class Logger:
    def __init__(self, log_dir, use_tb=True):
        self.d = Path(log_dir)
        self.d.mkdir(parents=True, exist_ok=True)

        self.rewards     = []
        self.deliveries  = []
        self.completions = []
        self.epsilons    = []
        self.losses      = []
        self.q_vals      = []
        self.agent_labels = []
        self.agent_episode_metrics = {
            'reward': {},
            'deliveries': {},
            'assists': {},
            'distance': {},
            'collisions': {},
            'wait_steps': {},
            'charging_steps': {},
            'charge_visits': {},
        }

        self.writer = None
        if use_tb and _TB:
            tb = self.d / 'tensorboard'
            tb.mkdir(exist_ok=True)
            self.writer = SummaryWriter(str(tb))

    def _ensure_agent_metrics(self, agents):
        if not agents:
            return
        for agent in agents:
            label = agent.get('label', f"Agent{len(self.agent_labels)}")
            if label not in self.agent_labels:
                self.agent_labels.append(label)
            for metric in self.agent_episode_metrics.values():
                metric.setdefault(label, [])

    def _log_agent_episode(self, ep, agents):
        if not agents:
            return

        self._ensure_agent_metrics(agents)
        metric_names = (
            'reward', 'deliveries', 'assists', 'distance', 'collisions',
            'wait_steps', 'charging_steps', 'charge_visits'
        )
        for agent in agents:
            label = agent['label']
            for name in metric_names:
                self.agent_episode_metrics[name][label].append(float(agent.get(name, 0)))

            if self.writer:
                tb_label = label.replace('-', '_')
                self.writer.add_scalar(f'Agent/{tb_label}/Reward',     agent.get('reward', 0),     ep)
                self.writer.add_scalar(f'Agent/{tb_label}/Deliveries', agent.get('deliveries', 0), ep)
                self.writer.add_scalar(f'Agent/{tb_label}/Assists',    agent.get('assists', 0),    ep)
                self.writer.add_scalar(f'Agent/{tb_label}/Distance',   agent.get('distance', 0),   ep)
                self.writer.add_scalar(f'Agent/{tb_label}/Collisions', agent.get('collisions', 0), ep)
                self.writer.add_scalar(f'Agent/{tb_label}/WaitSteps',  agent.get('wait_steps', 0), ep)
                self.writer.add_scalar(f'Agent/{tb_label}/Charging',   agent.get('charging_steps', 0), ep)
                self.writer.add_scalar(f'Agent/{tb_label}/ChargeVisits', agent.get('charge_visits', 0), ep)

    def log_ep(self, ep, m):
        self.rewards.append(float(m.get('reward', 0)))
        self.deliveries.append(int(m.get('deliveries', 0)))
        cr = m.get('deliveries', 0) / max(m.get('orders', 1), 1) * 100
        self.completions.append(float(cr))
        self.epsilons.append(float(m.get('epsilon', 0)))
        self._log_agent_episode(ep, m.get('agent_metrics', []))

        if self.writer:
            self.writer.add_scalar('Ep/Reward',      m.get('reward', 0),     ep)
            self.writer.add_scalar('Ep/Deliveries',  m.get('deliveries', 0), ep)
            self.writer.add_scalar('Ep/Completion%', cr,                     ep)
            self.writer.add_scalar('Ep/Epsilon',     m.get('epsilon', 0),    ep)

        # Auto-save every 10 episodes
        if ep % 10 == 0:
            self._write_json()

    def log_train(self, step, loss, q):
        self.losses.append(float(loss))
        self.q_vals.append(float(q))
        if self.writer:
            self.writer.add_scalar('Train/Loss', loss, step)
            self.writer.add_scalar('Train/Q',    q,    step)

    def _write_json(self):
        data = {
            'episode_rewards':  self.rewards,
            'deliveries':       self.deliveries,
            'completion_rates': self.completions,
            'epsilons':         self.epsilons,
            'losses':           self.losses,
            'q_values':         self.q_vals,
            'agent_labels':     self.agent_labels,
            'agent_episode_metrics': self.agent_episode_metrics,
        }
        with open(self.d / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def summary(self, ep, w=50):
        w = min(w, len(self.rewards))
        if w == 0:
            return
        print(f"\n{'='*56}")
        print(f"  Episode {ep}  |  last {w} episodes")
        print(f"{'='*56}")
        print(f"  Avg Reward     : {np.mean(self.rewards[-w:]):+.2f}")
        print(f"  Avg Deliveries : {np.mean(self.deliveries[-w:]):.2f}")
        print(f"  Avg Completion : {np.mean(self.completions[-w:]):.1f}%")
        print(f"  Epsilon        : {self.epsilons[-1]:.4f}")
        if self.losses:
            print(f"  Avg Loss       : {np.mean(self.losses[-500:]):.4f}")
        if self.agent_labels:
            reward_line = "  Agent Reward    : " + "  ".join(
                f"{label}={self.agent_episode_metrics['reward'][label][-1]:+.1f}"
                for label in self.agent_labels
                if self.agent_episode_metrics['reward'][label]
            )
            print(reward_line)
        print(f"{'='*56}\n")

    def save(self):
        self._write_json()
        print(f"  Metrics -> {(self.d / 'metrics.json').resolve()}")

    def close(self):
        if self.writer:
            self.writer.close()
        self.save()
