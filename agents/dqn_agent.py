import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .network import DuelingDQN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        ) if device is None else torch.device(device)

        self.action_dim = action_dim
        hidden          = cfg['agent']['hidden_dims']

        self.policy = DuelingDQN(state_dim, action_dim, hidden).to(self.device)
        self.target = DuelingDQN(state_dim, action_dim, hidden).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt     = optim.Adam(
            self.policy.parameters(),
            lr=cfg['agent']['learning_rate']
        )
        self.buf     = ReplayBuffer(
            cfg['agent']['buffer_capacity'], str(self.device)
        )
        self.gamma   = cfg['agent']['gamma']
        self.bs      = cfg['agent']['batch_size']
        self.tgt_upd = cfg['agent']['target_update_freq']
        self.upd_frq = cfg['agent']['update_freq']

        self.eps     = cfg['agent']['epsilon_start']
        self.eps_end = cfg['agent']['epsilon_end']
        self.eps_dec = cfg['agent']['epsilon_decay']

        self.upd_ctr = 0
        self.ep_ctr  = 0

    def act(self, state, train=True):
        if train and np.random.random() < self.eps:
            return np.random.randint(self.action_dim)
        t = torch.FloatTensor(state).to(self.device)
        return self.policy.act(t)

    def store(self, s, a, r, s2, done):
        self.buf.push(s, a, r, s2, done)

    def learn(self):
        if not self.buf.ready(self.bs):
            return {}

        s, a, r, s2, d = self.buf.sample(self.bs)

        # Double DQN target
        with torch.no_grad():
            best_a = self.policy(s2).argmax(1, keepdim=True)
            tgt_q  = r + (1 - d) * self.gamma * \
                     self.target(s2).gather(1, best_a).squeeze(1)

        cur_q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss  = nn.SmoothL1Loss()(cur_q, tgt_q)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        self.upd_ctr += 1
        if self.upd_ctr % self.tgt_upd == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return {'loss': loss.item(), 'q': cur_q.mean().item()}

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_dec)
        self.ep_ctr += 1

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'target': self.target.state_dict(),
            'opt':    self.opt.state_dict(),
            'eps':    self.eps,
            'ep':     self.ep_ctr,
            'upd':    self.upd_ctr,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        self.target.load_state_dict(ck['target'])
        self.opt.load_state_dict(ck['opt'])
        self.eps     = ck.get('eps', self.eps_end)
        self.ep_ctr  = ck.get('ep',  0)
        self.upd_ctr = ck.get('upd', 0)
        print(f"  Loaded {path}  (ep={self.ep_ctr}, eps={self.eps:.3f})")