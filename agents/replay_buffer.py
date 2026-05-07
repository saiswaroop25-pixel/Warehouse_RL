import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.buf    = deque(maxlen=capacity)
        self.device = device

    def push(self, s, a, r, s2, done):
        self.buf.append((
            np.array(s,  dtype=np.float32),
            int(a),
            float(r),
            np.array(s2, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(self.device),
            torch.LongTensor(a).to(self.device),
            torch.FloatTensor(r).to(self.device),
            torch.FloatTensor(np.array(s2)).to(self.device),
            torch.FloatTensor(d).to(self.device),
        )

    def __len__(self):
        return len(self.buf)

    def ready(self, batch_size):
        return len(self.buf) >= batch_size
    