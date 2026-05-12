import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """
    Dueling Double DQN network.

    The model estimates:

        Q(s, a) = V(s) + A(s, a) - mean_a(A(s, a))

    V(s) captures how useful the warehouse state is regardless of the next
    action, while A(s, a) captures whether a specific move is better than the
    average move in that state. This is useful in warehouse routing because
    many states have similar action values: for example, when an agent is
    blocked, charging, or close to a narrow corridor.

    Mean-centering the advantage stream keeps Q-values identifiable and avoids
    arbitrary drift between V and A. The shared backbone learns compact state
    features, then separate value and advantage heads specialize for state
    quality and action ranking.
    """

    def __init__(self, state_dim, action_dim, hidden=(256, 128, 64)):
        super().__init__()
        layers, in_d = [], state_dim
        for h in hidden[:-1]:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        self.shared = nn.Sequential(*layers)

        self.value = nn.Sequential(
            nn.Linear(in_d, hidden[-1]), nn.ReLU(),
            nn.Linear(hidden[-1], 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(in_d, hidden[-1]), nn.ReLU(),
            nn.Linear(hidden[-1], action_dim)
        )
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.shared(x)
        V = self.value(f)
        A = self.adv(f)
        return V + (A - A.mean(dim=1, keepdim=True))

    def act(self, x):
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).argmax(1).item()
