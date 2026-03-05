import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os, sys

# 允许直接运行本文件（把项目根目录加入 sys.path）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from code.gridworld.gridworld_env import Gridworld


def one_hot(state, n_states):
    x = torch.zeros(n_states, dtype=torch.float32)
    x[state] = 1.0
    return x


class QNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    s: int
    a: int
    r: float
    s2: int
    done: bool


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buf = []
        self.i = 0

    def push(self, t: Transition):
        if len(self.buf) < self.capacity:
            self.buf.append(t)
        else:
            self.buf[self.i] = t
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size=64):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    env = Gridworld(height=6, width=6, start=(0, 0), goal=(5, 5), step_cost=-0.01)
    n_states, n_actions = env.n_states, env.n_actions

    q = QNet(n_states, n_actions)
    q_tgt = QNet(n_states, n_actions)
    q_tgt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=1e-3)
    rb = ReplayBuffer(capacity=5000)

    gamma = 0.98
    eps_start, eps_end = 0.3, 0.05
    episodes = 1200
    max_steps = 200

    batch_size = 64
    target_update_every = 50

    returns = []

    def act_eps_greedy(s, eps):
        if random.random() < eps:
            return random.randrange(n_actions)
        with torch.no_grad():
            qs = q(one_hot(s, n_states))
            return int(torch.argmax(qs).item())

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * (1 - ep / episodes)
        s = env.reset()
        total = 0.0

        for _ in range(max_steps):
            a = act_eps_greedy(s, eps)
            s2, r, done = env.step(a)
            total += r

            rb.push(Transition(s, a, r, s2, done))
            s = s2

            if len(rb) >= batch_size:
                batch = rb.sample(batch_size)
                s_b = torch.stack([one_hot(t.s, n_states) for t in batch])
                a_b = torch.tensor([t.a for t in batch], dtype=torch.int64)
                r_b = torch.tensor([t.r for t in batch], dtype=torch.float32)
                s2_b = torch.stack([one_hot(t.s2, n_states) for t in batch])
                d_b = torch.tensor([t.done for t in batch], dtype=torch.float32)

                q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    max_q_s2 = q_tgt(s2_b).max(dim=1).values
                    target = r_b + gamma * (1.0 - d_b) * max_q_s2

                loss = nn.functional.smooth_l1_loss(q_sa, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if done:
                break

        returns.append(total)

        if (ep + 1) % 50 == 0:
            print(f"ep {ep+1:4d} | eps={eps:.3f} | avg_return(last50)={np.mean(returns[-50:]):.3f}")

        if (ep + 1) % target_update_every == 0:
            q_tgt.load_state_dict(q.state_dict())


if __name__ == "__main__":
    main()
