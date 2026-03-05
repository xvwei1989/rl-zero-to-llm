import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from code.gridworld.gridworld_env import Gridworld


def one_hot(state, n_states):
    x = torch.zeros(n_states, dtype=torch.float32)
    x[state] = 1.0
    return x


class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_states, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    env = Gridworld(height=6, width=6, start=(0, 0), goal=(5, 5), step_cost=-0.01)
    n_states, n_actions = env.n_states, env.n_actions

    ac = ActorCritic(n_states, n_actions)
    opt = optim.Adam(ac.parameters(), lr=3e-4)

    gamma = 0.98
    clip_eps = 0.2
    ent_coef = 0.01
    vf_coef = 0.5

    steps_per_iter = 1024
    iters = 80
    minibatch = 256
    epochs = 4

    def collect_rollout():
        s = env.reset()
        S, A, R, D, LOGP, V = [], [], [], [], [], []

        for _ in range(steps_per_iter):
            x = one_hot(s, n_states)
            with torch.no_grad():
                logits, value = ac(x)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)

            s2, r, done = env.step(int(a.item()))

            S.append(s)
            A.append(int(a.item()))
            R.append(float(r))
            D.append(float(done))
            LOGP.append(float(logp.item()))
            V.append(float(value.item()))

            s = s2
            if done:
                s = env.reset()

        # bootstrap value for last state
        with torch.no_grad():
            _, v_last = ac(one_hot(s, n_states))
        v_last = float(v_last.item())

        # 计算回报与优势（简化版：GAE(λ) 取 λ=1 => Monte Carlo with bootstrap）
        G = []
        g = v_last
        for r, d in zip(reversed(R), reversed(D)):
            g = r + gamma * g * (1.0 - d)
            G.append(g)
        G = list(reversed(G))

        adv = np.array(G) - np.array(V)
        # 标准化优势，训练更稳
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch = {
            "s": torch.tensor(S, dtype=torch.int64),
            "a": torch.tensor(A, dtype=torch.int64),
            "logp_old": torch.tensor(LOGP, dtype=torch.float32),
            "ret": torch.tensor(G, dtype=torch.float32),
            "adv": torch.tensor(adv, dtype=torch.float32),
        }
        return batch

    for it in range(1, iters + 1):
        batch = collect_rollout()

        # 训练多轮
        idx = np.arange(steps_per_iter)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, steps_per_iter, minibatch):
                mb = idx[start : start + minibatch]

                s = batch["s"][mb]
                a = batch["a"][mb]
                logp_old = batch["logp_old"][mb]
                ret = batch["ret"][mb]
                adv = batch["adv"][mb]

                x = torch.stack([one_hot(int(si.item()), n_states) for si in s])
                logits, v = ac(x)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = (ret - v).pow(2).mean()

                loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                opt.step()

        if it % 5 == 0:
            # 粗略监控：平均回报（rollout 回报均值）
            print(f"iter {it:3d} | mean_return={batch['ret'].mean().item():.3f} | pi_loss={pi_loss.item():.3f} | v_loss={v_loss.item():.3f}")


if __name__ == "__main__":
    main()
