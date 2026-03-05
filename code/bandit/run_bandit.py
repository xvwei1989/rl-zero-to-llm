import numpy as np
import matplotlib.pyplot as plt

from bandit_env import BernoulliBandit
from agents import EpsilonGreedy, UCB1, ThompsonSamplingBeta


def run(agent, env, steps=200):
    rewards = []
    for _ in range(steps):
        a = agent.act()
        r = env.pull(a)
        agent.update(a, r)
        rewards.append(r)
    rewards = np.array(rewards)
    return rewards, np.cumsum(rewards)


def main():
    env = BernoulliBandit.random(k=5, seed=42)
    print("真实中奖概率 p:", np.round(env.probs, 3), "最优臂:", int(np.argmax(env.probs)))

    steps = 400
    agents = {
        "eps=0.1": EpsilonGreedy(env.k, eps=0.1, seed=0),
        "ucb1": UCB1(env.k),
        "thompson": ThompsonSamplingBeta(env.k, seed=0),
    }

    plt.figure(figsize=(8, 4))
    for name, ag in agents.items():
        _, cum = run(ag, env, steps=steps)
        plt.plot(cum, label=name)

    plt.title("Bandit: 累计奖励曲线（越高越好）")
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
