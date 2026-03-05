import math
import numpy as np


class EpsilonGreedy:
    def __init__(self, k, eps=0.1, seed=0):
        self.k = k
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(k, dtype=int)
        self.values = np.zeros(k, dtype=float)  # 经验均值

    def act(self):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.k))
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # 增量更新均值
        self.values[arm] += (reward - self.values[arm]) / n


class UCB1:
    def __init__(self, k):
        self.k = k
        self.t = 0
        self.counts = np.zeros(k, dtype=int)
        self.values = np.zeros(k, dtype=float)

    def act(self):
        self.t += 1
        # 先保证每个臂至少拉一次
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
        bonus = np.sqrt(2 * np.log(self.t) / self.counts)
        ucb = self.values + bonus
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSamplingBeta:
    """伯努利奖励下的 Thompson Sampling：Beta(α,β) 共轭先验。"""

    def __init__(self, k, alpha=1.0, beta=1.0, seed=0):
        self.k = k
        self.alpha = np.ones(k) * alpha
        self.beta = np.ones(k) * beta
        self.rng = np.random.default_rng(seed)

    def act(self):
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        # reward ∈ {0,1}
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
