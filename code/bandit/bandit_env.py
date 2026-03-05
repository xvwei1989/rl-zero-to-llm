import numpy as np


class BernoulliBandit:
    """K-臂伯努利老虎机：每个臂 i 有一个未知中奖概率 p_i，拉一次得到 r∈{0,1}。"""

    def __init__(self, probs, seed=0):
        self.probs = np.array(probs, dtype=float)
        self.k = len(self.probs)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def random(k=5, seed=0):
        rng = np.random.default_rng(seed)
        probs = rng.uniform(0.05, 0.95, size=k)
        return BernoulliBandit(probs, seed=seed + 1)

    def pull(self, arm: int) -> int:
        p = self.probs[arm]
        return int(self.rng.random() < p)
