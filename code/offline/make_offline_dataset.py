import os, sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from code.gridworld.gridworld_env import Gridworld


def main():
    env = Gridworld(height=6, width=6, start=(0, 0), goal=(5, 5), step_cost=-0.01)
    rng = np.random.default_rng(0)

    n = 20000
    eps = 0.2  # 行为策略：epsilon-greedy（这里用随机近似）

    rows = []
    s = env.reset()
    for _ in range(n):
        if rng.random() < eps:
            a = int(rng.integers(0, env.n_actions))
        else:
            # 这里为了简单：用“朝向目标”的启发式作为近似贪心
            r, c = divmod(s, env.w)
            gr, gc = env.goal
            if abs(gr - r) > abs(gc - c):
                a = 1 if gr > r else 0
            else:
                a = 3 if gc > c else 2

        s2, rwd, done = env.step(a)
        rows.append({"s": s, "a": a, "r": rwd, "s2": s2, "done": int(done)})
        s = env.reset() if done else s2

    out = os.path.join(ROOT, "data", "offline_grid.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
