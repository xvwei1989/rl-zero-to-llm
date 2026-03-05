import os
import pandas as pd
import numpy as np


def main():
    path = os.path.join(os.path.dirname(__file__), "../..", "data", "offline_grid.csv")
    df = pd.read_csv(path)

    # 这里演示最小概念：
    # - 行为策略 b(a|s) 近似：epsilon-greedy（我们假设 eps=0.2，且贪心动作概率=0.8）
    # - 目标策略 π(a|s) 近似：总是选 a=3（向右）作为一个玩具示例
    # 注意：真实项目中 b/π 往往由模型给出概率。

    eps = 0.2

    def greedy_action(s):
        # 和数据生成时的启发式一致，用来估 b(a|s)
        w = 6
        r, c = divmod(int(s), w)
        gr, gc = 5, 5
        if abs(gr - r) > abs(gc - c):
            return 1 if gr > r else 0
        else:
            return 3 if gc > c else 2

    def b_prob(s, a):
        ga = greedy_action(s)
        if a == ga:
            return 1 - eps + eps / 4
        return eps / 4

    def pi_prob(s, a):
        # 目标策略：永远向右（a=3）
        return 1.0 if int(a) == 3 else 0.0

    # 重要性采样：w = Π_t π(a_t|s_t) / b(a_t|s_t)
    # 为简单，我们按“每一步独立”做 per-step 的比值，且只估计一步期望回报（演示概念）
    ratios = []
    for s, a in zip(df["s"], df["a"]):
        denom = b_prob(s, int(a))
        num = pi_prob(s, int(a))
        ratios.append(0.0 if denom == 0 else num / denom)
    ratios = np.array(ratios)

    is_est = np.mean(ratios * df["r"].values)
    print("One-step IS estimate (toy):", float(is_est))

    # Doubly Robust（玩具版）：
    # 需要一个 Q_hat(s,a) 近似，这里用“直接用样本奖励作为 Q_hat”仅做演示。
    q_hat = df["r"].values
    v_hat = []
    for s in df["s"].values:
        # V_hat(s) = Σ_a π(a|s) Q_hat(s,a)
        # 我们只有样本上的 q_hat，所以用 0 近似，演示结构即可
        v_hat.append(0.0)
    v_hat = np.array(v_hat)

    dr_est = np.mean(v_hat + ratios * (df["r"].values - q_hat))
    print("One-step DR estimate (toy):", float(dr_est))


if __name__ == "__main__":
    main()
