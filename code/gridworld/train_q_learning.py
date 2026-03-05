import numpy as np

from gridworld_env import Gridworld


def epsilon_greedy(q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, q.shape[1]))
    return int(np.argmax(q[s]))


def main():
    env = Gridworld(height=5, width=5, start=(0, 0), goal=(4, 4), step_cost=-0.01)
    rng = np.random.default_rng(0)

    q = np.zeros((env.n_states, env.n_actions), dtype=float)

    alpha = 0.2
    gamma = 0.98
    eps_start, eps_end = 0.3, 0.05

    episodes = 500
    max_steps = 200

    returns = []

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * (1 - ep / episodes)
        s = env.reset()
        total = 0.0

        for _ in range(max_steps):
            a = epsilon_greedy(q, s, eps, rng)
            s2, r, done = env.step(a)
            total += r

            target = r + gamma * np.max(q[s2]) * (0.0 if done else 1.0)
            q[s, a] += alpha * (target - q[s, a])

            s = s2
            if done:
                break

        returns.append(total)
        if (ep + 1) % 50 == 0:
            print(f"ep {ep+1:4d} | eps={eps:.3f} | avg_return(last50)={np.mean(returns[-50:]):.3f}")

    # 展示学到的贪心策略（用箭头）
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    print("\nGreedy policy:")
    for r in range(env.h):
        row = []
        for c in range(env.w):
            s = r * env.w + c
            if (r, c) == env.goal:
                row.append("G")
            else:
                row.append(arrows[int(np.argmax(q[s]))])
        print(" ".join(row))


if __name__ == "__main__":
    main()
