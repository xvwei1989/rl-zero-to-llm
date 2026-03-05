"""Generate simple, reproducible figures for the tutorial.

Run:
  python3 code/utils/make_figures.py

Outputs to ./figures
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_rl_loop():
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    ax.text(0.02, 0.75, "Agent", fontsize=12, weight="bold")
    ax.text(0.82, 0.75, "Environment", fontsize=12, weight="bold")

    # arrows
    ax.annotate("action a", xy=(0.78, 0.65), xytext=(0.20, 0.65),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("state s, reward r", xy=(0.22, 0.35), xytext=(0.78, 0.35),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.46, 0.08, "loop: observe → act → get feedback → learn", ha="center", fontsize=10)
    save(fig, "00_rl_loop.png")


def fig_bellman():
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    ax.axis("off")
    ax.text(0.04, 0.7, "V(s)", fontsize=14, weight="bold")
    ax.annotate("=", xy=(0.16, 0.7), xytext=(0.16, 0.7), fontsize=16)
    ax.text(0.22, 0.7, "expected[ r + γ · V(s') ]", fontsize=13)
    ax.text(0.04, 0.32, "Bellman intuition:", fontsize=11, weight="bold")
    ax.text(0.26, 0.32, "now value = now reward + discounted future value", fontsize=11)
    save(fig, "02_bellman_intuition.png")


def fig_td_error():
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.axis("off")
    ax.text(0.03, 0.75, "TD error δ", fontsize=14, weight="bold")
    ax.text(0.20, 0.75, "= target − current", fontsize=13)
    ax.text(0.03, 0.40, "target = r + γ · max_a' Q(s',a')", fontsize=12)
    ax.text(0.03, 0.15, "update: Q(s,a) ← Q(s,a) + α · δ", fontsize=12)
    save(fig, "03_td_error.png")


def fig_dqn_two_nets():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axis("off")
    ax.text(0.05, 0.83, "Online Qθ", fontsize=12, weight="bold")
    ax.text(0.68, 0.83, "Target Qθ−", fontsize=12, weight="bold")

    ax.annotate("sample batch", xy=(0.52, 0.65), xytext=(0.08, 0.65),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.10, 0.57, "Replay Buffer", fontsize=10)

    ax.annotate("compute target", xy=(0.88, 0.45), xytext=(0.52, 0.45),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.54, 0.37, "r + γ max Qθ−(s',·)", fontsize=10)

    ax.annotate("periodic copy", xy=(0.70, 0.20), xytext=(0.30, 0.20),
                arrowprops=dict(arrowstyle="->", lw=2, linestyle="--"))
    ax.text(0.33, 0.12, "θ− ← θ", fontsize=10)

    save(fig, "04_dqn_replay_target.png")


def fig_ppo_clip():
    # visualize clipped objective shape (toy)
    eps = 0.2
    r = np.linspace(0.0, 2.0, 400)
    adv = 1.0
    unclipped = r * adv
    clipped = np.clip(r, 1 - eps, 1 + eps) * adv
    obj = np.minimum(unclipped, clipped)

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(r, unclipped, label="unclipped: r·A")
    ax.plot(r, clipped, label="clipped")
    ax.plot(r, obj, label="min(unclipped, clipped)", lw=3)
    ax.axvline(1 - eps, color="gray", ls="--")
    ax.axvline(1 + eps, color="gray", ls="--")
    ax.set_xlabel("ratio r = πθ(a|s) / πold(a|s)")
    ax.set_ylabel("objective (A>0 case)")
    ax.set_title("PPO clip intuition: don't change policy too much")
    ax.legend()
    save(fig, "05_ppo_clip.png")


def fig_offline_ood():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    rng = np.random.default_rng(0)
    # behavior data cluster
    x1 = rng.normal(-1.0, 0.4, 200)
    y1 = rng.normal(0.0, 0.5, 200)
    ax.scatter(x1, y1, s=12, alpha=0.6, label="behavior data support")
    # ood region
    ax.scatter(rng.normal(1.6, 0.25, 40), rng.normal(0.8, 0.25, 40),
               s=18, alpha=0.8, label="OOD actions (unsafe to trust)")
    ax.set_title("Offline RL core risk: OOD actions / extrapolation error")
    ax.set_xlabel("(state,action) feature axis 1 (toy)")
    ax.set_ylabel("feature axis 2 (toy)")
    ax.legend()
    save(fig, "06_offline_ood.png")


def main():
    fig_rl_loop()
    fig_bellman()
    fig_td_error()
    fig_dqn_two_nets()
    fig_ppo_clip()
    fig_offline_ood()


if __name__ == "__main__":
    main()
