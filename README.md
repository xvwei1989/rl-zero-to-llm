# 零基础强化学习（RL）通俗教程：从掷硬币到训练 LLM

面向：**完全零基础**（会一点 Python 更好）。目标：用最少数学、最多直觉 + 小案例，把 RL 的核心方法讲清楚，并配**可运行代码**与**模拟数据**。

## 你将学到什么
- RL 是什么？和监督学习有什么不同？
- 从 **多臂老虎机（Bandit）**理解“探索 vs 利用”
- 从 **网格世界（Gridworld）**理解状态、动作、奖励、折扣、价值函数
- **Q-learning / SARSA**：最经典的表格型 RL
- **DQN**：用神经网络做 Q 函数（深度强化学习入门）
- **PPO**：现代主力算法之一（策略梯度 + 稳定训练）
- **离线 RL / 反事实评估**：只有历史数据也能学
- **LLM 里的 RL（RLHF/RLAIF/DPO/GRPO…）**：最新进展与工程实践直觉

## 目录
- `docs/`（每章正文，含 Mermaid 图解）
  - `00_rl_in_10_minutes.md`
  - `01_bandit_explore_exploit.md`
  - `02_mdp_gridworld.md`
  - `03_tabular_q_learning.md`
  - `04_dqn_minimal.md`
  - `05_ppo_minimal.md`
  - `06_offline_rl_and_evaluation.md`
  - `07_rl_for_llm_frontiers.md`
- `code/`：每章配套代码（尽量少依赖）
- `data/`：模拟数据（CSV/NPZ）

## 快速开始
### 1) 环境
建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 跑一个最小例子：多臂老虎机
```bash
python code/bandit/run_bandit.py
```

### 3) 训练一个最小 DQN（网格世界）
```bash
python code/dqn/train_dqn_grid.py
```

## 为什么这份教程“零基础友好”
- 每个概念都用一个**可画出来的**小故事解释
- 先直觉，再公式；公式只保留最必要的
- 每章都有**能跑的代码**，并给出“你应该看到什么现象”

## 参考
- Sutton & Barto: *Reinforcement Learning: An Introduction*（经典）
- OpenAI Spinning Up（入门实践）
- DeepMind x UCL RL Course（系统课程）

---
如果你希望我把它直接发布到某个 GitHub repo（创建仓库、推送、写 CI/徽章、加中文/英文双语），把 repo 地址或组织名发我即可。
