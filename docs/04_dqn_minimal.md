# 04 · DQN：用神经网络来学 Q(s,a)

表格法的问题：状态一多就爆炸（比如图像像素状态）。
**DQN（Deep Q-Network）**用神经网络近似 Q(s,a)。

## DQN 解决了表格 Q-learning 的两个“深坑”
直接用神经网络做 Q-learning 会很不稳定，主要因为：
1. 目标值 target 依赖同一个网络（自己追自己）
2. 连续采样的经验强相关（相邻步骤很像）

DQN 的两个关键技巧：
- **经验回放（Replay Buffer）**：把经验存起来，随机采样训练，打破相关性
- **目标网络（Target Network）**：用一个“慢更新”的网络来算 target，让训练更稳

## 本教程的极简 DQN
为了“零基础可读”，我们用一个简单的离散环境（gridworld），状态用整数/one-hot 编码。

配套代码：
- `code/dqn/train_dqn_grid.py`

运行：
```bash
python code/dqn/train_dqn_grid.py
```

你应该看到：
- loss 逐渐下降（不一定单调）
- 平均回报上升

下一章：`05_ppo_minimal.md`
