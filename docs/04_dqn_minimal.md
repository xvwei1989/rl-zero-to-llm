# 04 · DQN：用神经网络来学 Q(s,a)

> 这一章目标：理解“为什么表格法不行、为什么神经网络直接套 Q-learning 会不稳定、DQN 的两个稳定器是什么”。

---

## 1. 为什么表格 Q-learning 不够用
表格法要求你对每个 (s,a) 存一个数。
- Gridworld：25 个状态没问题
- Atari：状态是 84x84x4 的像素，根本存不下

所以需要函数逼近：
> 用神经网络 Q_θ(s,a) 近似 Q(s,a)

---

## 2. 直接“Q-learning + 神经网络”会不稳定的原因
如果你直接用
\[ target = r + \gamma \max_{a'} Q_\theta(s',a') \]

来训练同一个网络 Q_θ，会出现两个主要问题：

### 问题 A：目标在动（moving target）
- 你用网络自己算 target
- 但网络参数更新后，target 也变了
- 相当于“边换尺子边量身高”，容易震荡

### 问题 B：样本强相关（correlated samples）
- 连续交互得到的数据高度相关
- SGD 更喜欢 i.i.d 随机样本
- 否则容易过拟合局部轨迹，训练不稳

---

## 3. DQN 的两个关键稳定器
### 3.1 经验回放 Replay Buffer
把经历 `(s,a,r,s',done)` 存起来，然后**随机采样**训练。
好处：
- 打破时间相关性
- 提升样本利用率（一条经验可以训练多次）

### 3.2 目标网络 Target Network
维护两个网络：
- 在线网络 Q_θ：负责被训练
- 目标网络 Q_θ−：隔一段时间从 Q_θ 拷贝一次

target 用 Q_θ− 计算：
\[ target = r + \gamma \max_{a'} Q_{\theta^-}(s',a') \]

好处：
- target 稳定很多

---

## 4. DQN 的训练流程（你应该能复述）
1) 用 ε-greedy 与环境交互
2) 把经验放进 Replay Buffer
3) 从 Buffer 随机采一批
4) 用 TD loss（常用 Huber loss）训练 Q_θ
5) 每隔 N 步更新一次 Target 网络

---

## 5. 进阶但常见的 DQN 改进（知道名字就够）
- **Double DQN**：减少 `max` 带来的过估计
- **Dueling DQN**：把 V(s) 和 Advantage 分开建模
- **Prioritized Replay**：更常采“TD error 大”的样本
- **Noisy Nets**：把探索写进网络噪声

---

## 6. 本仓库的最小 DQN（为了可读性做的取舍）
- 环境：gridworld（离散动作）
- 状态编码：one-hot（方便解释）
- 网络：两层 MLP（容易读）

文件：`code/dqn/train_dqn_grid.py`

运行：
```bash
python3 code/dqn/train_dqn_grid.py
```

---

## 7. 小练习
1) 把目标网络更新频率调大/调小，会发生什么？
2) 把 replay buffer 变得很小，会更不稳定吗？为什么？
3) 试着实现 Double DQN（核心：选动作用在线网，评估用目标网）。

下一章：`05_ppo_minimal.md`
