# 03 · 表格型 Q-learning：最经典的 RL 入门算法

我们先从“状态数量不大”的场景开始（比如 5x5 网格世界只有 25 个状态）。

## Q-learning 解决什么？
当你不知道环境的转移概率 P(s'|s,a) 时，你仍然想学到：
- 在每个状态 s 下，哪个动作 a 更好？

Q-learning 的做法：
> 直接用交互数据（s,a,r,s'）把 Q(s,a) 慢慢“修正到正确值”。

## 一条核心更新（直觉版）
每次你在 s 做 a 得到 r，并到达 s'：
1. 你对 (s,a) 的旧看法：Q(s,a)
2. 你对未来最乐观的估计：max_a' Q(s',a')
3. 目标值 target = r + γ * max_a' Q(s',a')
4. 用学习率 α 把 Q(s,a) 往 target 拉近

这是一种“自举（bootstrap）”：
- 用自己当前的估计来更新自己

## 为什么它能工作？
直觉：
- 如果 Q 低估了好动作，它会被未来回报拉上来
- 如果 Q 高估了坏动作，它会被实际奖励拉下来

## ε-greedy 策略
训练时常用 ε-greedy 来探索：
- 大概率选当前 Q 最大的动作
- 小概率随机

## 配套代码
- `code/gridworld/gridworld_env.py`
- `code/gridworld/train_q_learning.py`

运行：
```bash
python code/gridworld/train_q_learning.py
```
你应该看到：
- 回合长度逐渐变短（更快到终点）
- 平均回报逐渐上升

下一章：`04_dqn_minimal.md`
