# 06 · 离线 RL 与评估：只有历史数据也能学

很多真实业务（推荐、广告、医疗、金融）里，你不能随便让策略在线探索。
你往往只有一堆历史日志（state, action, reward, next_state）。

这就是 **离线 RL（Offline RL）**。

## 离线 RL 的核心难点：分布外（OOD）动作
在线 RL：你可以采样到“你想尝试的动作”。
离线 RL：数据只覆盖了“历史策略做过的动作”。

如果你在离线里学出一个策略，喜欢选择历史从没做过的动作：
- 你的 Q 值/回报估计可能是“瞎猜”
- 上线风险巨大

很多离线 RL 算法的本质是在做：
> **别太相信数据没覆盖的地方。**

代表思路：CQL、BCQ、IQL 等。

## 评估：不上线怎么知道好不好？（OPE）
**离线策略评估（Off-Policy Evaluation, OPE）**常见方法：
1. **重要性采样（IS）**：用概率比修正分布偏差（但方差可能很大）
2. **模型法（Model-based）**：学一个环境模型再模拟（模型误差会累积）
3. **Doubly Robust（DR）**：混合两者，通常更稳

## 本教程提供的最小可跑例子
我们会生成一个“日志数据集”：
- 行为策略（behavior policy）是 ε-greedy
- 我们用它采样出一堆 (s,a,r,s')
- 再尝试用离线数据学习/评估

文件：
- `code/offline/make_offline_dataset.py`
- `code/offline/ope_is_dr.py`
- 数据输出到 `data/offline_grid.csv`

运行：
```bash
python code/offline/make_offline_dataset.py
python code/offline/ope_is_dr.py
```

下一章：`07_rl_for_llm_frontiers.md`
