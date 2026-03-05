import numpy as np


class Gridworld:
    """一个极简网格世界：
    - 0..H-1, 0..W-1
    - 起点 start，终点 goal
    - 动作: 0上 1下 2左 3右
    - 到达 goal: +1 并终止
    - 每步小惩罚 step_cost（鼓励更短路径）
    """

    def __init__(self, height=5, width=5, start=(0, 0), goal=(4, 4), step_cost=-0.01):
        self.h = height
        self.w = width
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.step_cost = float(step_cost)
        self.reset()

    def reset(self):
        self.pos = self.start
        return self._state()

    def _state(self):
        # 用一个整数编码状态，便于做表格法
        r, c = self.pos
        return r * self.w + c

    @property
    def n_states(self):
        return self.h * self.w

    @property
    def n_actions(self):
        return 4

    def step(self, action):
        r, c = self.pos
        if action == 0:
            r2, c2 = r - 1, c
        elif action == 1:
            r2, c2 = r + 1, c
        elif action == 2:
            r2, c2 = r, c - 1
        elif action == 3:
            r2, c2 = r, c + 1
        else:
            raise ValueError("invalid action")

        # 撞墙：留在原地
        if not (0 <= r2 < self.h and 0 <= c2 < self.w):
            r2, c2 = r, c

        self.pos = (r2, c2)
        done = self.pos == self.goal
        reward = 1.0 if done else self.step_cost
        return self._state(), reward, done
