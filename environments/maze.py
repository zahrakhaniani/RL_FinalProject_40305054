import numpy as np
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MazeEnv:
    PATH = 0
    WALL = 1
    GOAL = 2
    TRAP = 3
    KEY = 4
    DOOR = 5
    PENALTY = 6

    ACTIONS = {
        Action.UP: (-1, 0),
        Action.RIGHT: (0, 1),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
    }

    def __init__(self, grid, start=(0, 0), goal=None, key=None, door=None,
                 traps=None, penalties=None,
                 reward_step=-0.1, reward_goal=10, reward_trap=-10,
                 reward_key=5, reward_penalty=-3):
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.goal = goal if goal else (self.rows - 1, self.cols - 1)
        self.key = key
        self.door = door
        self.traps = traps if traps else []
        self.penalties = penalties if penalties else []
        self.reward_step = reward_step
        self.reward_goal = reward_goal
        self.reward_trap = reward_trap
        self.reward_key = reward_key
        self.reward_penalty = reward_penalty
        self.agent_pos = start
        self.has_key = False
        self.door_open = False
        self.done = False

    def reset(self):
        self.agent_pos = self.start
        self.has_key = False
        self.door_open = False
        self.done = False
        return self._get_state()

    def copy(self):
        return MazeEnv(
            self.grid.copy(),
            start=self.start,
            goal=self.goal,
            key=self.key,
            door=self.door,
            traps=list(self.traps),
            penalties=list(self.penalties),
            reward_step=self.reward_step,
            reward_goal=self.reward_goal,
            reward_trap=self.reward_trap,
            reward_key=self.reward_key,
            reward_penalty=self.reward_penalty,
        )

    def _get_state(self):
        return (*self.agent_pos, int(self.has_key), int(self.door_open))

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        dr, dc = self.ACTIONS[action]
        new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            return self._get_state(), self.reward_step, False, {"hit_wall": True}

        if self.grid[new_r][new_c] == self.WALL:
            return self._get_state(), self.reward_step, False, {"hit_wall": True}

        if self.grid[new_r][new_c] == self.DOOR and not self.door_open:
            if self.has_key:
                self.door_open = True
            else:
                return self._get_state(), self.reward_step, False, {"door_locked": True}

        self.agent_pos = (new_r, new_c)
        reward = self.reward_step
        info = {}

        if self.agent_pos == self.key and not self.has_key:
            self.has_key = True
            reward = self.reward_key
            info["picked_key"] = True

        if self.agent_pos == self.goal:
            self.done = True
            return self._get_state(), self.reward_goal, True, info

        if self.agent_pos in self.traps:
            self.done = True
            return self._get_state(), self.reward_trap, True, info

        if self.agent_pos in self.penalties:
            reward = self.reward_penalty
            info["penalty"] = True

        return self._get_state(), reward, False, info

    def get_valid_actions(self, state=None):
        if state is None:
            pos = self.agent_pos
            has_key = self.has_key
            door_open = self.door_open
        elif len(state) >= 4:
            pos = state[:2]
            has_key = bool(state[2])
            door_open = bool(state[3])
        else:
            pos = state[:2]
            has_key = self.has_key
            door_open = self.door_open

        valid = []
        for action in Action:
            dr, dc = self.ACTIONS[action]
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                cell = self.grid[r][c]
                if cell == self.WALL:
                    continue
                if cell == self.DOOR and not door_open and not has_key:
                    continue
                valid.append(action)
        return valid

    def get_state_space(self):
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != self.WALL:
                    for has_key in [0, 1]:
                        for door_open in [0, 1]:
                            if self.door is not None and has_key == 0 and door_open == 1:
                                continue
                            states.append((r, c, has_key, door_open))
        return states

    def render(self):
        symbols = {
            self.WALL: "#", self.PATH: ".", self.GOAL: "G",
            self.TRAP: "T", self.KEY: "K", self.DOOR: "D",
            self.PENALTY: "X"
        }
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row += "A"
                else:
                    row += symbols.get(self.grid[r][c], "?")
            print(row)
        print()

    def save_map(self, filepath):
        import json
        data = {
            "grid": self.grid.tolist(),
            "start": list(self.start),
            "goal": list(self.goal),
            "key": list(self.key) if self.key else None,
            "door": list(self.door) if self.door else None,
            "traps": [list(t) for t in self.traps],
            "penalties": [list(p) for p in self.penalties],
            "rows": self.rows,
            "cols": self.cols,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_map(cls, filepath):
        import json
        with open(filepath) as f:
            data = json.load(f)
        grid = np.array(data["grid"], dtype=int)
        start = tuple(data["start"])
        goal = tuple(data["goal"])
        key = tuple(data["key"]) if data["key"] else None
        door = tuple(data["door"]) if data["door"] else None
        traps = [tuple(t) for t in data["traps"]]
        penalties = [tuple(p) for p in data["penalties"]]
        return cls(grid, start=start, goal=goal, key=key, door=door,
                   traps=traps, penalties=penalties)
