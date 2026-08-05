import numpy as np
import random
import json
import os
from collections import deque
from .maze import MazeEnv


class MazeGenerator:
    def __init__(self, student_id="40305054", wall_density=0.15, seed=None):
        b = int(student_id[-2])
        self.N = 15 + (b % 4)
        self.wall_density = max(wall_density, 0.15)
        self.student_id = student_id
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        else:
            combined = hash(student_id) % (2**32)
            np.random.seed(combined)
            random.seed(combined)

    def generate(self):
        while True:
            grid = self._create_grid()
            key_pos, door_pos = self._place_key_and_door(grid)
            penalty_pos = self._place_penalties(grid)
            trap_pos = self._place_traps(grid)

            start = (0, 0)
            goal = (self.N - 1, self.N - 1)
            grid[start[0]][start[1]] = MazeEnv.PATH
            grid[goal[0]][goal[1]] = MazeEnv.GOAL
            grid[key_pos[0]][key_pos[1]] = MazeEnv.KEY
            grid[door_pos[0]][door_pos[1]] = MazeEnv.DOOR
            for p in penalty_pos:
                grid[p[0]][p[1]] = MazeEnv.PENALTY
            for t in trap_pos:
                grid[t[0]][t[1]] = MazeEnv.TRAP

            if self._validate(grid, start, key_pos, door_pos, goal):
                env = MazeEnv(
                    grid, start=start, goal=goal,
                    key=key_pos, door=door_pos,
                    traps=trap_pos, penalties=penalty_pos,
                )
                return env

    def _create_grid(self):
        grid = np.zeros((self.N, self.N), dtype=int)
        total_cells = self.N * self.N
        num_walls = int(total_cells * self.wall_density)

        candidates = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if (r, c) != (0, 0) and (r, c) != (self.N - 1, self.N - 1)
        ]
        random.shuffle(candidates)
        for r, c in candidates[:num_walls]:
            grid[r][c] = MazeEnv.WALL

        return grid

    def _place_key_and_door(self, grid):
        candidates = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if grid[r][c] == MazeEnv.PATH
        ]
        random.shuffle(candidates)

        key_pos = candidates[0]
        grid[key_pos[0]][key_pos[1]] = MazeEnv.KEY

        door_candidates = [
            (r, c) for r, c in candidates[1:]
            if abs(r - key_pos[0]) + abs(c - key_pos[1]) > self.N // 3
        ]
        if not door_candidates:
            door_candidates = candidates[1:2]

        door_pos = door_candidates[0]
        return key_pos, door_pos

    def _place_penalties(self, grid, num_penalties=5):
        candidates = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if grid[r][c] == MazeEnv.PATH
        ]
        random.shuffle(candidates)
        penalties = []
        for pos in candidates[:num_penalties]:
            penalties.append(pos)
        return penalties

    def _place_traps(self, grid, num_traps=3):
        candidates = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if grid[r][c] == MazeEnv.PATH
        ]
        random.shuffle(candidates)
        traps = []
        for pos in candidates[:num_traps]:
            traps.append(pos)
        return traps

    def _validate(self, grid, start, key, door, goal):
        path_start_to_key = self._bfs(grid, start, key)
        if not path_start_to_key:
            return False

        path_key_to_door = self._bfs(grid, key, door)
        if not path_key_to_door:
            return False

        path_door_to_goal = self._bfs(grid, door, goal)
        if not path_door_to_goal:
            return False

        return True

    def _bfs(self, grid, start, goal):
        if start == goal:
            return [start]

        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)

        while queue:
            (r, c), path = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    if (nr, nc) not in visited and grid[nr][nc] != MazeEnv.WALL:
                        new_path = path + [(nr, nc)]
                        if (nr, nc) == goal:
                            return new_path
                        visited.add((nr, nc))
                        queue.append(((nr, nc), new_path))
        return None

    def save_map(self, env, filepath):
        env.save_map(filepath)

    def load_map(self, filepath):
        return MazeEnv.load_map(filepath)

    def print_info(self):
        print(f"Student ID: {self.student_id}")
        print(f"Grid size: {self.N}x{self.N}")
        print(f"Wall density: {self.wall_density:.0%}")
