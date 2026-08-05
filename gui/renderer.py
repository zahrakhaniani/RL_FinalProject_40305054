"""Pygame drawing for the maze.

Everything is drawn with vector primitives, so the GUI needs no image assets and
cannot break because a file is missing. The renderer is stateless: it draws
whatever environment and overlays it is handed.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pygame

from environments.maze import Cell

BACKGROUND = (24, 26, 33)
PANEL_BG = (32, 35, 43)
PANEL_LINE = (58, 62, 74)
TEXT = (226, 230, 238)
TEXT_DIM = (146, 154, 170)

COLORS: Dict[int, Tuple[int, int, int]] = {
    Cell.PATH: (238, 239, 241),
    Cell.WALL: (47, 52, 64),
    Cell.START: (123, 192, 67),
    Cell.KEY: (242, 177, 52),
    Cell.DOOR: (181, 101, 29),
    Cell.GOAL: (3, 146, 207),
    Cell.PENALTY: (238, 64, 53),
}
AGENT_COLOR = (231, 76, 118)
TRAIL_COLOR = (120, 180, 220)
DOOR_OPEN = (150, 196, 138)
ARROW_COLOR = (70, 80, 100)

LEGEND = (
    ("start", Cell.START),
    ("key", Cell.KEY),
    ("locked door", Cell.DOOR),
    ("goal (vault)", Cell.GOAL),
    ("penalty cell", Cell.PENALTY),
    ("wall", Cell.WALL),
)

ARROW_VECTORS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    try:
        return pygame.font.SysFont("Consolas", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)


class MazeRenderer:
    def __init__(self, env, cell_size: int = 34, panel_width: int = 330, margin: int = 16):
        self.cell_size = cell_size
        self.panel_width = panel_width
        self.margin = margin
        self.rows, self.cols = env.rows, env.cols
        self._fonts: Dict[Tuple[int, bool], pygame.font.Font] = {}

    # ------------------------------------------------------------------ layout

    @property
    def maze_width(self) -> int:
        return self.cols * self.cell_size

    @property
    def maze_height(self) -> int:
        return self.rows * self.cell_size

    @property
    def window_size(self) -> Tuple[int, int]:
        width = self.maze_width + self.panel_width + 3 * self.margin
        height = max(self.maze_height + 2 * self.margin, 620)
        return width, height

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        return pygame.Rect(
            self.margin + col * self.cell_size,
            self.margin + row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def get_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = font(size, bold)
        return self._fonts[key]

    # ------------------------------------------------------------------- draw

    def draw(
        self,
        surface: pygame.Surface,
        env,
        trail: Optional[Sequence[Tuple[int, int]]] = None,
        value_grid: Optional[np.ndarray] = None,
        policy_grid: Optional[np.ndarray] = None,
        hud_lines: Sequence[Tuple[str, str]] = (),
        status: str = "",
    ) -> None:
        surface.fill(BACKGROUND)
        self._draw_maze(surface, env, value_grid, policy_grid)
        if trail:
            self._draw_trail(surface, trail)
        self._draw_agent(surface, env)
        self._draw_panel(surface, env, hud_lines, status)

    def _draw_maze(self, surface, env, value_grid, policy_grid) -> None:
        normalised = self._normalise(value_grid)

        for row in range(self.rows):
            for col in range(self.cols):
                rect = self.cell_rect(row, col)
                cell = Cell(int(env.grid[row, col]))

                if cell == Cell.WALL:
                    pygame.draw.rect(surface, COLORS[Cell.WALL], rect)
                    pygame.draw.rect(surface, (38, 42, 52), rect, 1)
                    continue

                base = COLORS[Cell.PATH]
                if normalised is not None and np.isfinite(normalised[row, col]):
                    base = self._heat(float(normalised[row, col]))
                pygame.draw.rect(surface, base, rect)
                pygame.draw.rect(surface, (214, 216, 222), rect, 1)

                if policy_grid is not None:
                    self._draw_arrow(surface, rect, int(policy_grid[row, col]))

                self._draw_feature(surface, env, rect, cell, (row, col))

    def _draw_feature(self, surface, env, rect: pygame.Rect, cell: Cell, position) -> None:
        centre = rect.center
        radius = self.cell_size // 2 - 4

        if cell == Cell.START:
            pygame.draw.circle(surface, COLORS[Cell.START], centre, radius - 2, 3)
        elif cell == Cell.KEY:
            if not env.has_key:
                pygame.draw.circle(surface, COLORS[Cell.KEY], centre, radius - 3)
                pygame.draw.circle(surface, COLORS[Cell.PATH], centre, radius - 7)
                pygame.draw.line(
                    surface, COLORS[Cell.KEY],
                    (centre[0], centre[1] + 1), (centre[0] + radius, centre[1] + 1), 3,
                )
            else:
                pygame.draw.circle(surface, (206, 210, 214), centre, radius - 6, 1)
        elif cell == Cell.DOOR:
            colour = DOOR_OPEN if env.has_key else COLORS[Cell.DOOR]
            inner = rect.inflate(-6, -6)
            pygame.draw.rect(surface, colour, inner, border_radius=3)
            if not env.has_key:
                pygame.draw.circle(surface, (60, 40, 20), (inner.centerx, inner.centery), 3)
                for offset in (-4, 4):
                    pygame.draw.line(
                        surface, (120, 70, 25),
                        (inner.left, inner.centery + offset),
                        (inner.right, inner.centery + offset), 1,
                    )
        elif cell == Cell.GOAL:
            points = []
            for index in range(10):
                angle = np.pi / 2 + index * np.pi / 5
                length = radius if index % 2 == 0 else radius * 0.45
                points.append(
                    (centre[0] + length * np.cos(angle), centre[1] - length * np.sin(angle))
                )
            pygame.draw.polygon(surface, COLORS[Cell.GOAL], points)
        elif cell == Cell.PENALTY:
            inset = rect.inflate(-10, -10)
            pygame.draw.line(surface, COLORS[Cell.PENALTY], inset.topleft, inset.bottomright, 3)
            pygame.draw.line(surface, COLORS[Cell.PENALTY], inset.bottomleft, inset.topright, 3)

    def _draw_arrow(self, surface, rect: pygame.Rect, action: int) -> None:
        dx, dy = ARROW_VECTORS.get(action, (0, 0))
        length = self.cell_size * 0.28
        centre = rect.center
        tip = (centre[0] + dx * length, centre[1] + dy * length)
        tail = (centre[0] - dx * length * 0.5, centre[1] - dy * length * 0.5)
        pygame.draw.line(surface, ARROW_COLOR, tail, tip, 2)
        pygame.draw.circle(surface, ARROW_COLOR, (int(tip[0]), int(tip[1])), 2)

    def _draw_trail(self, surface, trail: Sequence[Tuple[int, int]]) -> None:
        recent = list(trail)[-90:]
        for index, (row, col) in enumerate(recent):
            strength = (index + 1) / len(recent)
            rect = self.cell_rect(row, col)
            size = max(3, int(self.cell_size * 0.18 * strength) + 2)
            colour = tuple(
                int(component * strength + 235 * (1 - strength)) for component in TRAIL_COLOR
            )
            pygame.draw.circle(surface, colour, rect.center, size // 2)

    def _draw_agent(self, surface, env) -> None:
        rect = self.cell_rect(*env.agent_pos)
        radius = self.cell_size // 2 - 5
        pygame.draw.circle(surface, AGENT_COLOR, rect.center, radius)
        pygame.draw.circle(surface, (255, 255, 255), rect.center, radius, 2)
        if env.has_key:
            pygame.draw.circle(surface, COLORS[Cell.KEY], rect.center, max(2, radius // 3))

    # ------------------------------------------------------------------ panel

    def _draw_panel(self, surface, env, hud_lines, status: str) -> None:
        left = self.margin * 2 + self.maze_width
        panel = pygame.Rect(left, self.margin, self.panel_width, surface.get_height() - 2 * self.margin)
        pygame.draw.rect(surface, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(surface, PANEL_LINE, panel, 1, border_radius=8)

        x = panel.x + 16
        y = panel.y + 14
        title = self.get_font(19, bold=True).render("RL Maze Solver", True, TEXT)
        surface.blit(title, (x, y))
        y += 30

        if status:
            for line in status.split("\n"):
                surface.blit(self.get_font(14).render(line, True, TEXT_DIM), (x, y))
                y += 19
        y += 6

        label_font = self.get_font(14)
        value_font = self.get_font(14, bold=True)
        for label, value in hud_lines:
            surface.blit(label_font.render(label, True, TEXT_DIM), (x, y))
            rendered = value_font.render(str(value), True, TEXT)
            surface.blit(rendered, (panel.right - 16 - rendered.get_width(), y))
            y += 21

        y += 10
        y = self._draw_energy_bar(surface, env, panel, x, y)
        y += 14
        y = self._draw_legend(surface, panel, x, y)
        self._draw_controls(surface, panel, x, y + 10)

    def _draw_energy_bar(self, surface, env, panel, x: int, y: int) -> int:
        surface.blit(self.get_font(13).render("energy", True, TEXT_DIM), (x, y))
        y += 18
        width = panel.width - 32
        track = pygame.Rect(x, y, width, 14)
        pygame.draw.rect(surface, (48, 52, 64), track, border_radius=7)
        fraction = env.energy / env.max_energy if env.max_energy else 0.0
        if fraction > 0:
            filled = pygame.Rect(x, y, max(3, int(width * fraction)), 14)
            colour = (
                (238, 64, 53) if fraction < 0.2
                else (242, 177, 52) if fraction < 0.5
                else (123, 192, 67)
            )
            pygame.draw.rect(surface, colour, filled, border_radius=7)
        pygame.draw.rect(surface, PANEL_LINE, track, 1, border_radius=7)
        text = self.get_font(12).render(
            f"{env.energy} / {env.max_energy}", True, TEXT
        )
        surface.blit(text, (x + width - text.get_width(), y + 17))
        return y + 34

    def _draw_legend(self, surface, panel, x: int, y: int) -> int:
        surface.blit(self.get_font(13).render("legend", True, TEXT_DIM), (x, y))
        y += 20
        for index, (label, cell) in enumerate(LEGEND):
            column = index % 2
            row = index // 2
            cx = x + column * (panel.width // 2 - 8)
            cy = y + row * 22
            swatch = pygame.Rect(cx, cy, 12, 12)
            pygame.draw.rect(surface, COLORS[cell], swatch, border_radius=2)
            surface.blit(self.get_font(12).render(label, True, TEXT_DIM), (cx + 18, cy - 1))
        return y + ((len(LEGEND) + 1) // 2) * 22

    def _draw_controls(self, surface, panel, x: int, y: int) -> None:
        lines = [
            "space  play / pause",
            "n      single step",
            "r      reset episode",
            "1 2 3  VI / Q-learn / SARSA",
            "m      sparse <-> shaped",
            "v      value heat map",
            "p      policy arrows",
            "t      trail",
            "+ -    speed",
            "c      record frames",
            "esc    quit",
        ]
        surface.blit(self.get_font(13).render("controls", True, TEXT_DIM), (x, y))
        y += 19
        for line in lines:
            if y > panel.bottom - 18:
                break
            surface.blit(self.get_font(12).render(line, True, TEXT_DIM), (x, y))
            y += 17

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _normalise(grid: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if grid is None:
            return None
        finite = np.isfinite(grid)
        if not finite.any():
            return None
        low, high = float(grid[finite].min()), float(grid[finite].max())
        if high - low < 1e-9:
            return None
        scaled = np.full_like(grid, np.nan, dtype=float)
        scaled[finite] = (grid[finite] - low) / (high - low)
        return scaled

    @staticmethod
    def _heat(value: float) -> Tuple[int, int, int]:
        """Simple blue -> yellow ramp for the value overlay."""
        value = float(np.clip(value, 0.0, 1.0))
        start = np.array([54, 76, 138])
        end = np.array([248, 226, 130])
        return tuple(int(component) for component in start + (end - start) * value)
