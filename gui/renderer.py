"""Pygame drawing for the maze.

Everything is drawn with vector primitives, so the GUI needs no image assets and
cannot break because a file is missing. The renderer is stateless: it draws
whatever environment, overlays and readouts it is handed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from environments.maze import Cell

BACKGROUND = (24, 26, 33)
PANEL_BG = (32, 35, 43)
PANEL_LINE = (58, 62, 74)
TEXT = (226, 230, 238)
TEXT_DIM = (146, 154, 170)
ACCENT = (0, 158, 175)

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
SUCCESS_COLOR = (123, 192, 67)
FAILURE_COLOR = (238, 64, 53)
BUMP_COLOR = (255, 214, 102)

LEGEND = (
    ("start", Cell.START),
    ("key", Cell.KEY),
    ("locked door", Cell.DOOR),
    ("goal (vault)", Cell.GOAL),
    ("penalty cell", Cell.PENALTY),
    ("wall", Cell.WALL),
)

ARROW_VECTORS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

OUTCOME_TEXT = {
    "success": ("EPISODE SOLVED", SUCCESS_COLOR),
    "out_of_energy": ("FAILED -- OUT OF ENERGY", FAILURE_COLOR),
    "max_steps": ("FAILED -- STEP LIMIT", FAILURE_COLOR),
}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    try:
        return pygame.font.SysFont("Consolas", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)


class MazeRenderer:
    def __init__(self, env, cell_size: int = 32, panel_width: int = 372, margin: int = 14):
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
        height = max(self.maze_height + 2 * self.margin + 34, 760)
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
        sections: Sequence[Tuple[str, Sequence[Tuple[str, object]]]] = (),
        status: str = "",
        progress: Optional[Tuple[float, str]] = None,
        bump_cell: Optional[Tuple[int, int]] = None,
        controls: Sequence[str] = (),
    ) -> None:
        surface.fill(BACKGROUND)
        self._draw_maze(surface, env, value_grid, policy_grid)
        if trail:
            self._draw_trail(surface, trail)
        if bump_cell is not None:
            self._draw_bump(surface, bump_cell)
        self._draw_agent(surface, env)
        self._draw_outcome_banner(surface, env)
        self._draw_panel(surface, env, sections, status, progress, controls)

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

    def _draw_bump(self, surface, cell: Tuple[int, int]) -> None:
        """Flash the cell the agent just failed to leave."""
        rect = self.cell_rect(*cell)
        pygame.draw.rect(surface, BUMP_COLOR, rect.inflate(2, 2), 3)

    def _draw_agent(self, surface, env) -> None:
        rect = self.cell_rect(*env.agent_pos)
        radius = self.cell_size // 2 - 5
        pygame.draw.circle(surface, AGENT_COLOR, rect.center, radius)
        pygame.draw.circle(surface, (255, 255, 255), rect.center, radius, 2)
        if env.has_key:
            pygame.draw.circle(surface, COLORS[Cell.KEY], rect.center, max(2, radius // 3))

    def _draw_outcome_banner(self, surface, env) -> None:
        if not env.done or env.outcome not in OUTCOME_TEXT:
            return
        label, colour = OUTCOME_TEXT[env.outcome]
        text = self.get_font(17, bold=True).render(label, True, colour)
        box = pygame.Rect(
            self.margin, self.margin + self.maze_height + 6, self.maze_width, 26
        )
        pygame.draw.rect(surface, PANEL_BG, box, border_radius=5)
        pygame.draw.rect(surface, colour, box, 1, border_radius=5)
        surface.blit(text, (box.centerx - text.get_width() // 2, box.y + 4))

    # ------------------------------------------------------------------ panel

    def _draw_panel(self, surface, env, sections, status, progress, controls) -> None:
        left = self.margin * 2 + self.maze_width
        panel = pygame.Rect(
            left, self.margin, self.panel_width, surface.get_height() - 2 * self.margin
        )
        pygame.draw.rect(surface, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(surface, PANEL_LINE, panel, 1, border_radius=8)

        x = panel.x + 15
        y = panel.y + 12
        title = self.get_font(18, bold=True).render("RL Maze Solver", True, TEXT)
        surface.blit(title, (x, y))
        y += 25

        if status:
            for line in status.split("\n"):
                surface.blit(self.get_font(12).render(line, True, TEXT_DIM), (x, y))
                y += 16
        y += 4

        if progress is not None:
            y = self._draw_progress(surface, panel, x, y, *progress)

        y = self._draw_energy_bar(surface, env, panel, x, y)
        y += 6

        for heading, rows in sections:
            if y > panel.bottom - 40:
                break
            surface.blit(self.get_font(12, bold=True).render(heading, True, ACCENT), (x, y))
            y += 17
            for label, value in rows:
                if y > panel.bottom - 26:
                    break
                surface.blit(self.get_font(12).render(label, True, TEXT_DIM), (x, y))
                rendered = self.get_font(12, bold=True).render(str(value), True, TEXT)
                surface.blit(rendered, (panel.right - 15 - rendered.get_width(), y))
                y += 16
            y += 6

        y = self._draw_legend(surface, panel, x, y)
        self._draw_controls(surface, panel, x, y + 4, controls)

    def _draw_progress(self, surface, panel, x: int, y: int, fraction: float, label: str) -> int:
        surface.blit(self.get_font(12).render(label, True, TEXT_DIM), (x, y))
        y += 16
        width = panel.width - 30
        track = pygame.Rect(x, y, width, 9)
        pygame.draw.rect(surface, (48, 52, 64), track, border_radius=5)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if fraction > 0:
            pygame.draw.rect(
                surface, ACCENT, pygame.Rect(x, y, max(3, int(width * fraction)), 9),
                border_radius=5,
            )
        pygame.draw.rect(surface, PANEL_LINE, track, 1, border_radius=5)
        return y + 18

    def _draw_energy_bar(self, surface, env, panel, x: int, y: int) -> int:
        surface.blit(self.get_font(12).render("energy remaining", True, TEXT_DIM), (x, y))
        y += 16
        width = panel.width - 30
        track = pygame.Rect(x, y, width, 13)
        pygame.draw.rect(surface, (48, 52, 64), track, border_radius=6)
        fraction = env.energy / env.max_energy if env.max_energy else 0.0
        if fraction > 0:
            colour = (
                (238, 64, 53) if fraction < 0.2
                else (242, 177, 52) if fraction < 0.5
                else (123, 192, 67)
            )
            pygame.draw.rect(
                surface, colour, pygame.Rect(x, y, max(3, int(width * fraction)), 13),
                border_radius=6,
            )
        pygame.draw.rect(surface, PANEL_LINE, track, 1, border_radius=6)
        text = self.get_font(11).render(f"{env.energy} / {env.max_energy}", True, TEXT)
        surface.blit(text, (x + width - text.get_width(), y + 15))
        return y + 32

    def _draw_legend(self, surface, panel, x: int, y: int) -> int:
        if y > panel.bottom - 70:
            return y
        surface.blit(self.get_font(12, bold=True).render("legend", True, ACCENT), (x, y))
        y += 17
        for index, (label, cell) in enumerate(LEGEND):
            column = index % 2
            row = index // 2
            cx = x + column * (panel.width // 2 - 8)
            cy = y + row * 18
            pygame.draw.rect(surface, COLORS[cell], pygame.Rect(cx, cy, 11, 11), border_radius=2)
            surface.blit(self.get_font(11).render(label, True, TEXT_DIM), (cx + 16, cy - 1))
        return y + ((len(LEGEND) + 1) // 2) * 18 + 4

    def _draw_controls(self, surface, panel, x: int, y: int, controls: Sequence[str]) -> None:
        if not controls or y > panel.bottom - 30:
            return
        surface.blit(self.get_font(12, bold=True).render("controls", True, ACCENT), (x, y))
        y += 16
        for line in controls:
            if y > panel.bottom - 16:
                break
            surface.blit(self.get_font(11).render(line, True, TEXT_DIM), (x, y))
            y += 14

    # ------------------------------------------------------------------ gallery

    def draw_gallery(
        self,
        surface: pygame.Surface,
        image: Optional[pygame.Surface],
        caption: str,
        index: int,
        total: int,
        hint: str,
    ) -> None:
        """Full-window viewer for the saved figures."""
        surface.fill(BACKGROUND)
        width, height = surface.get_size()
        header = self.get_font(15, bold=True).render(
            f"saved results  [{index + 1}/{total}]" if total else "no figures found",
            True, TEXT,
        )
        surface.blit(header, (self.margin, 10))
        surface.blit(self.get_font(12).render(caption, True, TEXT_DIM), (self.margin, 32))
        surface.blit(self.get_font(12).render(hint, True, TEXT_DIM), (self.margin, height - 24))

        if image is None:
            return
        area = pygame.Rect(self.margin, 54, width - 2 * self.margin, height - 90)
        scale = min(area.width / image.get_width(), area.height / image.get_height(), 1.0)
        scaled = pygame.transform.smoothscale(
            image, (int(image.get_width() * scale), int(image.get_height() * scale))
        )
        surface.blit(
            scaled,
            (area.centerx - scaled.get_width() // 2, area.centery - scaled.get_height() // 2),
        )

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
