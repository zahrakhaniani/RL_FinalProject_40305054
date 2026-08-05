import pygame
from environments.maze import MazeEnv
from environments.icon_assets import IconAtlas, LEGEND_ITEMS


class MazeRenderer:
    def __init__(self, cell_size=32):
        self.cell_size = cell_size
        self.atlas = IconAtlas(cell_size=cell_size)

    def maze_pixel_size(self, env):
        return env.cols * self.cell_size, env.rows * self.cell_size

    def draw_maze(self, surface, env, offset=(0, 0), visited=None, show_agent=True):
        ox, oy = offset
        visited = visited or set()

        for r in range(env.rows):
            for c in range(env.cols):
                x = ox + c * self.cell_size
                y = oy + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size - 1, self.cell_size - 1)
                cell = env.grid[r][c]
                is_visited = (r, c) in visited
                is_agent = show_agent and (r, c) == env.agent_pos

                if cell == MazeEnv.WALL:
                    self.atlas.draw_wall_tile(surface, rect)
                    continue

                self.atlas.draw_path_tile(surface, rect, visited=is_visited and not is_agent)

                if (r, c) == env.goal:
                    self.atlas.draw_icon_centered(surface, "goal", rect)
                elif (r, c) == env.key and not env.has_key:
                    self.atlas.draw_icon_centered(surface, "key", rect)
                elif (r, c) == env.door and not env.door_open:
                    self.atlas.draw_icon_centered(surface, "door", rect)
                elif (r, c) in env.traps:
                    self.atlas.draw_icon_centered(surface, "trap", rect)
                elif (r, c) in env.penalties:
                    self.atlas.draw_icon_centered(surface, "penalty", rect)

                if is_agent:
                    self.atlas.draw_icon_centered(surface, "agent", rect)

    def draw_status_bar(self, surface, rect, lines):
        pygame.draw.rect(surface, (240, 242, 245), rect)
        pygame.draw.line(surface, (210, 214, 220), rect.topleft, rect.topright, 1)
        font = self._font(14)
        y = rect.y + 10
        for line in lines:
            text = font.render(line, True, (30, 35, 40))
            surface.blit(text, (rect.x + 12, y))
            y += 22

    def draw_legend_panel(self, surface, rect):
        self.atlas.ensure_loaded()
        pygame.draw.rect(surface, (236, 239, 244), rect, border_radius=6)
        pygame.draw.rect(surface, (210, 216, 224), rect, 1, border_radius=6)

        title_font = self._font(14, bold=True)
        label_font = self._font(12)
        title = title_font.render("Legend", True, (35, 40, 48))
        surface.blit(title, (rect.x + 10, rect.y + 8))

        cols = 2
        row_h = 28
        col_w = rect.width // cols
        start_y = rect.y + 32

        for idx, (label, icon_name) in enumerate(LEGEND_ITEMS):
            col = idx % cols
            row = idx // cols
            x = rect.x + 10 + col * col_w
            y = start_y + row * row_h
            icon_rect = pygame.Rect(x, y, 22, 22)

            if icon_name is None:
                self.atlas.draw_wall_tile(surface, icon_rect)
            else:
                legend_icon = self.atlas.get_legend(icon_name)
                if legend_icon:
                    icon_pos = (
                        icon_rect.x + (icon_rect.width - legend_icon.get_width()) // 2,
                        icon_rect.y + (icon_rect.height - legend_icon.get_height()) // 2,
                    )
                    surface.blit(legend_icon, icon_pos)

            text = label_font.render(label, True, (55, 60, 68))
            surface.blit(text, (x + 28, y + 4))

    @staticmethod
    def _font(size, bold=False):
        try:
            return pygame.font.SysFont("Segoe UI", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)
