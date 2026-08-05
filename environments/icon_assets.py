import os
import pygame

ICONS_DIR = os.path.join(os.path.dirname(__file__), "Icons")

ICON_FILES = {
    "agent": "Roobot.png",
    "goal": "Goal.png",
    "key": "Key.png",
    "door": "Door.png",
    "trap": "Trap.png",
    "penalty": "Penalty.png",
}

LEGEND_ITEMS = [
    ("Agent", "agent"),
    ("Goal", "goal"),
    ("Key", "key"),
    ("Door", "door"),
    ("Trap", "trap"),
    ("Penalty", "penalty"),
    ("Wall", None),
]


class IconAtlas:
    TILE_COLORS = {
        "path": (228, 230, 235),
        "wall": (45, 48, 55),
        "visited": (200, 206, 214),
    }

    def __init__(self, cell_size=32):
        self.cell_size = cell_size
        self.icons = {}
        self._legend_icons = {}
        self._loaded = False

    def ensure_loaded(self):
        if not self._loaded:
            self._load_icons()
            self._loaded = True

    def _load_icons(self):
        legend_size = 22
        for name, filename in ICON_FILES.items():
            path = os.path.join(ICONS_DIR, filename)
            if not os.path.exists(path):
                continue
            image = pygame.image.load(path)
            if pygame.display.get_surface() is not None:
                image = image.convert_alpha()
            inner = max(4, self.cell_size - 6)
            self.icons[name] = pygame.transform.smoothscale(image, (inner, inner))
            self._legend_icons[name] = pygame.transform.smoothscale(image, (legend_size, legend_size))

    def get(self, name):
        self.ensure_loaded()
        return self.icons.get(name)

    def get_legend(self, name):
        self.ensure_loaded()
        return self._legend_icons.get(name)

    def draw_wall_tile(self, surface, rect):
        pygame.draw.rect(surface, self.TILE_COLORS["wall"], rect)

    def draw_path_tile(self, surface, rect, visited=False):
        color = self.TILE_COLORS["visited"] if visited else self.TILE_COLORS["path"]
        pygame.draw.rect(surface, color, rect)

    def draw_icon_centered(self, surface, name, rect):
        self.ensure_loaded()
        icon = self.icons.get(name)
        if icon is None:
            return
        pos = (
            rect.x + (rect.width - icon.get_width()) // 2,
            rect.y + (rect.height - icon.get_height()) // 2,
        )
        surface.blit(icon, pos)
