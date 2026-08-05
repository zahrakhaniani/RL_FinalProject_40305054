"""Repository-relative path helpers.

Every path in this project is derived from the repository root, which is the
directory holding this file. That keeps all file access relative to the repo so
the scripts can be launched from any working directory without absolute paths.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

ENVIRONMENTS = ROOT / "environments"
MAPS = ENVIRONMENTS / "maps"
EXPERIMENTS = ROOT / "experiments"
CONFIGS = EXPERIMENTS / "configs"
RESULTS = ROOT / "results"
RAW_DATA = RESULTS / "raw_data"
MODELS = RESULTS / "models"
FIGURES = RESULTS / "figures"
VIDEOS = RESULTS / "videos"

DEFAULT_CONFIG = CONFIGS / "default.json"

ALGORITHMS = ("value_iteration", "q_learning", "sarsa_lambda", "transfer", "comparison")


def map_file(student_id: str, variant: str = None) -> Path:
    """Map file for the source maze, or for a transfer target variant."""
    if variant:
        return MAPS / f"maze_{student_id}_{variant}.json"
    return MAPS / f"maze_{student_id}.json"


def subdir(base: Path, name: str) -> Path:
    """Return ``base/name``, creating it if needed."""
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    """Create the full results tree, one subfolder per algorithm."""
    MAPS.mkdir(parents=True, exist_ok=True)
    for base in (RAW_DATA, MODELS, FIGURES, VIDEOS):
        base.mkdir(parents=True, exist_ok=True)
        for algorithm in ALGORITHMS:
            (base / algorithm).mkdir(parents=True, exist_ok=True)


def rel(path) -> str:
    """Format ``path`` relative to the repository root, for logging."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
