import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_path(key):
    """Directory `paths.<key>` of project-config.toml, created if missing."""
    with open(PROJECT_ROOT / "project-config.toml", "rb") as f:
        config = tomllib.load(f)

    path = PROJECT_ROOT / config["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path
