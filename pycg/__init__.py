from pathlib import Path

__version__ = '0.1.0'


def get_assets_path():
    return Path(__file__).parent / "assets"
