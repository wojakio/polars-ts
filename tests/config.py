from pathlib import Path

def get_config_filename(name: str) -> str:
    path = Path("tests", "00_config", name).absolute()
    return path
