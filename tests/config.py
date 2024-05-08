from pathlib import Path


def get_config_filename(name: str) -> str:
    path = Path("tests", "00_config", name).absolute()
    return str(path)


def get_data_filename(group: str, name: str) -> str:
    path = Path("tests", "01_data", group, name).absolute()
    return str(path)
