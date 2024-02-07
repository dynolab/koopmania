from pathlib import Path

def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent

def get_config_path() -> Path:
    return get_project_path()/"config"

def split(x, splitter):
    return x[:-splitter], x[-splitter:]

def split_t(t, splitter):
    return t[:-splitter], t[:splitter]
