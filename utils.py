from pathlib import Path


def create_path(*args):
    for path in args:
        if not Path.exists(path):
            Path.mkdir(path, parents=True)
