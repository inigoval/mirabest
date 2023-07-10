from pathlib import Path
import yaml


def create_path(*args):
    for path in args:
        if not Path.exists(path):
            Path.mkdir(path, parents=True)


def load_config():
    """Helper function to load yaml config file, convert to python dictionary and return."""

    # load global config
    global_path = "config.yml"
    with open(global_path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    return config
