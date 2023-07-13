import yaml

from pathlib import Path


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

    survey_hashmap = {"vla_first": "VLA FIRST (1.4 GHz)", "nvss": "NVSS"}

    # Transcribe survey name for SkyView
    config["survey"] = survey_hashmap[config["survey"]]

    return config
