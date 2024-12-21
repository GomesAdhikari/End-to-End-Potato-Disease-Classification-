import sys
from omegaconf import OmegaConf
from pathlib import Path
import yaml
from src.logger import info_logger


def read_yaml(path_to_yaml:Path) -> OmegaConf:
    """
    Reads a yaml file and returns the content as a ConfigBox object.
    The ConfigBox is a special type of dictionary that allows you to use the keys as attributes.
    :param path_to_yaml: Path to the yaml file
    :return: ConfigBox object
    """
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        info_logger.info(f"yaml file: {path_to_yaml} loaded successfully as {type(content)}")
        return OmegaConf.create(content)