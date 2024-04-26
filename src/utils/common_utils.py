import logging

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """
    load the config.yaml filespecified in the path.

    Args:
        config_path (str): Path to the config.yaml file.

    Returns:
        dict: The loaded config dictionary.
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.info("Config loaded successfully.")
    except FileNotFoundError:
        logger.error("config.yaml file not found.")
        exit(1)
    except yaml.YAMLError as e:
        logger.error("Error loading config.yaml:", e)
        exit(1)

    # Check if config is loaded successfully
    if config is None:
        logger.error("Config could not be loaded.")
        exit(1)

    return config
