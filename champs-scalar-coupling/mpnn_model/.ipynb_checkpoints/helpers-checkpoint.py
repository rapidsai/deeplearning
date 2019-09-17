import yaml 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#############################################################################################################
#                                                                                                           #
#                                   Load experiment configuration                                           #
#                                                                                                           #
#############################################################################################################
import yaml
def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    return cfg
