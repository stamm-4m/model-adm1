"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: Parameters

Author: Margaux Bonal, David Camilo Corrales
Email: margaux.bonal@inrae.fr, David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026

"""

import yaml

class ADM1Parameters:
    def __init__(self, yaml_file="adm1_parameters.yaml"):
        self.yaml_file = yaml_file
        self.params = {}
        self.load_yaml()

    def load_yaml(self):
        """Load parameters from a YAML file and flatten them."""
        with open(self.yaml_file, "r") as file:
            params_dict = yaml.safe_load(file)
        self.params = self.flatten_params(params_dict)

    @staticmethod
    def flatten_params(params_dict):
        """
        Flatten nested parameter dictionary.
        Converts all numeric values to float for safe operations.
        """
        flat = {}
        for category in params_dict.values():
            for key, val in category.items():
                flat[key] = float(val["value"])
        return flat

    def __getattr__(self, name):
        """Allow access to parameters as attributes."""
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"ADM1 parameter '{name}' not found.")

    def __repr__(self):
        return f"<ADM1Parameters: {len(self.params)} parameters loaded>"