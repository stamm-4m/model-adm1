"""
ADM1 Reactor Simulation (Adapted from PyADM1)
Class: Influent

Author: David Camilo Corrales
Email: David-Camilo.Corrales-Munoz@inrae.fr
Date: 16/03/2026

"""
import pandas as pd

class Influent:
    """
    Class to handle influent state data for ADM1 simulation.
    Reads the CSV and provides a method to get the influent
    values at a given simulation step.
    """
    def __init__(self, csv_file):
        """
        Initialize Influent object and load CSV data.
        """
        self.data = pd.read_csv(csv_file)
        self.vars = ['S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac',
                     'S_h2', 'S_ch4', 'S_IC', 'S_IN', 'S_I',
                     'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa', 
                     'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
                     'S_cation', 'S_anion']

    def get(self, step):
        """
        Return a dictionary of '_in' variables for the given step.
        """
        return {var + "_in": self.data[var][step] for var in self.vars}
    
    def get_time(self):
        """Return the time array from the CSV."""
        return self.data['time'].values