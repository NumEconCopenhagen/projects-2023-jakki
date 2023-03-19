from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class householdClass:

    def __init__(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # Preferences
        par.rho = 2
        par.nu = 0.001
        par.epsilon = 1
        par.omega = 0.5

        # Household production
        par.alpha = 0.5
        par.sigma = 1

        # Wages
        par.wageF = 1
        par.wageM = 1
        par.wF_vec = np.linspace(0.8,1.2,5)