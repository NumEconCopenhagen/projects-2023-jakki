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

        #Solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

    def calc_utility(self,LM,HM,LF,HF):

        par = self.par
        sol = self.sol

        # Consumption of market goods
        C = par.wageM*LM + par.wageF*LF
        
        # Home production
        H = HM**(1-par.alpha)*HF**par.alpha

        # Total consumption
        Q = C**par.omega*H**(1-par.omega)

        # Utility
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
    
        # Disutility
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility
