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
        par.alpha_vec = [0.25, 0.50, 0.75]
        par.sigma_vec = [0.5, 1.0, 1.5]

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
            H = ((1-par.alpha)*HM**(par.sigma-1/par.sigma)+par.alpha*HF**(par.sigma-1/par.sigma))**(par.sigma/par.sigma-1)

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

    def solve_discrete(self,do_print=False):
    
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # All possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x)
        
        LM = LM.ravel() 
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # Calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # Set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) 
        u[I] = -np.inf
    
        # Find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # Print results
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt