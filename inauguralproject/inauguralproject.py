from types import SimpleNamespace

import numpy as np
from scipy import optimize
from scipy.optimize import minimize

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
        par.alpha = 0.25
        par.sigma = 1

        # Wages
        par.wageF = 1
        par.wageM = 1
        par.wF_vec = np.linspace(0.8,1.2,5)

        # Solution
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
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = min(HM,HF)
        else:
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
    
    def solve_continuous(self, do_print=False):
        
        par = self.par
        sol = self.sol
        
        # Define the objective function to be minimized
        def objective(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)
        
        # Define the constraint functions
        def constraint1(x):
            LM, HM, LF, HF = x
            return 24 - (LM + HM)
        
        def constraint2(x):
            LM, HM, LF, HF = x
            return 24 - (LF + HF)
        
        # Define the initial guess for the decision variables
        x0 = [12, 12, 12, 12]
        
        # Define the bounds for the decision variables
        bounds = ((0, 24), (0, 24), (0, 24), (0, 24))
        
        # Define the constraints
        cons = [{'type': 'ineq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2}]
        
        # Solve the optimization problem
        opt = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        # Save the optimal decision variables
        sol.LM = opt.x[0]
        sol.HM = opt.x[1]
        sol.LF = opt.x[2]
        sol.HF = opt.x[3]
        
        # Print the results
        if do_print:
            for k, v in sol.__dict__.items():
                print(f"{k} = {v:6.4f}")
        
        return sol