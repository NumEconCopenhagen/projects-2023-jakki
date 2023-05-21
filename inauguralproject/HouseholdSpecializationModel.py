
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        opt = self.opt = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu_M = 0.001
        par.nu_F = 0.001
        par.epsilon_M = 1.0
        par.epsilon_F = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        opt.LM_vec = np.zeros(par.wF_vec.size)
        opt.HM_vec = np.zeros(par.wF_vec.size)
        opt.LF_vec = np.zeros(par.wF_vec.size)
        opt.HF_vec = np.zeros(par.wF_vec.size)

        opt.beta0 = np.nan
        opt.beta1 = np.nan
        opt.residual = np.nan

    def calc_utility(self,LM,HM,LF,HF):
     
        par = self.par
        opt = self.opt

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        with np.errstate(divide='ignore', invalid='ignore'):
            if par.sigma == 1:
                H = HM**(1-par.alpha)*HF**par.alpha
            elif par.sigma == 0:
                H = np.minimum(HM,HF)
            else:
                H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
        
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)

        with np.errstate(invalid='ignore'):
            utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_F_ = 1+1/par.epsilon_F
        epsilon_M_ = 1+1/par.epsilon_M
        TM = LM+HM
        TF = LF+HF

        disutility = par.nu_M*(TM**epsilon_M_/epsilon_M_)+par.nu_F*(TF**epsilon_F_/epsilon_F_)
        
        return utility - disutility

    def solve_discrete(self):
        
        par = self.par
        opt = self.opt
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel()
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument and store solution
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        return opt

    def solve_continous(self,do_print=False):
        """ solve model continously """
        par = self.par
        opt = self.opt

        def objective(x):
            return self.calc_utility(x[0], x[1], x[2], x[3])
        
        obj = lambda x: - objective(x)
        constraints = ({'type': 'ineq', 'fun': lambda x: (24 - (x[0]+x[1]) ) and (24 - (x[2]+x[3]))})
        guess = [4]*4
        bounds = [(0, 24)]*4
     # d. find maximizing argument
        solution = optimize.minimize(obj,
                            guess,
                            method='Nelder-Mead',
                            bounds=bounds,
                            constraints=constraints)
    
        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]
        opt.u = self.calc_utility(opt.LM, opt.HM, opt.LF, opt.HF)

        return opt

    def solve_wF_vec(self, discrete=False, basin=False, do_print=False):

        par = self.par
        opt = self.opt

        # a. setting up vectors to store results
        opt.logHFHM = np.zeros(par.wF_vec.size)
        opt.HF_vec = np.zeros(par.wF_vec.size)
        opt.HM_vec = np.zeros(par.wF_vec.size)
        opt.LF_vec = np.zeros(par.wF_vec.size)
        opt.LM_vec = np.zeros(par.wF_vec.size)

        # b. loop over wF
        for i,wF in enumerate(par.wF_vec):
            par.wF = wF

            opt = self.solve_continous()
            
            # ii. store results
            opt.logHFHM[i] = np.log(opt.HF/opt.HM)
            opt.HM_vec[i] = opt.HM
            opt.HF_vec[i] = opt.HF
            opt.LF_vec[i] = opt.LF
            opt.LM_vec[i] = opt.LM

        return opt
    
    def run_regression(self):

        par = self.par
        opt = self.opt

        x = np.log(par.wF_vec)
        y = np.log(opt.HF_vec/opt.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        opt.beta0,opt.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,do_print=False):
        
        par = self.par
        opt = self.opt

        # Target function
        def target(x):
                par.alpha, par.sigma = x
                self.solve_wF_vec()
                self.run_regression()
                opt.residual = (opt.beta0-par.beta0_target)**2  + (opt.beta1-par.beta1_target)**2
                return opt.residual
            
        # Initial start guesses for Nelder-Mead Optimization
        x0=[0.5,0.1]
        bounds = ((0,1),(0,1))

        solution = optimize.minimize(target,
                                        x0,
                                        method='Nelder-Mead',
                                        bounds=bounds)
            
        # Storing results
        opt.alpha = solution.x[0]
        opt.sigma = solution.x[1]

        # Printing
        if do_print:
                print(f'\u03B1_opt = {opt.alpha:6.4f}') 
                print(f'\u03C3_opt = {opt.sigma:6.4f}')
                print(f'Residual_opt = {opt.residual:6.4f}')
    
    def estimate_extended(self,sigma=None,epsilon_M=None,epsilon_F=None,extend=True):
        par = self.par
        opt = self.opt

        if extend==True:
            #We make a new function, which defines the dif "different"
            def dif(x):
                par = self.par
                opt = self.opt
                par.sigma = x[0]
                par.epsilon_M = x[1]
                par.epsilon_F = x[2]
                self.solve_wF_vec()
                self.run_regression()
                dif = (opt.beta0 - par.beta0_target)**2 + (opt.beta1 - par.beta1_target)**2
                return dif
        
            result = optimize.minimize(dif, [sigma,epsilon_F,epsilon_M], bounds=[(0.01,2.0),(0.01,2.0),(0.01,2.0)], method='Nelder-Mead')
            opt.sigma = result.x[0]
            opt.epsilon_M = result.x[1]
            opt.epsilon_F = result.x[2]

            return opt
        
        elif extend==False:
            def dif(x):
                par = self.par
                opt = self.opt
                par.sigma = x[0]
                self.solve_wF_vec()
                self.run_regression()
                dif = (opt.beta0 - par.beta0_target)**2 + (opt.beta1 - par.beta1_target)**2  
                return dif
        
            result = optimize.minimize(dif, [sigma], bounds=[(0.01,2.0)], method='Nelder-Mead')
            opt.sigma = result.x[0]

            return opt