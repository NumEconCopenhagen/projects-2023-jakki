from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilonM = 1.0
        par.epsilonF = 1.0
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
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ 
        calculating the utility given the parmeters LM, HM, LF and HF 
        """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma==1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma==0:
            H = np.min(np.array([HM,HF]))
        else:
            HM = np.fmax(HM,1e-8)
            HF = np.fmax(HF,1e-8)
            sigma_ = (par.sigma-1)/par.sigma
            H = ((1-par.alpha)*HM**sigma_ + (par.alpha)*HF**sigma_)**((sigma_)**-1)
            
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilonM_ = 1+1/par.epsilonM
        epsilonF_ = 1+1/par.epsilonF
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilonM_/epsilonM_+TF**epsilonF_/epsilonF_)
        
        return utility - disutility

    def solve_discrete(self):
        """ solves the model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        return opt
        

    def solve_con(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective(x):
            return self.calc_utility(x[0], x[1], x[2], x[3])
        
        obj = lambda x: - objective(x)
        constraints = ({'type': 'ineq', 'fun': lambda x: (24 - (x[0]+x[1]) ) and (24 - (x[2]+x[3]))})
        guess = [4]*4
        bounds = [(0, 24)]*4
     # d. find maximizing argument
        result = optimize.minimize(obj,
                            guess,
                            method='Nelder-Mead',
                            bounds=bounds,
                            constraints=constraints)
    
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]
        opt.u = self.calc_utility(opt.LM, opt.HM, opt.LF, opt.HF)


        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
  

    def solve_wF_vec(self,discrete=False):
    
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. solve the model (discretly or continously) for a given female wage
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF 
            
            if discrete==False:
                opt = self.solve_con() 
            elif discrete==True:
                opt = self.solve_discrete() 
            else:
                print("discrete must be True or False")

            # saves solution for each female wage 
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF
            
        return sol


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        # a. define log values
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)

        # b. run regression 
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        return sol
    
    def estimate(self,alpha=0.5,sigma=0.5):
        """estimation of the optimal values for alpha and sigma """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # The error function
        def error(x):
            alpha, sigma = x.ravel()
            par.alpha = alpha 
            par.sigma = sigma 
            
            self.solve_wF_vec() 
            sol = self.run_regression() # estimating beta0 and beta1
            error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #The errors are calculated as the sum of squared differences between the estimated and the target values of beta0 and beta1
            return error
        
        # 'Nelder-Mead' to minimize the error function with bounds
        solution = optimize.minimize(error,[alpha,sigma],method='Nelder-Mead', bounds=[(0.0001,0.999), (0.0001,10)])
        
        opt.alpha = solution.x[0]
        opt.sigma = solution.x[1]
        error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #The errors are calculated as the sum of squared differences between the estimated and the target values of beta0 and beta1
        opt.error = error
        
        return opt
    


    def estimation_extended(self,sigma=0.5,epsilon=1,extended=True):
        "Estimation when alpha is constant"
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        par.alpha = 0.5

        if extended==True: # if extended is true, the estimation is done with the extended model
            def error(x):
                sigma, epsilon = x.ravel()
                par.sigma = sigma 
                par.epsilon = epsilon

                #The optimal household production
                self.solve_wF_vec()  
                sol = self.run_regression() 
                error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 
                return error
            
            results = optimize.minimize(error,[sigma, epsilon],method='Nelder-Mead', bounds=[(0,2),(0.5,2),(0.5,2)])
            

            opt.sigma = results.x[0]
            opt.epsilon = results.x[1]
        
        
        elif extended==False:
            def error(x):
                par.sigma = x 
                
                self.solve_wF_vec() #Optimal household production 
                sol = self.run_regression() # beta0 and beta1
                error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 
                return error

          
            results = optimize.minimize(error,[sigma],method='Nelder-Mead', bounds=[(0,2)])


            opt.sigma = results.x

        else:
            print('extended must be either True or False')

        # d. saves error value  
        error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #calculates error
        opt.error = error

        return opt