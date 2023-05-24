
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass: #We create a class to call upon in our Notebook. 

    def __init__(self): #We use the init function to set up parameters, which we use through the assignment.
        """ setup model """

        # a. create namespaces. This makes us able to work in a dot.variable field. Furthermore, we create different variabletypes. 
        par = self.par = SimpleNamespace() #A type for parameters 
        opt = self.opt = SimpleNamespace() #A type for optimazion results 

        # b. We define our preference parameters using the par type. 
        par.rho = 2.0
        par.nu_M = 0.001
        par.nu_F = 0.001
        par.epsilon_M = 1.0
        par.epsilon_F = 1.0
        par.omega = 0.5 

        # c. Household production 
        par.alpha = 0.5
        par.sigma = 1.0

        # d. We define wages 
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)#Creates a vector for womens wages from 0.8-1.2 with 5 equally spaced observations. 

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. We define vectors which the goal of filling them with optimezed model results later. From the beginning, they are filled with 0
        opt.LM_vec = np.zeros(par.wF_vec.size)
        opt.HM_vec = np.zeros(par.wF_vec.size)
        opt.LF_vec = np.zeros(par.wF_vec.size)
        opt.HF_vec = np.zeros(par.wF_vec.size)
        #Initialize the betas, which we want to compute. Again, these are initialized as 0
        opt.beta0 = np.nan
        opt.beta1 = np.nan
        opt.residual = np.nan

    def calc_utility(self,LM,HM,LF,HF):  
        # Attributes the simplenamespace
        par = self.par
        opt = self.opt

        # a. Consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. Home production
        with np.errstate(divide='ignore', invalid='ignore'): #We do not want divide with zero, obtain NaN etc. Therefore, we ignore these 
            if par.sigma == 1: #For the case, when sigma = 1 
                H = HM**(1-par.alpha)*HF**par.alpha 
            elif par.sigma == 0: #For the case, when sigma = 0
                H = np.minimum(HM,HF)
            else: #Otherwise this is the model. We use par. to call on the parameters set in the init function
                H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
        
        # c. Total consumption utility
        Q = C**par.omega*H**(1-par.omega)

        with np.errstate(invalid='ignore'): #Again we ignore invalid computations, while calculating the utility. 
            utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho) #We find the max from 1e-8 and above to ensure the return of a apositive number

        # d. disutlity of work
        epsilon_F_ = 1+1/par.epsilon_F
        epsilon_M_ = 1+1/par.epsilon_M
        TM = LM+HM
        TF = LF+HF
        #We then find the disutility of work:
        disutility = par.nu_M*(TM**epsilon_M_/epsilon_M_)+par.nu_F*(TF**epsilon_F_/epsilon_F_) 
        
        return utility - disutility 

    def solve_discrete(self): #This function 
        #Calling on parameters 
        par = self.par
        opt = self.opt
        
        # a. All possible choices
        x = np.linspace(0,24,49) 
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) #Creating four arrays, one for each. This represent all possible combinations

        #We then make the arrays into a singular line and overwriting them
        LM = LM.ravel()
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. We call upon the function created earlier and give it the vectors as inputs. 
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or". If the hours surpass 24. We get the notion back as True
        #If the I variable is true, then the value will be set to negative infinite. 
        #Setting it negative makes sure it will not be taking into consideration, as we only look at positive numbers
        u[I] = -np.inf 
        
    
        # d. Find maximizing argument and store solution
        j = np.argmax(u)
        
        #We store the solutions attained
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        return opt

    def solve_continous(self,do_print=False):
        """ solve model continously """
        par = self.par
        opt = self.opt

        def objective(x): #"Imports" the utility function into the continous solver
            return self.calc_utility(x[0], x[1], x[2], x[3])
        
        obj = lambda x: - objective(x) #Creating a new objective, which takes the negative of the utility function in order to maximize
        constraints = ({'type': 'ineq', 'fun': lambda x: (24 - (x[0]+x[1]) ) and (24 - (x[2]+x[3]))}) #Setting constraints in the model by defining the type and the function
        guess = [4]*4 #Our initial guess the model will start with
        bounds = [(0, 24)]*4 #Boundaries for the model 
     # d. find maximizing argument
        solution = optimize.minimize(obj,
                            guess,
                            method='Nelder-Mead',
                            bounds=bounds,
                            constraints=constraints)
        #Storing the solutions
        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]
        opt.u = self.calc_utility(opt.LM, opt.HM, opt.LF, opt.HF)

        return opt

    def solve_wF_vec(self, discrete=False, basin=False, do_print=False):

        par = self.par
        opt = self.opt

        # a. Creating vectors to store later results. These are initialized by 0
        opt.logHFHM = np.zeros(par.wF_vec.size)
        opt.HF_vec = np.zeros(par.wF_vec.size)
        opt.HM_vec = np.zeros(par.wF_vec.size)
        opt.LF_vec = np.zeros(par.wF_vec.size)
        opt.LM_vec = np.zeros(par.wF_vec.size)

        # b. loop over wF 
        for i,wF in enumerate(par.wF_vec):
            par.wF = wF
            #We call the contionous function to find the optimal level
            opt = self.solve_continous()
            
            # ii. We then store the results 
            opt.logHFHM[i] = np.log(opt.HF/opt.HM)
            opt.HM_vec[i] = opt.HM
            opt.HF_vec[i] = opt.HF
            opt.LF_vec[i] = opt.LF
            opt.LM_vec[i] = opt.LM

        return opt
    
    def run_regression(self):

        par = self.par
        opt = self.opt

        x = np.log(par.wF_vec) #Computes the logarithm of the female wage
        y = np.log(opt.HF_vec/opt.HM_vec) #Computes the logarithm of the relationship between hours for men and female 
        A = np.vstack([np.ones(x.size),x]).T #Creating a matrix and then transposing it
        opt.beta0,opt.beta1 = np.linalg.lstsq(A,y,rcond=None)[0] #We then perform a least squared regression.
    
    def estimate(self,do_print=False):
        
        par = self.par
        opt = self.opt

        # Target function
        def target(x): 
                par.alpha, par.sigma = x 
                self.solve_wF_vec() 
                self.run_regression() #Calls upon the regression 
                opt.residual = (opt.beta0-par.beta0_target)**2  + (opt.beta1-par.beta1_target)**2 #Calculating the squared the residual
                return opt.residual
            
        # Initial start guesses for Nelder-Mead Optimization
        x0=[0.5,0.1] 
        bounds = ((0,1),(0,1))
        #minimizing
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
            def dif(x):
                par = self.par #Calls upon the parameters
                opt = self.opt #Calls upon the optimal values 
                par.sigma = x[0] 
                par.epsilon_F = x[1]
                self.solve_wF_vec() #Calls upon female wage solver to solve 
                self.run_regression() #Runs the regression 
                dif = (opt.beta0 - par.beta0_target)**2 + (opt.beta1 - par.beta1_target)**2 #Finds the difference
                return dif
        
            result = optimize.minimize(dif, [sigma,epsilon_F], bounds=[(0.01,5.0),(0.01,5.0)], method='Nelder-Mead') #Minimizes sigma and epsilon_f
            opt.sigma = result.x[0] #Stores the results
            opt.epsilon_F = result.x[1]

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
        
            result = optimize.minimize(dif, [sigma], bounds=[(0.2,5.0)], method='Nelder-Mead')
            opt.sigma = result.x[0]

            return opt