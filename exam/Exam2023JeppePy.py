
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt

class ModelQ2:

    def __init__(self):
        """ setup model """

        # Create namespaces
        par = self.par = SimpleNamespace()
        opt = self.opt = SimpleNamespace()
        
        # Baseline parameters 
        par.eta = .5 
        par.wage = 1   
        par.kappa = np.linspace(1,2,2)
        par.rho = .9
        par.iota = .01
        par.sigma = 0.1 
        par.rate = (1+0.01)**(1/12)
        
        #Dynamic model parameters
        par.time = 120
        par.l0 = 0
        par.kappaprev = 1
        par.K = 1000
        par.delta = 0.05
        
        # Defining variables
        par.l = (((1-par.eta)*par.kappa)/par.wage)**(1/par.eta)
        par.y = par.l
        par.p = par.kappa*par.y**(-par.eta)
        par.l_star = ((1-par.eta) * par.kappa / par.wage)**(1/par.eta)

        #Storages to overwrite later and starting vals
        opt.epsilon = np.zeros(len(par.time))
        opt.ex_postval = 0
        opt.ex_postval_pol = 0

        #Vectors for storage 
        opt.ex_postvals = [] 
        opt.ex_postvals_pol = []

    def calculate_profit(self):

        #Calls on the Params
        par = self.par
        opt = self.opt
        #Sets up the function
        Pi = par.kappa*par.l**(1-par.eta)-par.wage*par.l
        #Finding max value for Pi and l
        Profit = np.argmax(Pi) #Finds the max value of Pi given inputs 
        Profit_l = par.l[Profit] #Finding the subsequential highest value of l based on Profit
        Profit_val = Pi[Profit] #Finding the highest value of Pi. Max profit

        #Printing my findings 
        result1 = f'The maximum profit is {Profit_val}'
        result2 = f'This happens when l is {Profit_l}'
        result3 = f'When kappa is {par.kappa[Profit]}'
        #Make it return our prints
        return result1, result2, result3 
    
    def AR_shocks(self):
        #Calls on the params
        par = self.par
        opt = self.opt
        #Creating shocks 
        np.random.seed(117)
        opt.epsilon = np.random.normal(-.5*par.sigma**2, par.sigma)
        return opt.epsilon 
    
    def calc_ex_post(self):
        #Calls on the params
        par = self.par
        opt = self.opt

        for t in range(par.time):
            par.kappa = par.rho *np.log(par.kappaprev)+opt.epsilon[t]
            par.kappa = np.exp(par.kappa)
            Profit = par.kappa * par.l **(1-par.eta)-par.wage*par.l-(1 if par.l != par.l0 else 0)*par.iota
            opt.ex_postval += par.rate**(-t)*Profit
            par.l0 = par.l
            par.kappaprev = par.kappa
    
        for t in range(par.K):
            opt.epsilon = self.AR_shocks()
            opt.ex_postval = self.calc_ex_post()
            opt.ex_postvals.append(opt.ex_postval)

        H = np.mean(opt.ex_postvals)
        print_result = print(f"The approximate ex ante expected value of the salon (H) is: {H}")
        
        return print_result

    def adjustl(self): 
        #Calls on the params 
        par = self.par 
        opt = self.opt
        if abs(par.l0 - par.l_star) > par.delta:
            return par.l_star
        else: 
            return par.l0

    def calc_ex_post_policy(self):
        #Calls on the params 
        par = self.par 
        opt = self.opt

        for t in range(par.time):
            par.kappa = par.rho * np.log(par.kappaprev) + [t]
            par.kappa = np.exp(par.kappa)
            par.l = self.adjustl()
            Profit = par.kappa * par.l ** (1 - par.eta) - par.wage * par.l - (1 if par.l != par.l0 else 0) * par.iota
            opt.ex_postval_pol += par.R ** (-t) * Profit
            par.l0 = par.l
            par.kappaprev = par.kappa


        for t in range(par.K):
            opt.epsilon = self.AR_shocks()
            opt.ex_postval_pol = self.calc_ex_post_policy
            opt.ex_postval
    
    





        