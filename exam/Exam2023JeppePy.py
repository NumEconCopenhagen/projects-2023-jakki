
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

        # Defining variables
        par.l = (((1-par.eta)*par.kappa)/par.wage)**(1/par.eta)
        par.y = par.l
        par.p = par.kappa*par.y**(-par.eta)

        #Vectors to store results
        opt.logkappa = np.zeros(par.time)
        opt.H_val =np.zeros(par.time)    

    def function1(self):

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

        for i in range(1,par.time):
            par.kappa[i] = par.rho*np.log(par.kappa[i-1])+np.random.normal(-.5*par.sigma**2,par.sigma, par.time) 
            opt.logkappa = np.log(par.kappa[i])

        return opt.logkappa
    
    def functionH(self):
        #Calls on the params
        par = self.par
        opt = self.opt

        for t in range(1, par.time):
            par.l[t] = ((1-par.eta)*par.kappa[t]/par.wage)**(1/par.eta)
            opt.H_val[t] = par.rate **(-t) * (par.kappa[t] * par.l[t]**(1-par.eta)-par.wage*par.l[t]-int(par.l[t]!= par.l[t-1])*par.iota)

            #Then taking the mean 
    



    





        