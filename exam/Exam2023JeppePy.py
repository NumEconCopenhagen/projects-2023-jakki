
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt

class ModelQ2:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        opt = self.opt = SimpleNamespace()

        # b. baseline parameters 
        par.eta = .5 
        par.wage = 1   
        par.kappa = np.linspace(1,2,2)
        par.rho = .9
        par.iota = .01
        par.sigma = 0.1 
        par.rate = (1+0.01)**(1/12)

        # c.  



        