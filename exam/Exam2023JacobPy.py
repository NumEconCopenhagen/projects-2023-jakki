from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize_scalar

class ClassQ1:

    def __init__(self):
        """ setup model """
        # a. create namespaces
        par = self.par = SimpleNamespace()

        # b. parameters
        par.alpha = 0.5
        par.kappa = 1.0
        par.nu = 1/(2*(16**2))
        par.omega = 1.0
        par.tau = 0.30
        par.G = 5
        par.omega_tilde = (1 - par.tau) * par.omega
        par.optimal_L_numerator = -par.kappa+math.sqrt(par.kappa**2+4*(par.alpha/par.nu)*par.omega_tilde**2)
        par.optimal_L_denominator = 2*par.omega_tilde
        par.optimal_L = par.optimal_L_numerator/par.optimal_L_denominator

        # c. new parameters to Q5
        par.sigma_set1 = 1.001
        par.sigma_set2 = 1.5
        par.rho_set1 = 1.001
        par.rho_set2 = 1.5
        par.epsilon = 1

    def calc_utility(self, L):
        par = self.par

        # a. consumption restriction
        C = par.kappa + par.omega_tilde * L

        # b. utility
        utility = np.log(C ** par.alpha * par.G ** (1 - par.alpha))

        # c. disutility
        disutility = par.nu * (L ** 2 / 2)

        return -1*(utility - disutility)  # Minimize the negative of the utility

    def maximize_utility(self):
        result = minimize(self.calc_utility, x0=0.0, method='BFGS')
        return result.x

    def optimal_labor_supply(self):
        par = self.par
        numerator = -par.kappa+math.sqrt(par.kappa**2+4*(par.alpha/par.nu)*par.omega_tilde**2)
        denominator = 2*par.omega_tilde
        return numerator / denominator

    def solve_utility_q5_set1(self, L):
        par = self.par

        # a. consumption restriction
        C = par.kappa + par.omega_tilde * L

        # b. utility
        utility = (((par.alpha*C**(par.sigma_set1-1/par.sigma_set1)+(1-par.alpha)*par.G**(par.sigma_set1/1-par.sigma_set1))**(par.sigma_set1-1/par.sigma_set1))**(par.sigma_set1/par.sigma_set1-1))**(1-par.rho_set1)-1/1 - par.rho_set1

        # c. disutility
        disutility = par.nu*(L**(1+par.epsilon)/1+par.epsilon)

        return utility - disutility

    

