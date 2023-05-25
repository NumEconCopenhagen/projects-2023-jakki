from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
import math


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
        par.G = 1
        par.omega_tilde = (1 - par.tau) * par.omega
        par.optimal_L_numerator = -par.kappa+math.sqrt(par.kappa**2+4*(par.alpha/par.nu)*par.omega_tilde**2)
        par.optimal_L_denominator = 2*par.omega_tilde
        par.optimal_L = par.optimal_L_numerator/par.optimal_L_denominator

    def calc_utility(self, L):
        par = self.par

        # b. consumption restriction
        C = par.kappa + par.omega_tilde * L

        # c. utility
        utility = np.log(C ** par.alpha * par.G ** (1 - par.alpha))

        # d. disutility
        disutility = par.nu * (L ** 2 / 2)

        return -1*(utility - disutility)  # Minimize the negative of the utility

    def maximize_utility(self):
        result = minimize(self.calc_utility, x0=0.0, method='BFGS')
        return result.x




