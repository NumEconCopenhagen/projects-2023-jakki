from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
from scipy import optimize

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

    def optimal_labor_supply(self):
        par = self.par
        numerator = -par.kappa+math.sqrt(par.kappa**2+4*(par.alpha/par.nu)*par.omega_tilde**2)
        denominator = 2*par.omega_tilde
        return numerator / denominator

class GriewankOptimizer:
    def __init__(self, bounds, tol, warmup_iter, max_iter):
        self.bounds = bounds
        self.tol = tol
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter

    def griewank(self, x):
        return self.griewank_(x[0], x[1])

    def griewank_(self, x1, x2):
        A = x1**2 / 4000 + x2**2 / 4000
        B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
        return A - B + 1

    def refined_global_optimizer(self):
        x_best = None
        x0_values = []  # Store x0 values at each iteration

        np.random.seed(2000)

        for k in range(self.max_iter):
            x = np.random.uniform(self.bounds[0], self.bounds[1], size=2)
            x0 = np.zeros_like(x)  # Initialize x0 with zeros

            if k >= self.warmup_iter:
                chi = 0.5 * (2 / (1 + np.exp((k - self.warmup_iter) / 100)))
                x0 = chi * x + (1 - chi) * x_best
                res = minimize(self.griewank, x0, method='BFGS', tol=self.tol)
            else:
                res = minimize(self.griewank, x, method='BFGS', tol=self.tol)

            x_best = res.x if k == 0 or res.fun < self.griewank(x_best) else x_best
            x0_values.append(x0)  # Append x0 to the list

            if self.griewank(x_best) < self.tol:
                break

            print(f"Iteration {k+1}: x0 = {x.round(4)}, x_best = {x_best.round(4)}")

        return x_best, x0_values

    def plot_results(self, x0_values):
        iterations = np.arange(1, len(x0_values) + 1)
        x0_values = np.array(x0_values)

        plt.plot(iterations, x0_values[:, 0], label='x0[0]')
        plt.plot(iterations, x0_values[:, 1], label='x0[1]')
        plt.xlabel('Iteration')
        plt.ylabel('x0')
        plt.legend()
        plt.title('Effective Initial Guesses (x0) vs. Iteration Counter')
        plt.show()

    def run_optimization(self):
        result, x0_values = self.refined_global_optimizer()
        print("Optimization Result:")
        print(f"x_best = {result.round(4)}, f(x_best) = {self.griewank(result):.8f}")
        self.plot_results(x0_values)