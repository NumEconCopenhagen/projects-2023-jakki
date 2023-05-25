from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
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
    
class SalonModel:
    def __init__(self, eta, wage, rho, iota, sigma_epsilon, rate, K, delta, deltarange):
        # Create baseline parameters 
        self.eta = eta
        self.wage = wage
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon 
        self.rate = rate
        self.K = K
        self.delta = delta
        self.deltarange = deltarange

    def calc_profit(self, kappa):        
        l = ((1 - self.eta) * kappa / self.wage) ** (1 / self.eta)
        profit = kappa * l ** (1 - self.eta) - self.wage * l
        return profit
    
    def Findmaxprofit(self, kappa_values):
        max_profits = []
        maxprofit_kappa = None
        max_profit = float('-inf')
        
        for kappa in kappa_values:
            profit = self.calc_profit(kappa)
            max_profits.append(profit)
        
        return maxprofit_kappa, max_profits
    
    def AR_schocks(self):
        np.random.seed(0)
        shocks = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
        return shocks
    
    def adjust_l(self, kappa, l_prev, delta):
        l_star = ((1 - self.eta) * kappa / self.wage) ** (1 / self.eta)
        if abs(l_prev - l_star) > delta:
            return l_star
        else:
            return l_prev
    
    def calc_ex_postval(self, shocks):
        kappa_prev = 1.0
        l_prev = 0.0
        ex_postval = 0.0

        for t in range(120):
            kappa = self.rho * np.log(kappa_prev) + shocks[t]
            kappa = np.exp(kappa)
            l = ((1 - self.eta) * kappa / self.wage) ** (1 / self.eta)
            profit = kappa * l ** (1 - self.eta) - self.wage * l - (1 if l != l_prev else 0) * self.iota
            ex_postval += self.rate ** (-t) * profit
            l_prev = l
            kappa_prev = kappa

        return ex_postval
    
    def calc_ex_postval_pol(self, shocks):
        kappa_prev = 1.0
        l_prev = 0.0
        ex_postval_pol = 0.0

        for t in range(120):
            kappa = self.rho * np.log(kappa_prev) + shocks[t]
            kappa = np.exp(kappa)
            l = self.adjust_l(kappa, l_prev, self.delta)
            profit = kappa * l ** (1 - self.eta) - self.wage * l - (1 if l != l_prev else 0) * self.iota
            ex_postval_pol += self.rate ** (-t) * profit
            l_prev = l
            kappa_prev = kappa

        return ex_postval_pol
    
    def Compare(self):
        kappa_values = [1.0, 2.0]
        maxprofit_kappa, _ = self.Findmaxprofit(kappa_values)
        
        ex_postvalues = []
        ex_postvalues_pol = []
        
        for _ in range(self.K):
            shocks = self.AR_schocks()
            
            ex_post_value = self.calc_ex_postval(shocks)
            ex_postvalues.append(ex_post_value)
            
            ex_post_value_policy_adjusted = self.calc_ex_postval_pol(shocks)
            ex_postvalues_pol.append(ex_post_value_policy_adjusted)
        
        H = np.mean(ex_postvalues)
        H_pol = np.mean(ex_postvalues_pol)
        
        improvement = H_pol - H
        
        return H, H_pol, improvement
    
    def optimize_delta(self):
        def objective(delta):
            self.delta = delta
            _, H_pol, _ = self.Compare()
            return -H_pol
        
        optimal_result = minimize_scalar(objective, bounds=self.deltarange, method='bounded')
        optimal_delta = optimal_result.x
        max_Hpol = -optimal_result.fun
        
        return optimal_delta, max_Hpol

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