import numpy as np
from scipy.optimize import minimize_scalar

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
