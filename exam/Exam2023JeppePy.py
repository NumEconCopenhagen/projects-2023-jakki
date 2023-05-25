import numpy as np

class SalonModel:
    def __init__(self,eta, wage, rho, iota, sigma_epsilon, rate, K, delta):
        # Create baseline parameters 
        self.eta = eta
        self.wage = wage
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon 
        self.rate =rate
        self.K = K
        self.delta = delta

    def calc_profit(self,kappa):        
        l = ((1-self.eta) * kappa/self.wage) **(1/self.eta)
        Profit = kappa * l **(1-self.eta) - self.wage *l
        return Profit
    
    def Findmaxprofit(self, kappa_values):
        Max_profits = []
        
        for kappa in kappa_values:
            Profit = self.calc_profit(kappa)
            Max_profits.append(Profit)
        
        Maxprofit_index = max(range(len(Max_profits)), key=Max_profits.__getitem__)
        Maxprofit_kappa = kappa_values[Maxprofit_index]
        
        return Maxprofit_kappa
    
    def AR_schocks(self):
        np.random.seed(0)
        shocks = np.random.normal(-0.5 * self.sigma_epsilon ** 2, self.sigma_epsilon, size=120)
        return shocks
    
    def adjust_l(self, kappa, l_prev, delta):
        l_star = ((1 - self.eta) * kappa / self.wage) ** (1 / self.eta)
        if abs(l_prev - l_star) > self.delta:
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
            l = self.adjust_l(kappa, l_prev, self.delta)
            Profit = kappa * l ** (1 - self.eta) - self.wage * l - (1 if l != l_prev else 0) * self.iota
            ex_postval += self.rate ** (-t) * Profit
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
            Profit = kappa * l ** (1 - self.eta) - self.wage * l - (1 if l != l_prev else 0) * self.iota
            ex_postval_pol += self.rate ** (-t) * Profit
            l_prev = l
            kappa_prev = kappa

        return ex_postval_pol
    
    def Compare(self):
        kappa_values = [1.0, 2.0]
        Maxprofit_kappa = self.Findmaxprofit(kappa_values)
        
        ex_postvalues = []
        ex_postvalues_pol = []
        
        for _ in range(self.K):
            shocks = self.AR_schocks()
            
            ex_post_value = self.calc_ex_postval(shocks)
            ex_postvalues.append(ex_post_value)
            
            ex_post_value_policy_adjusted = self.calc_ex_postval_pol(shocks)
            ex_postvalues_pol.append(ex_post_value_policy_adjusted)
        
        H = np.mean(ex_postvalues)
        Hpol = np.mean(ex_postvalues_pol)
        
        Improvement = Hpol - H
        
        return H, Hpol, Improvement

# Example usage
eta = 0.5
w = 1.0
rho = 0.90
iota = 0.01
sigma_epsilon = 0.10
R = (1 + 0.01) ** (1 / 12)
K = 1000

model = SalonModel(eta, w, rho, iota, sigma_epsilon, R, K)
H, H_policy_adjusted, profitability_improvement = model.compare_profitability()

print(f"The approximate ex ante expected value of the salon (H) is: {H}")
print(f"The approximate ex ante expected value of the salon (H) with policy adjustment is: {H_policy_adjusted}")

if profitability_improvement > 0:
    print("The policy adjustment improves profitability.")
else:
    print("The policy adjustment does not improve profitability.")




