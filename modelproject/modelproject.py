
from types import SimpleNamespace

import numpy as np

class FlydendeValuta: #Creates a class to call upon in the notebook

    def __init__(self): #Initializing a init function to contain parameters
        """ initialize the model """

        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model

        # a. externally given parameters
        par.gamma = 0.36
        par.beta1 = 0.30
        par.beta2 = 0.1
        par.h = 0.5
        par.theta = 2

        # b. parameters to be chosen (here guesses)
        par.delta = 0.2 # AR(1) of demand shock
        par.omega = 0.55 # AR(1) of supply shock
        par.sigma_x = 1.35 # std. of demand shock
        par.sigma_c = 0.96 # st.d of supply shock

        # c. misc paramters
        par.simT = 10_000 # length of simulation

        # d. calculate compound paramters
        self.calc_compound_par() 

        # e. Creating vectors to store results of the simulation 
        sim.y_hat = np.zeros(par.simT)
        sim.pi_hat = np.zeros(par.simT)
        sim.er = np.zeros(par.simT)
        sim.z = np.zeros(par.simT)
        sim.x = np.zeros(par.simT)
        sim.s = np.zeros(par.simT)
        sim.c = np.zeros(par.simT)

        # f. data (numbers given in notebook)
        datamoms.std_y = 2.0392 
        datamoms.std_pi = 1.3841
        datamoms.corr_y_pi = 0.4757
        datamoms.autocorr_y = 0.3873
        datamoms.autocorr_pi = 0.2701

    def calc_compound_par(self):
        """ calculates compound parameters """

        par = self.par

        par.betahat = par.beta1+(par.h*par.beta1/par.theta)+par.h*par.beta2

    def simulate(self):
        """ simulate the full model """

        np.random.seed(1917)

        par = self.par
        sim = self.sim

        # a. Draw random shock innovations
        sim.x = np.random.normal(loc=0.0,scale=par.sigma_x,size=par.simT)
        sim.c = np.random.normal(loc=0.0,scale=par.sigma_c,size=par.simT)

        # b. period-by-period
        for t in range(par.simT):

            # i. lagged
            if t == 0:
                z_lag = 0.0
                s_lag = 0.0
                er_lag = 0.0

            else:
                z_lag = sim.z[t-1]
                s_lag = sim.s[t-1]
                er_lag = sim.er[t-1]

            # ii. AR(1) shocks
            z = sim.z[t] = par.delta*z_lag + sim.x[t]
            s = sim.s[t] = par.omega*s_lag + sim.c[t]

            # iii. output and inflation
            sim.y_hat[t] = 1/(1+par.betahat*par.gamma)*(par.beta1*er_lag - par.betahat*par.omega*s_lag - par.betahat*sim.c[t] + par.delta*z_lag + sim.x[t])
            sim.pi_hat[t] = par.gamma/(1+par.betahat*par.gamma)*(par.beta1*er_lag-par.betahat*par.omega*s_lag-par.betahat*sim.c[t]+par.delta*z_lag+sim.x[t])+par.omega*s_lag+sim.c[t]
            sim.er[t] = er_lag - (1+(par.h/par.theta))*sim.pi_hat[t]
            
    def calc_moms(self):
        """ calculate moments """

        # note: same moments as in the data

        sim = self.sim
        moms = self.moms

        moms.std_y = np.std(sim.y_hat)
        moms.std_pi = np.std(sim.pi_hat)
        moms.corr_y_pi = np.corrcoef(sim.y_hat,sim.pi_hat)[0,1] 
        moms.autocorr_y = np.corrcoef(sim.y_hat[1:],sim.y_hat[:-1])[0,1]
        moms.autocorr_pi = np.corrcoef(sim.pi_hat[1:],sim.pi_hat[:-1])[0,1]   

    def calc_diff_to_data(self,do_print=False):
        """ calculate difference to data """

        moms = self.moms
        datamoms = self.datamoms

        error = 0.0 # sum of squared differences
        for k in self.datamoms.__dict__.keys():

            diff = datamoms.__dict__[k]-moms.__dict__[k]
            error += diff**2

            if do_print: print(f'{k:12s}| data = {datamoms.__dict__[k]:.4f}, model = {moms.__dict__[k]:.4f}')

        if do_print: print(f'{error = :12.8f}')

        return error