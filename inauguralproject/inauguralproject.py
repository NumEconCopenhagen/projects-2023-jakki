from types import SimpleNamespace

class householdClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        par.rho = 2
        par.nu = 0.001
        par.epsilon = 1
        par.omega = 0.5
        par.alpha = 0.5
        par.sigma = 1
        par.wageF = 1
        par.wageM = 1