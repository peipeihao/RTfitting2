from lmfit import Model
import numpy as np

def quadra_func(x, alpha0, alpha1, alpha2):
    return alpha0 + alpha1 * x + alpha2 * (x ** 2)

Quadra_model = Model(quadra_func)

def pll_func(x, R0, A, alpha):
    kB = 8.61733e-5    # Boltzmann constant in [eV K-1]
    return R0 + A*((kB*x)**(2*alpha))

PLL_model = Model(pll_func) # The original PLL model for resistivity

def pll_func2(x, R0, A, alpha):
    '''
    v2 of the pll function, after adding a cutoff temperature
    '''
    kB = 8.61733e-5    # Boltzmann constant in [eV K-1]
    wN = 0.5   # Cutoff energy in eV
    Tn = wN/(kB*np.pi)
    return R0 + A*(x**(2*alpha))/(Tn**(2*alpha-1))
    
PLL_model2 = Model(pll_func2)  # The new form of PLL model for resistivity

def linear_func(x, R0, A):
    return R0 + A * x

Linear_model = Model(linear_func)
