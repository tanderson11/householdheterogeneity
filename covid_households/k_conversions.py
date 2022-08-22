import fractions
from scipy.stats import nbinom
import scipy.integrate
import scipy.optimize
import numpy as np

def nbinom_from_mean_and_k(mean, k):
    p = mean / (mean + 1/k * (mean**2))
    n = k
    #print(n,p)
    return nbinom(n,p)

def p80_from_nb_and_x80(rv, x80, use_floor=True):
    if use_floor:
        p80 = 1 - scipy.integrate.quad(lambda x: rv.pmf(np.floor(x)), 0, x80)[0]
    else:
        p80 = 1 - scipy.integrate.quad(lambda x: rv.pmf(x), 0, x80)[0]
    return p80

def transmission_fraction_above_x_cutoff(rv, x, use_floor=True):
    r0 = rv.mean()
    if use_floor:
        fraction = 1 - scipy.integrate.quad(lambda x: rv.pmf((x)) * x, 0, x)[0] / r0
    else:
        fraction = 1 - scipy.integrate.quad(lambda x: rv.pmf(np.floor(x)) * np.floor(x), 0, x)[0] / r0
    
    return np.abs(fraction)

def find_p80_from_mean_and_dispersion(mean, dispersion, use_floor=True):
    rv = nbinom_from_mean_and_k(mean, dispersion)
    def transmission_objective_function(x):
        return transmission_fraction_above_x_cutoff(rv, x, use_floor=use_floor) - 0.8
    x80 = scipy.optimize.least_squares(transmission_objective_function, np.array((1.5)), bounds=((1.0), (1.0e5))).x
    print(x80)
    return p80_from_nb_and_x80(rv, x80, use_floor=use_floor)