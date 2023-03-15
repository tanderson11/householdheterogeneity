from scipy.stats import nbinom
import scipy.integrate
import scipy.optimize
import numpy as np
from traits import GammaTrait

def gamma_dist_from_ph_and_dh_of_beta_binomial(ph, dh):
    gamma_dist = GammaTrait(mean=ph, variance=ph*(1-ph)/(dh+1))
    return gamma_dist

def xp80_for_continuous_rv(rv):
    def xp80_objective_function(x):
        """A function that takes its minimum value 0 when x = x_p80.
        
        xp80 = the infectiousness that divides the top p80-th percent from the bottom 1-p80th percent"""
        return np.abs(scipy.integrate.quad(lambda v: rv.pdf(v) * v, 0, x)[0] - 0.8)
    
    xp80 = scipy.optimize.least_squares(xp80_objective_function, np.array((1.0)), bounds=((1.0e-5), (1.0e5))).x
    return xp80

def p80_from_continuous_rv_and_xp80(rv, xp80):
    p80 = 1 - scipy.integrate.quad(lambda x: rv.pdf(x), 0, xp80)[0]
    return p80