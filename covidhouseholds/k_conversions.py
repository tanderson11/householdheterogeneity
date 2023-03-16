import scipy.stats
import scipy.integrate
import scipy.optimize
import numpy as np

def gamma_dist_from_ph_and_dh_of_beta_binomial(ph, dh, tau=1.0):
    def k_rate_objective_function(x):
        k, r = x
        """Takes a minimum value of 0 when k is properly related to ph and dh."""
        mean_residual = np.abs(ph - 1 + (r/(r+tau))**k)
        variance_residual = np.abs(ph * (1 - ph)/(dh + 1) - (r/(r + 2 * tau))**k + (r/(r+tau))**(2*k))
        return np.array([mean_residual, variance_residual])

    k, rate = scipy.optimize.least_squares(k_rate_objective_function, np.array([1.0, 1.0])).x
    print(k, rate)
    # why do we multiply the rate by tau (dividing the scale by tau) only to then multiply the scale by tau?
    # (1) to make the analogy to Toth's work explicit
    # (2) because our p80 is defined relative to nu, the individual reproductive nubmer
    #     and nu_i = beta * phi_i * tau_i * n (see supplement)
    # in other words: we bucket the period into our notion of individual reproductive number
    gamma_dist = scipy.stats.gamma(k, scale=tau/rate)
    return gamma_dist

# Indonesia study: R0 = 6.79, k = 0.06
#

def gamma_dist_from_k_and_r0(k, r0):
    return scipy.stats.gamma(k, scale=r0/k)

def xp80_for_continuous_rv(rv):
    def xp80_objective_function(x):
        """A function that takes its minimum value 0 when x = x_p80.
        
        xp80 = the infectiousness that divides the top p80-th percent from the bottom 1-p80th percent"""
        return np.abs(scipy.integrate.quad(lambda v: rv.pdf(v) * v, 0, x)[0] - 0.8)
    
    xp80 = scipy.optimize.least_squares(xp80_objective_function, np.array((1.0)), bounds=((1.0e-5), (1.0e5))).x
    print("xp80:", xp80)
    return xp80

def p80_from_continuous_rv_and_xp80(rv, xp80):
    p80 = 1 - scipy.integrate.quad(lambda x: rv.pdf(x), 0, xp80)[0]
    assert p80 <= 0.8
    return p80

def p80_from_rv(rv):
    xp80 = xp80_for_continuous_rv(rv)
    p80 = p80_from_continuous_rv_and_xp80(rv, xp80)
    return p80