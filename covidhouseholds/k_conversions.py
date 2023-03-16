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
    #print(k, rate)
    # why do we multiply the rate by tau (dividing the scale by tau) only to then multiply the scale by tau?
    # (1) to make the analogy to Toth's work explicit
    # (2) because our p80 is defined relative to nu, the individual reproductive nubmer
    #     and nu_i = beta * phi_i * tau_i * n (see supplement)
    # in other words: we bucket the period into our notion of individual reproductive number
    gamma_dist = scipy.stats.gamma(k, scale=tau/rate)
    return gamma_dist

# Indonesia study: R0 = 6.79, k = 0.06

def gamma_dist_from_k_and_r0(k, r0):
    return scipy.stats.gamma(k, scale=r0/k)

def xp80_for_continuous_rv(rv):
    def xp80_objective_function(x):
        """A function that takes its minimum value 0 when x = x_p80.
        
        xp80 = the infectiousness that divides the top p80-th percent from the bottom 1-p80th percent"""

        # -20% because 20% should be below the cutoff
        return np.abs(scipy.integrate.quad(lambda v: rv.pdf(v) * v, 0, x)[0]/rv.mean() - 0.2)
    
    xp80 = scipy.optimize.least_squares(xp80_objective_function, np.array((1.0)), bounds=((1.0e-5), (1.0e5))).x
    #print("xp80:", xp80)
    #print(scipy.integrate.quad(lambda v: rv.pdf(v) * v, 0, xp80)[0])
    return xp80

def p80_from_continuous_rv_and_xp80(rv, xp80):
    p80 = 1 - rv.cdf(xp80)
    assert p80 <= 0.8
    return p80

def p80_from_rv(rv):
    xp80 = xp80_for_continuous_rv(rv)
    p80 = p80_from_continuous_rv_and_xp80(rv, xp80)
    return p80

if __name__ == '__main__':
    import unittest
    import covidhouseholds.model_inputs as model_inputs
    class TestAgainstOurLognormalCalculation(unittest.TestCase):
        def test(self):
            p80_range = np.linspace(0.02, 0.74, 37)
            _ = 0.8  # dummy s80
            __ = 0.1 # dummy SAR
            for p80 in p80_range:
                infectiousness = model_inputs.S80_P80_SAR_Inputs(_, p80, __).to_normal_inputs(use_trait_crib=False)['inf']
                rv = infectiousness.distribution
                print(p80, p80_from_rv(rv)[0])
                self.assertAlmostEqual(p80, p80_from_rv(rv)[0], 5)
    #unittest.main()

    # TOTH:
    print("TOTH:", p80_from_rv(gamma_dist_from_ph_and_dh_of_beta_binomial(0.35, 0.43)))
    # TSANG:
    rv = scipy.stats.lognorm(s=1.03, scale=np.exp(0.0))
    print("TSANG lower bound:", p80_from_rv(rv))
    rv = scipy.stats.lognorm(s=2.83, scale=np.exp(0.0))
    print("TSANG lower bound:", p80_from_rv(rv))