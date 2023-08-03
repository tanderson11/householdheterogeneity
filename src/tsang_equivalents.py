import scipy.optimize
import scipy.integrate
import numpy as np

import utilities
import traits

def get_distribution(p80):
    inf_variance = utilities.lognormal_p80_solve(p80)
    assert(inf_variance.success is True)
    inf_variance = inf_variance.x[0]
    inf_dist = traits.LognormalTrait.from_natural_mean_variance(1., inf_variance)
    return inf_dist.distribution

def find_ratio_of_average_to_top_20(p80):
    f = get_distribution(p80)
    objective_function_for_top_20_percent = lambda x: np.abs(0.8 - f.cdf(x))
    infectiousness_of_top_20 = scipy.optimize.least_squares(objective_function_for_top_20_percent, np.array(1.0), bounds=((1.0e-6), (1.0e5))).x
    print("infectiousness_of_top_20:", infectiousness_of_top_20)
    average_infectivity_above_threshold = scipy.integrate.quad(lambda x: f.pdf(x)*x, infectiousness_of_top_20, np.inf)/(1-f.cdf(infectiousness_of_top_20))

    return average_infectivity_above_threshold / f.mean() # f.mean should always be 1, but hey why not


def from_n_fold_to_fraction_from_top_20(n):
    return n / (n+1)

def find_p80_such_that_x_transmission_comes_from_top_20_individuals(x_transmission):
    def find_transmission_from_top_20(p80):
        f = get_distribution(p80)
        objective_function_for_top_20_percent = lambda x: np.abs(0.8 - f.cdf(x))
        infectiousness_of_top_20 = scipy.optimize.least_squares(objective_function_for_top_20_percent, np.array(1.0), bounds=((1.0e-6), (1.0e5))).x
        print("infectiousness_of_top_20:", infectiousness_of_top_20)
        from_top_20 = scipy.integrate.quad(lambda x: f.pdf(x)*x, infectiousness_of_top_20, np.inf)[0]
        print("from_top_20", from_top_20)
        return from_top_20
    
    p80 = scipy.optimize.least_squares(lambda p80: np.abs(find_transmission_from_top_20(p80) - x_transmission), np.array(0.2), bounds=((1.0e-6), (1.0e5))).x
    return p80, get_distribution(p80)
