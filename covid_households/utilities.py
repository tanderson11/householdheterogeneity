import scipy.optimize
import scipy.integrate
from scipy import stats
import numpy as np
from settings import STATE
import traits

### Probability math
def normalize_logl_as_probability(logl_df):
    prob_df = np.exp(logl_df.sort_values(ascending=False)-logl_df.max())
    prob_df.name = 'probability'
    prob_df = prob_df / prob_df.sum()
    return prob_df

### Graphing utility functions
def simple_bar_chart(df, key=None, drop_level=None, **kwargs):
    print(df)

    unstacked = df.unstack(list(range(len(key))))
    print(unstacked)
    ax = unstacked.plot.bar(**kwargs)
    return ax

### Parametrization utility functions
def objective_function_crafter(p80):
    '''
    Takes in the desired p80

    Returns a function that accepts a variance and returns the difference between the actual number of secondary infections caused by p80 fraction of individuals and the expected number (80%)
    '''
    assert p80 <= 0.78, "fitting is not well behaved for p80 and/or s80 > 0.78"
    def objective_function(variance):
        sigma = np.sqrt(np.log(variance + 1))
        mu = -1/2 * np.log(variance + 1)
        rv = stats.lognorm(s=sigma, scale=np.exp(mu))

        # we want x80 with F(x80) = 1 - p80
        x80_defining_function = lambda x: np.abs(1 - p80 - rv.cdf(x))
        x80 = scipy.optimize.fsolve(x80_defining_function, 0.5)[0]

        integral, abserror = scipy.integrate.quad(lambda x: rv.pdf(x) * x, 0, x80)
        #print(variance, x80, p80, rv.cdf(x80), np.abs(0.8 - 1.0 + integral))
        # ... in other words: 20% of spread from the bottom (1 - p80)%, ---> 80% from the top (p80)%
        return np.abs(0.8 - 1.0 + integral)
    return objective_function

def least_squares_solve(p80):
    objective_function = objective_function_crafter(p80)
    variance = scipy.optimize.least_squares(objective_function, np.array((1.0)), bounds=((1.0e-6), (1.0e5)))
    return variance

def lognormal_p80_solve(p80):
    variance = least_squares_solve(p80)
    return variance

def lognormal_s80_solve(s80):
    return lognormal_p80_solve(s80)

def lognormal_calculate_generalized_period(sus, inf):
    infectious_period_distribution = lognormal_DISTS[STATE.infectious]
    mu_t, sigma_t = infectious_period_distribution.mu, infectious_period_distribution.sigma

    if isinstance(sus, traits.ConstantTrait):
        assert sus.trait_value == 1., "Unimplemented calculation of beta with constant trait where mean != 1"
        mu_s, sigma_s = 0.,0.
    else:
        assert isinstance(sus, traits.LognormalTrait)
        mu_s, sigma_s = sus.mu, sus.sigma
    if isinstance(inf, traits.ConstantTrait):
        assert inf.trait_value == 1., "Unimplemented calculation of beta with constant trait where mean != 1"
        mu_f, sigma_f = 0., 0.
    else:
        assert isinstance(inf, traits.LognormalTrait)
        mu_f, sigma_f = inf.mu, inf.sigma

    # because the product of lognormal random variables is itself lognormal, we introduce a 'generalized period'
    # a random variable that behaves like a duration of the infectious state but takes into account variable susceptibility and infectivity
    mu = mu_t + mu_s + mu_f
    sigma = np.sqrt(sigma_t**2 + sigma_s**2 + sigma_f**2)
    generalized_period_rv = stats.lognorm(s=sigma, scale=np.exp(mu)) # as specified in the `scipy` documentation
    return generalized_period_rv

def lognormal_SAR_objective_function_crafter(SAR_target, generalized_period_rv):
    """A function that builds and returns an objective function with the signature objective_function(beta) --> residual of SAR and SAR_target.

    Args:
        SAR_target (float): the average SAR in the population that is desired.
            In other words, the value such that objective_function(SAR_target) = 0.
        generalized_period_rv (traits.LognormalTrait): The generalization of duration of infection
            that encompasses the variable susceptibility and infectivity of individuals.

    Returns:
        SAR_objective_function (function): a function that returns the difference between SAR_target and measured SAR. SAR_objective_function(beta) = 0 if and only if beta (probability/time) produces a secondary attack rate = SAR_target on average.
    """
    def SAR_objective_function(beta):
        integral, abserror = scipy.integrate.quad(lambda x: generalized_period_rv.pdf(x) * np.exp(-1 * beta * x), 0, np.inf)
        sar = 1 - integral
        return np.abs(SAR_target - sar)
    return SAR_objective_function

from state_lengths import lognormal_DISTS
def beta_from_sar_and_lognormal_traits(SAR, sus, inf):
    """Solves for the appropriate value of beta (probability/time of infection) given a target SAR and the distributions of susceptibility and infectivity.

    Args:
        SAR (float): the average secondary attack rate in the population (calculated in reference to households of size 2 with individuals chosen uniformly at random)
        sus (traits.LognormalTrait): the distribution of relative susceptibility among individuals in the population (must be lognormally distributed)
        inf (traits.LognormalTrait): the distribution of relative infectivity among individuals in the population (must be lognormally distributed)

    Returns:
       beta: the probability/time of infection between and infectious and susceptible contact that produces an average secondary attack rate equal to SAR
    """
    generalized_period_rv = lognormal_calculate_generalized_period(sus, inf)

    SAR_function = lognormal_SAR_objective_function_crafter(SAR, generalized_period_rv)
    beta = scipy.optimize.least_squares(SAR_function, np.array((0.05)), bounds=((1.0e-6), (1.0e5)))
    assert(beta.success is True)
    return beta.x[0]

def residual_wrapper(point, skip_old=True):
    if skip_old and (point in S80_P80_SAR_Inputs.bad_combinations_crib.index):
        return S80_P80_SAR_Inputs.bad_combinations_crib.loc[point]['residuals']
    return calculate_residual(point)

def calculate_residual(point):
    s80, p80, sar = point
    s80 = float(f"{s80:.3f}")
    p80 = float(f"{p80:.3f}")
    sar = float(f"{sar:.3f}")
    if s80 == 0.8:
        sus_dist = traits.ConstantTrait()
    else:
        mu, sigma = S80_P80_SAR_Inputs.s80_crib.loc[s80]
        sus_dist = traits.LognormalTrait(mu, sigma)
    if p80 == 0.8:
        inf_dist = traits.ConstantTrait()
    else:
        mu, sigma = S80_P80_SAR_Inputs.p80_crib.loc[p80]
        inf_dist = traits.LognormalTrait(mu, sigma)

    beta = S80_P80_SAR_Inputs.beta_crib.loc[s80, p80, sar]
    generalized_rv = lognormal_calculate_generalized_period(sus_dist, inf_dist)
    objective_function = lognormal_SAR_objective_function_crafter(sar, generalized_rv)
    return objective_function(beta)