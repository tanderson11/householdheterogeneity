import numpy as np
from settings import constants
from settings import STATE
import traits
import abc

### Initial infection seeding utility functions

### Graphing utility functions

def simple_bar_chart(df, key=None, drop_level=None, **kwargs):
    print(df)
    #import pdb; pdb.set_trace()

    unstacked = df.unstack(list(range(len(key))))
    print(unstacked)
    ax = unstacked.plot.bar(**kwargs)
    return ax

def bar_chart_new(df, key=["model"], grouping=["size"], title_prefix="", **kwargs):
    grouped = df.groupby(key+grouping)

    # count the occcurences and apply a sort so the shape of the dataframe doesn't change contextually
    counts = grouped.apply(lambda g: g.value_counts(subset="infections", normalize=True).sort_index())
    # This technology breaks if not all # of infections in the range in fact exist in the data. TODO fix this

    # for some reason a change in the dataframes mean that the counts were stacking the infections in this weird way that I don't understand. this fixes it
    counted_unstacked = counts.T.unstack(fill_value=0.).unstack(level=list(range(len(key))))
    # and this was the old way
    #counted_unstacked = counts.unstack(level=list(range(len(key))))
    #import pdb; pdb.set_trace()
    ax = counted_unstacked.plot.bar(**kwargs)
    return ax

def make_bar_chart(df, color_by_column="model", axes=False, title_prefix=""):
    grouped = df.groupby(["size", "infections"])
    regrouped = grouped[color_by_column].value_counts().unstack().groupby("size")
    #regrouped.plot.bar(figsize=(8,8), ylabel="count")

    i = 0
    for k,g in regrouped:
        try:
            if axes.any():
                g.plot.bar(ax=axes[i], ylabel="count", title="{0}Distribution of # of infections in household for households of size {1}".format(title_prefix,k))
        except:
            g.plot.bar(figsize=(8,8), ylabel="count", title="{0}Distribution of # of infections in household for households of size {1}".format(title_prefix,k))
        i += 1

### Parametrization utility functions

def importation_rate_from_cumulative_prob(cumulative_probability, duration):
    if duration==0:
        print("WARNING: importation rate calculation received duration=0")
        return 0.
    else:
        return 1-(1-cumulative_probability)**(1/duration) # converting risk over study period to daily risk

def household_beta_from_hsar(hsar):
    # gamma distributed state lengths with shape k and period length T
    T = constants.mean_vec[STATE.infectious]
    k = constants.shape_vec[STATE.infectious]
    return (k/T) * ((1/(1-hsar)**(1/k))-1) # household beta as determined by hsar

def solve_for_beta_implicitly_no_state_lengths(hsar, trait, N=20000):
    # simple case with one trait and constant state lengths
    trait_draws = trait.draw_from_distribution(np.full((N,), True, dtype='bool'))
    T = constants.mean_vec[STATE.infectious]
    def g(beta):
        return np.sum(np.array([1 - np.exp(-1 * beta * T * f) for f in trait_draws])) - N * hsar

    from scipy.optimize import fsolve
    beta = fsolve(g, 0.03)
    return beta

def sample_hsar_no_state_lengths(beta, trait, N=20000):
    trait_draws = trait.draw_from_distribution(np.full((N,), True, dtype='bool'))
    T = constants.mean_vec[STATE.infectious]

    hsar_draws = np.array([1 - np.exp(-1 * beta * f * T) for f in trait_draws])
    hsar = np.average(hsar_draws)
    return hsar

def sample_hsar_with_state_lengths(beta, sus, inf, N=20000):
    sus_draws = sus.draw_from_distribution(np.full((N,), True, dtype='bool'))
    inf_draws = inf.draw_from_distribution(np.full((N,), True, dtype='bool'))
    #T = constants.mean_vec[STATE.infectious]

    from torch_forward_simulation import torch_state_length_sampler
    length_draws = np.array(torch_state_length_sampler(STATE.infectious, np.full((N,), True, dtype='bool')).cpu()) * constants.delta_t

    hsar_draws = np.array([1 - np.exp(-1 * beta * s * f * T) for s,f,T in zip(sus_draws, inf_draws, length_draws)])
    hsar = np.average(hsar_draws)
    return hsar

def implicit_solve_for_beta(hsar, sus, inf, N=20000):
    sus_draws = sus.draw_from_distribution(np.full((N,), True, dtype='bool'))
    inf_draws = inf.draw_from_distribution(np.full((N,), True, dtype='bool'))

    from torch_forward_simulation import torch_state_length_sampler
    state_length_draws = np.array(torch_state_length_sampler(STATE.infectious, np.full((N,), True, dtype='bool')).cpu()) * constants.delta_t
    #import pdb; pdb.set_trace()
    # a function that has a root when the populational average hsar (over N samples) is equal to the given hsar
    def g(beta):
        return np.sum(np.array([1 - np.exp(-1 * beta * T * f * s) for s,f,T in zip(sus_draws, inf_draws, state_length_draws)])) - N * hsar

    from scipy.optimize import fsolve
    beta = fsolve(g, 0.03) # 0.03 = x0
    return beta[0]

import scipy.optimize
def make_objective_function(percentile, target_fraction):
    def gamma_objective_function(x):
        '''A function that has the value 0 when the top {percentile} percent in the trait hold {target_fraction} fraction of the population-wide total of the trait.'''
        if x < 0: x = 0

        trait = traits.GammaTrait(mean=1.0, variance=x)
        sample = trait.draw_from_distribution(np.full((1000000,1), True, dtype=bool))
        fraction_held_in_top_percentile = np.sum(np.sort(sample, axis=0)[int(percentile/100*len(sample)):]/np.sum(sample))
        #print(x, fraction_held_in_top_percentile, target_fraction - fraction_held_in_top_percentile)
        return target_fraction - fraction_held_in_top_percentile
    return gamma_objective_function

def find_gamma_target_variance(percentile, target_fraction, method='fsolve'):
    func = make_objective_function(percentile, target_fraction)
    #print("Test:", [2.3], func(2.3))
    if method == 'least squares':
        root = scipy.optimize.least_squares(func, x0=0., bounds=(0., 100))
    elif method == 'fsolve':
        root = scipy.optimize.fsolve(func, x0=0.)
    return root[0]

import scipy.stats as stats
def lognormal_p80_solve(p80):
    def objective_function_crafter(p80):
        '''
        Takes in the desired p80

        Returns a function that accepts a variance
        and returns the difference between the actual number of secondary infections caused by p80 fraction of individuals and the expected number (80%)
        '''
        assert p80 <= 0.78, "fitting is not well behaved for p80 > 0.78"
        def objective_function(variance):
            sigma = np.sqrt(np.log(variance + 1))
            mu = -1/2 * np.log(variance + 1)
            rv = stats.lognorm(s=sigma, scale=np.exp(mu))

            # we want x80 with F(x80) = p80
            x80_defining_function = lambda x: np.abs(1 - p80 - rv.cdf(x))
            x80 = scipy.optimize.fsolve(x80_defining_function, 0.5)[0]


            integral, abserror = scipy.integrate.quad(lambda x: rv.pdf(x) * x, 0, x80)
            #print(variance, x80, p80, rv.cdf(x80), np.abs(0.8 - 1.0 + integral))
            return np.abs(0.8 - 1.0 + integral)
        return objective_function

    def least_squares_solve(p80):
        objective_function = objective_function_crafter(p80)
        variance = scipy.optimize.least_squares(objective_function, np.array((1.0)), bounds=((1.0e-6), (np.inf)))
        return variance

    variance = least_squares_solve(p80)
    return variance

def lognormal_s80_solve(s80):
    return lognormal_p80_solve(s80)

from state_lengths import lognormal_DISTS
def beta_from_sar_and_lognormal_traits(SAR, sus, inf):
    infectious_period_distribution = lognormal_DISTS[STATE.infectious]
    mu_t, sigma_t = infectious_period_distribution.mu, infectious_period_distribution.sigma

    if isinstance(sus, traits.ConstantTrait):
        assert sus.trait_value == 1., "Unimplemented calculation of beta with constant trait where mean != 1"
        mu_s, sigma_s = 0.,0.
    else:
        mu_s, sigma_s = sus.mu, sus.sigma
    if isinstance(inf, traits.ConstantTrait):
        assert inf.trait_value == 1., "Unimplemented calculation of beta with constant trait where mean != 1"
        mu_f, sigma_f = 0., 0.
    else:
        mu_f, sigma_f = inf.mu, inf.sigma

    mu = mu_t + mu_s + mu_f
    sigma = np.sqrt(sigma_t**2 + sigma_s**2 + sigma_f**2)
    generalized_period_rv = stats.lognorm(s=sigma, scale=np.exp(mu)) # as specified in the `scipy` documentation

    def SAR_objective_function_crafter(SAR_target):
        def SAR_objective_function(beta):
            integral, abserror = scipy.integrate.quad(lambda x: generalized_period_rv.pdf(x) * np.exp(-1 * beta * x), 0, np.inf)
            sar = 1 - integral
            return np.abs(SAR_target - sar)
        return SAR_objective_function

    SAR_function = SAR_objective_function_crafter(SAR)
    beta = scipy.optimize.least_squares(SAR_function, np.array((0.05)), bounds=((1.0e-6), (np.inf)))
    assert(beta.success is True)
    return beta.x[0]

class ModelInputs(abc.ABC):
    @abc.abstractmethod
    def to_normal_inputs(self):
        '''
        Returns a dictionary of the form:

        {
            'household_beta': float,
            'sus_dist': Trait,
            'inf_dist': Trait,
        }
        '''
        pass

class Lognormal_Variance_Variance_Beta_Inputs(ModelInputs):
    def __init__(self, sus_variance, inf_variance, household_beta) -> None:
        self.sus_varaince = sus_variance
        self.inf_variance = inf_variance
        self.household_beta = household_beta

    def to_normal_inputs(self):
        return {
            'household_beta': self.household_beta,
            'sus_dist': traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=self.sus_varaince),
            'inf_dist': traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=self.inf_variance),
        }

class S80_P80_SAR_Inputs(ModelInputs):
    def __init__(self, s80, p80, SAR) -> None:
        self.s80 = s80
        self.p80 = p80
        self.SAR = SAR

    def to_normal_inputs(self):
        if self.s80 == 0.8:
            sus_dist = traits.ConstantTrait()
        else:
            sus_variance = lognormal_s80_solve(self.s80)
            assert(sus_variance.success is True)
            sus_variance = sus_variance.x[0]
            sus_dist = traits.LognormalTrait.from_natural_mean_variance(1., sus_variance)

        if self.p80 == 0.8:
            inf_dist = traits.ConstantTrait()
        else:
            inf_variance = lognormal_p80_solve(self.p80)
            assert(inf_variance.success is True)
            inf_variance = inf_variance.x[0]
            inf_dist = traits.LognormalTrait.from_natural_mean_variance(1., inf_variance)
        
        beta = beta_from_sar_and_lognormal_traits(self.SAR, sus_dist, inf_dist)

        return {
            'household_beta': beta,
            'sus_dist': sus_dist,
            'inf_dist': inf_dist,
        }

    def as_dict(self):
        return {
            's80':self.s80,
            'p80':self.p80,
            'SAR':self.SAR,
        }

    def __repr__(self) -> str:
        rep = "ModelInputs"
        return rep + f"({self.as_dict()}\n{self.to_normal_inputs()})"