import numpy as np
from settings import STATE
import traits
import pandas as pd
import abc
from pathlib import Path

### Probability math
def normalize_logl_as_probability(logl_df):
    prob_df = np.exp(logl_df.sort_values(ascending=False)-logl_df.max())
    prob_df.name = 'probability'
    prob_df = prob_df / prob_df.sum()
    return prob_df

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
import scipy.optimize
import scipy.integrate
import scipy.stats as stats

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
        # ... in other words: 20% of spread from the bottom (1 - p80)%, 80% from the top p80%
        return np.abs(0.8 - 1.0 + integral)
    return objective_function

def least_squares_solve(p80):
    objective_function = objective_function_crafter(p80)
    variance = scipy.optimize.least_squares(objective_function, np.array((1.0)), bounds=((1.0e-6), (np.inf)))
    return variance

def lognormal_p80_solve(p80):
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
    beta = scipy.optimize.least_squares(SAR_function, np.array((0.05)), bounds=((1.0e-6), (0.999999)))
    assert(beta.success is True)
    return beta.x[0]

class ModelInputs(abc.ABC):
    @abc.abstractmethod
    def to_normal_inputs(self):
        '''
        Returns a dictionary of the form:

        {
            'household_beta': float,
            'sus': Trait,
            'inf': Trait,
        }
        '''
        pass

class Lognormal_Variance_Variance_Beta_Inputs(ModelInputs):
    def __init__(self, sus_variance, inf_variance, household_beta) -> None:
        self.sus_varaince = sus_variance
        self.inf_variance = inf_variance
        self.household_beta = household_beta

    def to_normal_inputs(self, use_crib=False):
        return {
            'household_beta': self.household_beta,
            'sus': traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=self.sus_varaince),
            'inf': traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=self.inf_variance),
        }

class S80_P80_SAR_Inputs(ModelInputs):
    key_names = ['s80', 'p80', 'SAR']
    s80_crib  = pd.read_csv(Path(__file__).parent / "../data/s80_lookup.csv").set_index('s80')
    p80_crib  = pd.read_csv(Path(__file__).parent / "../data/p80_lookup.csv").set_index('p80')
    beta_crib = pd.read_csv(Path(__file__).parent / "../data/beta_lookup.csv").set_index(['s80', 'p80', 'SAR'])

    def __init__(self, s80, p80, SAR) -> None:
        self.s80 = s80
        self.p80 = p80
        self.SAR = SAR

    def to_normal_inputs(self, use_crib=False):
        if self.s80 == 0.8:
            sus_dist = traits.ConstantTrait()
        else:
            if use_crib:
                sus_dist = traits.LognormalTrait(*tuple(self.s80_crib.loc[(np.float(f"{self.s80:.3f}"))]))
            else:
                sus_variance = lognormal_s80_solve(self.s80)
                assert(sus_variance.success is True)
                sus_variance = sus_variance.x[0]
                sus_dist = traits.LognormalTrait.from_natural_mean_variance(1., sus_variance)

        if self.p80 == 0.8:
            inf_dist = traits.ConstantTrait()
        else:
            if use_crib:
                inf_dist = traits.LognormalTrait(*tuple(self.p80_crib.loc[(np.float(f"{self.p80:.3f}"))]))
            else:
                inf_variance = lognormal_p80_solve(self.p80)
                assert(inf_variance.success is True)
                inf_variance = inf_variance.x[0]
                inf_dist = traits.LognormalTrait.from_natural_mean_variance(1., inf_variance)
        
        if use_crib:
            beta = float(self.beta_crib.loc[((np.float(f"{self.s80:.3f}")), (np.float(f"{self.p80:.3f}")), (np.float(f"{self.SAR:.3f}")))])
        else:
            beta = beta_from_sar_and_lognormal_traits(self.SAR, sus_dist, inf_dist)

        return {
            'household_beta': beta,
            'sus': sus_dist,
            'inf': inf_dist,
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

parameterization_by_keys = {}
parameterization_by_keys[frozenset(S80_P80_SAR_Inputs.key_names)] = S80_P80_SAR_Inputs