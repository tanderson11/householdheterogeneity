import numpy as np
import constants
import traits

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
    T = constants.mean_vec[constants.STATE.infectious]
    k = constants.shape_vec[constants.STATE.infectious]
    return (k/T) * ((1/(1-hsar)**(1/k))-1) # household beta as determined by hsar

def solve_for_beta_implicitly_no_state_lengths(hsar, trait, N=20000):
    # simple case with one trait and constant state lengths
    trait_draws = trait.draw_from_distribution(np.full((N,), True, dtype='bool'))
    T = constants.mean_vec[constants.STATE.infectious]
    def g(beta):
        return np.sum(np.array([1 - np.exp(-1 * beta * T * f) for f in trait_draws])) - N * hsar

    from scipy.optimize import fsolve
    beta = fsolve(g, 0.03)
    return beta

def sample_hsar_no_state_lengths(beta, trait, N=20000):
    trait_draws = trait.draw_from_distribution(np.full((N,), True, dtype='bool'))
    T = constants.mean_vec[constants.STATE.infectious]

    hsar_draws = np.array([1 - np.exp(-1 * beta * f * T) for f in trait_draws])
    hsar = np.average(hsar_draws)
    return hsar

def sample_hsar_with_state_lengths(beta, sus, inf, N=20000):
    sus_draws = sus.draw_from_distribution(np.full((N,), True, dtype='bool'))
    inf_draws = inf.draw_from_distribution(np.full((N,), True, dtype='bool'))
    #T = constants.mean_vec[constants.STATE.infectious]

    from torch_forward_simulation import torch_state_length_sampler
    length_draws = np.array(torch_state_length_sampler(constants.STATE.infectious, np.full((N,), True, dtype='bool')).cpu()) * constants.delta_t

    hsar_draws = np.array([1 - np.exp(-1 * beta * s * f * T) for s,f,T in zip(sus_draws, inf_draws, length_draws)])
    hsar = np.average(hsar_draws)
    return hsar

def implicit_solve_for_beta(hsar, sus, inf, N=20000):
    sus_draws = sus.draw_from_distribution(np.full((N,), True, dtype='bool'))
    inf_draws = inf.draw_from_distribution(np.full((N,), True, dtype='bool'))

    from torch_forward_simulation import torch_state_length_sampler
    state_length_draws = np.array(torch_state_length_sampler(constants.STATE.infectious, np.full((N,), True, dtype='bool')).cpu()) * constants.delta_t
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

if __name__ == '__main__':
    import traits
    s = traits.GammaTrait(mean=1.0, variance=1.0)
    f = traits.ConstantTrait()
#solve_for_beta_implicitly_no_state_lengths(0.80, t)
#f = traits.GammaTrait(mean=1.0, variance=1.0)
#s = traits.ConstantTrait()
#implicit_solve_for_beta(0.80, s, f)