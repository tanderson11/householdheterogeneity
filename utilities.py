import numpy as np
import constants

### Initial infection seeding utility functions

def seed_one_by_susceptibility(size, count, susceptibility):
    initial_state = np.zeros((count, size, 1), dtype='int32') * constants.SUSCEPTIBLE_STATE
    
    sus_p = [np.squeeze(susceptibility[i,:,:]) for i in range(count)]
    choices = [np.random.choice(size, 1, p=sus/np.sum(sus)) for sus in sus_p]
    
    choices = np.array(choices).reshape(count)
    #print("CHOICES", choices)
    
    initial_state[np.arange(count), choices, 0] = constants.EXPOSED_STATE
    return initial_state
seed_one_by_susceptibility.name="seed_one"

def seed_zero(size, count, susceptibility):
    initial_state = np.zeros((count,size,1), dtype='int32')
    return initial_state
seed_zero.name="seed_zero"

### Graphing utility functions

def bar_chart_new(df, key=["model"], grouping=["size"], title_prefix="", **kwargs): 
    grouped = df.groupby(key+grouping)

    # count the occcurences and apply a sort so the shape of the dataframe doesn't change contextually
    counts = grouped.apply(lambda g: g.value_counts(subset="infections", normalize=True).sort_index())

    
    # for some reason a change in the dataframes mean that the counts were stacking the infections in this weird way that I don't understand. this fixes it
    counted_unstacked = counts.T.unstack().unstack(level=list(range(len(key))))
    # and this was the old way
    #counted_unstacked = counts.unstack(level=list(range(len(key))))

    counted_unstacked.plot.bar(**kwargs)
    
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
    return 1-(1-cumulative_probability)**(1/duration) # converting risk over study period to daily risk

def household_beta_from_hsar(hsar):
    # gamma distributed state lengths with shape k and period length T
    T = constants.mean_vec[constants.INFECTIOUS_STATE]
    k = constants.shape_vec[constants.INFECTIOUS_STATE]
    return (k/T) * ((1/(1-hsar)**(1/k))-1) # household beta as determined by hsar

### Gamma distributed trait utility functions

def wrap_gamma(mean, variance):
    if variance == 0: # if the variance is 0 simply return the mean
        return lambda shape: np.ones(shape) * mean
    else:
        return lambda shape: np.random.gamma(mean**2/variance, scale=variance/mean, size=shape)

