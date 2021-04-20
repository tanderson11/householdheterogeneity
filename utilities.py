import numpy as np
### Initial infection seeding utility functions

def seed_one_by_susceptibility(size, count, susceptibility):
    initial_state = np.zeros((count, size, 1), dtype='int32') * SUSCEPTIBLE_STATE
    
    sus_p = [np.squeeze(susceptibility[i,:,:]) for i in range(count)]
    choices = [np.random.choice(size, 1, p=sus/np.sum(sus)) for sus in sus_p]
    
    choices = np.array(choices).reshape(count)
    #print("CHOICES", choices)
    
    initial_state[np.arange(count), choices, 0] = EXPOSED_STATE
    return initial_state

def seed_zero(size, count, susceptibility):
    initial_state = np.zeros((count,size,1), dtype='int32')
    return initial_state

### Graphing utility functions

def make_bar_chart(df, axes=False, title_prefix=""):
    grouped = df.groupby(["size", "infections"])
    regrouped = grouped["model"].value_counts().unstack().groupby("size")
    #regrouped.plot.bar(figsize=(8,8), ylabel="count")
    
    i = 0 
    for k,g in regrouped:
        try:
            if axes.any():
                g.plot.bar(ax=axes[i], ylabel="count", title="{0}Distribution of # of infections in household for households of size {1}".format(title_prefix,k))
        except:
            g.plot.bar(figsize=(8,8), ylabel="count", title="{0}Distribution of # of infections in household for households of size {1}".format(title_prefix,k))
        i += 1

