import numpy as np

def log_likelihood(grouping, observed, comparison):
    frequency_total = comparison["model"].count()
    frequencies_grouped=comparison.groupby(grouping)
    frequencies = frequencies_grouped["model"].count() / frequency_total

    grouped = observed.groupby(grouping)
    counts = grouped["model"].count()
    frequencies = frequencies.reindex(counts.index, fill_value=0.0)

    log_probabilities = (counts * np.log(frequencies)).sum(level=0) # verify that this is the sufficient expression of what's going on
    if np.isinf(log_probabilities).any():
        pass
        #make_bar_chart(observed)
        #make_bar_chart(comparison)

    return (log_probabilities).sum()
