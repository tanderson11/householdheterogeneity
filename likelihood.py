import numpy as np

def compute_frequencies(comparison, grouping):
    frequency_total = comparison["model"].count()
    frequencies_grouped=comparison.groupby(grouping)
    frequencies = frequencies_grouped["model"].count() / frequency_total

    return frequencies

def log_likelihood_with_freqs(grouping, observed, frequencies):
    grouped = observed.groupby(grouping)
    counts = grouped["model"].count()
    frequencies = frequencies.reindex(counts.index, fill_value=0.0)

    #import pdb; pdb.set_trace()

    log_probabilities = (counts * np.log(frequencies)).sum(level=0) # verify that this is the sufficient expression of what's going on

    return (log_probabilities).sum()

def log_likelihood(grouping, observed, comparison):
    frequencies = compute_frequencies(comparison, grouping)

    return log_likelihood_with_freqs(grouping, observed, frequencies)

