import numpy as np
import pandas as pd

def counts_from_empirical(empirical, keys, baseline_only_keys=["trialnum"], household_keys=["size", "infections"]):
    counts = empirical.groupby(keys + baseline_only_keys + household_keys)["model"].count()
    counts = counts.reindex(counts.index.rename(["baseline " + key for key in keys] + baseline_only_keys + household_keys))
    return counts

def frequencies_from_synthetic(synthetic, keys, household_keys=["size", "infections"]):
    comparison_grouped = comparison_df.groupby(keys) # groups by the two axes of the plots
    frequencies = comparison_grouped.apply(lambda g: compute_frequencies(g, household_keys)) # precompute the frequencies at each point because this is the expensive step
    return frequencies

def logl(synthetic, empirical, keys, **kwargs):
    frequencies = frequencies_from_synthetic(synthetic, keys, household_keys)
    counts = counts_from_empirical(empirical, keys, **kwargs)

    if isinstance(frequencies, pd.core.series.Series): # if there are multiple sizes, the result comes out in a series, if not it comes out in a dataframe that needs the size to be stacked back in as a column
        new_freqs = np.log(frequencies)
    else:
        new_freqs = np.log(frequencies).stack(level=[0,1])

    counts.name = "count"
    new_freqs.name = "freq"

    # we make a table of the log frequency values, and then we just have to 'line it up' with the counts and do arithmetic, most of the work is in lining it up
    merged = pd.merge(new_freqs.reset_index(), counts.reset_index(), on=household_keys)
    indexed_merge = merged.set_index(["baseline " + key for key in keys] + baseline_only_keys + keys + household_keys)
    indexed_merge["logl"] = indexed_merge["freq"] * indexed_merge["count"]

    full_logl_df = indexed_merge.groupby(["baseline " + key for key in keys] + baseline_only_keys + keys)["logl"].sum() # summing over household keys (because they are excluded)
    return full_logl_df

def logl_new(baseline_df, comparison_df, keys, baseline_only_keys=["trialnum"], household_keys=["size", "infections"]):
    comparison_grouped = comparison_df.groupby(keys) # groups by the two axes of the plots
    frequencies = comparison_grouped.apply(lambda g: compute_frequencies(g, household_keys)) # precompute the frequencies at each point because this is the expensive step
    print("FREQUENCIES\n", frequencies)
    #  --- NEW METHOD ---
    counts = baseline_df.groupby(keys + baseline_only_keys + household_keys)["model"].count()
    counts = counts.reindex(counts.index.rename(["baseline " + key for key in keys] + baseline_only_keys + household_keys))

    if isinstance(frequencies, pd.core.series.Series): # if there are multiple sizes, the result comes out in a series, if not it comes out in a dataframe that needs the size to be stacked back in as a column
        new_freqs = np.log(frequencies)
    else:
        new_freqs = np.log(frequencies).stack(level=[0,1])

    counts.name = "count"
    new_freqs.name = "freq"

    # we make a table of the log frequency values, and then we just have to 'line it up' with the counts and do arithmetic, most of the work is in lining it up
    merged = pd.merge(new_freqs.reset_index(), counts.reset_index(), on=household_keys)
    indexed_merge = merged.set_index(["baseline " + key for key in keys] + baseline_only_keys + keys + household_keys)
    indexed_merge["logl"] = indexed_merge["freq"] * indexed_merge["count"]

    full_logl_df = indexed_merge.groupby(["baseline " + key for key in keys] + baseline_only_keys + keys)["logl"].sum() # summing over household keys (because they are excluded)
    return full_logl_df

def compute_frequencies(comparison, grouping):
    frequency_total = comparison["model"].count()
    frequencies_grouped=comparison.groupby(grouping)
    frequencies = frequencies_grouped["model"].count() / frequency_total

    return frequencies

