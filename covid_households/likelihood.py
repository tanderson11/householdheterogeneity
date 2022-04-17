import numpy as np
import pandas as pd

def confidence_mask_from_logl(logl_df, percentiles=(0.95)):
    normalized_probability = normalize_probability(logl_df)
    return find_confidence_mask(normalized_probability, percentiles)

def normalize_probability(df):
    prob_space = np.exp(df.sort_values(ascending=False)-df.max())
    normalized_probability = prob_space/prob_space.sum()
    print(normalized_probability.sum())
    return normalized_probability

def find_confidence_mask(normalized_probability, percentiles=(0.95)):
    confidence_masks = []
    for p in percentiles:
        confidence_masks.append((normalized_probability.cumsum() < p).astype('int32'))
    confidence_mask = sum(confidence_masks)
    return confidence_mask

def counts_from_empirical(empirical, parameter_keys, sample_only_keys=["trial"], household_keys=["size", "infections"]):
    #counts = empirical.groupby(keys + sample_only_keys + household_keys)["model"].count()
    counts = empirical.groupby(parameter_keys + sample_only_keys + household_keys).size()
    #import pdb; pdb.set_trace()
    counts = counts.reindex(counts.index.rename(["sample " + key for key in parameter_keys] + sample_only_keys + household_keys))
    counts.name = "count"
    return counts

def frequencies_from_synthetic(synthetic, parameter_keys, household_keys=["size", "infections"]):
    if len(parameter_keys) > 0:
        comparison_grouped = synthetic.groupby(parameter_keys) # groups by the axes of the plot
        frequencies = comparison_grouped.apply(lambda g: compute_frequencies(g, household_keys)) # precompute the frequencies at each point because this is the expensive step
    else:
        frequencies = compute_frequencies(synthetic, household_keys)

    if not isinstance(frequencies, pd.core.series.Series): # if there are multiple sizes, the result comes out in a series, if not it comes out in a dataframe that needs the size to be stacked back in as a column
        frequencies = frequencies.stack(level=[0,1])

    return frequencies

def compute_frequencies(comparison, grouping):
    #import pdb; pdb.set_trace()
    #frequency_total = len(comparison)
    frequencies_grouped=comparison.groupby(grouping)
    frequency_totals = frequencies_grouped.size().groupby('size').sum()
    frequencies = frequencies_grouped.size() / frequency_totals
    frequencies.name = 'freqs'
    hh_size = frequencies.reset_index()['size'].iloc[0]
    repaired_parts = []
    for hh_size, g in frequencies.groupby(['size']):
        if len(g) != hh_size:
            #import pdb; pdb.set_trace()
            # add 0 in any entry missing from the index: ie, if we don't ever observe that number of infections, its frequency is 0
            z = pd.DataFrame({"size":hh_size, "infections":list(range(1,hh_size+1)), "freqs":0.})
            z = z.set_index(["size", "infections"]).squeeze()
            z.update(frequencies) # then add back in the frequencies we do know
            repaired_parts.append(z)
    # then put the repaired parts back in situ in the frequencies df
    for repaired_part in repaired_parts:
        frequencies = pd.concat([frequencies, repaired_part])
        duplicated = frequencies.index.duplicated()
        frequencies = frequencies[~duplicated]
    return frequencies

def logl_from_data(synthetic, empirical, parameter_keys, sample_only_keys=["trial"], household_keys=["size", "infections"], frequency_df=None):
    if frequency_df is not None:
        frequencies = frequency_df
    else:
        frequencies = frequencies_from_synthetic(synthetic, parameter_keys, household_keys=household_keys)
    counts = counts_from_empirical(empirical, parameter_keys, sample_only_keys=sample_only_keys, household_keys=household_keys)
    logl = logl_from_frequencies_and_counts(frequencies, counts, parameter_keys, sample_only_keys=sample_only_keys, household_keys=household_keys)
    return logl

def logl_from_frequencies_and_counts(frequencies, counts, parameter_keys, sample_prefix='sample ', sample_only_keys=["trial"], household_keys=["size", "infections"], count_columns_to_prefix=None):
    counts = counts.reset_index()
    if count_columns_to_prefix is not None:
        relabeled = {x:f'{sample_prefix}{x}' for x in count_columns_to_prefix}
        counts = counts.rename(columns=relabeled, inplace=False)
    log_freqs = np.log(frequencies)
    log_freqs.name = "log freq"

    log_freqs = log_freqs.reset_index()
    log_freqs["dummy"] = 0.0
    counts["dummy"] = 0.0
    # we make a table of the log frequency values, and then we just have to 'line it up' with the counts and do arithmetic, most of the work is in lining it up
    merged = pd.merge(counts, log_freqs, how='left', on=household_keys+["dummy"])
    #print(merged)
    indexed_merge = merged.set_index([sample_prefix + key for key in parameter_keys] + sample_only_keys + parameter_keys + household_keys)
    indexed_merge["logl"] = indexed_merge["log freq"] * indexed_merge["count"]

    if (indexed_merge.index.to_frame().isna()).any().any():
        raise ValueError("NaN in merged index suggesting that not all infection counts were present. Try running results.repair_mising_counts")

    try:
        logls = indexed_merge.groupby(["sample " + key for key in parameter_keys] + sample_only_keys + parameter_keys)["logl"]
    except ValueError:
        logls = indexed_merge['logl']

    return logls.sum() # summing over household keys (because they are excluded)