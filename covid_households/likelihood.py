import numpy as np
import pandas as pd
import scipy.interpolate

def interpolate_likelihood(logl_df):
    interp = scipy.interpolate.LinearNDInterpolator(points=logl_df.index.to_frame().values, values=logl_df.values)
    return interp

def confidence_mask_from_logl(logl_df, percentiles=(0.95,), **kwargs):
    normalized_probability = normalize_probability(logl_df)
    return find_confidence_mask(normalized_probability, percentiles, **kwargs)

def normalize_probability(logl_df):
    prob_space = np.exp(logl_df.sort_values(ascending=False)-logl_df.max())
    normalized_probability = prob_space/prob_space.sum()
    #print(normalized_probability.sum())
    return normalized_probability

def find_confidence_mask(normalized_probability, percentiles=(0.95,)):
    confidence_masks = []
    for p in percentiles:
        confidence_masks.append((normalized_probability.cumsum() <= p).astype('int32'))
    confidence_mask = sum(confidence_masks)
    if len(percentiles) == 1:
        return confidence_mask.astype('bool')
    return confidence_mask

def confidence_interval_from_confidence_mask(confidence_mask, key, include_endpoints=True):
    all_values = np.unique(confidence_mask.index.get_level_values(key))
    value_set = np.unique(confidence_mask[confidence_mask].index.get_level_values(key))
    min = value_set.min()
    max = value_set.max()

    if include_endpoints:
        values_above = all_values[np.where(all_values > max)]
        if len(values_above) > 0:
            max = values_above.min()

        values_below = all_values[np.where(all_values < min)]
        if len(values_below) > 0:
            min = values_below.max()

    return (min, max)

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
    if count_columns_to_prefix is not None:
        relabeled = {x:f'{sample_prefix}{x}' for x in count_columns_to_prefix}
        counts = counts.index.rename(columns=relabeled, inplace=False)

    logl = fast_logl_from_frequencies_and_counts(frequencies, counts, parameter_keys, household_keys=household_keys)
    # add back in the keys from the counts dataframe to be compatible with certain figures
    # TK
    return logl

def fast_logl_from_frequencies_and_counts(frequencies, counts, parameter_keys, household_keys=["size", "infections"]):
    if 'trial' not in counts.index.names:
        counts = pd.concat({0: counts}, names=['trial'])
    # First, verify the counts come from a single point in parameter space
    parameter_levels = counts.index.droplevel(['trial', 'size', 'infections'])
    # for other levels, check that there is only one unique value in the index
    for level_name in parameter_levels.names:
        assert len(np.unique(parameter_levels.get_level_values(level_name))) == 1, 'Fast logl calculation requires that the counts come from a single point in parameter space.'

    frequencies_g = (np.log(frequencies)).groupby(household_keys)
    counts_g = counts.groupby(household_keys)

    logl_parts = []
    # we group by size and # of infections,
    # then the observed count can be multiplied against all the frequencies (even at different parameter valeus)
    for k,g in counts_g:
        count_group = g
        count_group.index = (count_group.index.get_level_values('trial'))
        freq_group = frequencies_g.get_group(k)
        # we need both frames to have have the same name so that the pandas `dot` will work
        freq_group.name = 'x'
        count_group.name = 'x'
        # product of the count by every different frequency, for each trial (trials end up in separate columns)
        x = freq_group.to_frame().dot(count_group.to_frame().transpose())
        logl_parts.append(x)

    logl = pd.concat(logl_parts).groupby(parameter_keys).sum().stack()
    return logl