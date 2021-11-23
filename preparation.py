import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import sys
import numpy as np

import traits
import likelihood
import fancy_plotting

# components:
# pool_df
# sample_df [sometimes freestanding; sometimes within the pool_df]
# logl (pool_df, sample_df) --> logl_df

def load_dataframe(path, filename):
    df = pq.read_table(os.path.join(path, filename)).to_pandas()
    df = df.rename(index=lambda val: round(val, 3))
    df = df.squeeze()
    return df

#def load_empirical_dataset(path, filename):
#    df = pq.read_table(os.path.join(path, filename)).to_pandas()


def load_plot_keys(path, filename="keys.json"):
    with open(os.path.join(path, filename), 'r') as f:
        keys = json.load(f)
    
    return keys

def split_to_sample_and_pool(df, sample_size=1000):
    assert sample_size <= (df.index.max() + 1), "sample size too large"
    sample_df = df.loc[range(0, sample_size)]
    pool_df = df.loc[range(sample_size, df.index.max() + 1)]

    return sample_df, pool_df

def dump_keys(path, keys):
    with open(os.path.join(path, "keys.json"), 'w') as f:
        json.dump(keys, f)

def save_df(path, filename, df):
    parquet_df = pa.Table.from_pandas(pd.DataFrame(df))
    pq.write_table(parquet_df, os.path.join(path, filename))

def extract_slice(frequency_path, key_ranges):
    frequency_df = pq.read_table(frequency_path).to_pandas().squeeze()
    #old_index = frequency_df.index.names
    reset = frequency_df.reset_index()

    for k,target_range in key_ranges.items():
        if not isinstance(target_range, tuple):
            continue
        satisfying = (reset[k] <= target_range[1]) & (reset[k] >= target_range[0])
        reset = reset[satisfying]
    
    #reset.set_index(old_index, inplace=True)
    return reset

if __name__ == "__main__":
    baseline_values = (0.3, 1.0)
    directory = sys.argv[1]

    # GENEVA

    # INF VAR VS HSAR
    #plotting_keys = ["hsar", "inf_var"]
    #df = load_dataframe(directory, "pool_df-sus_var-0.000.parquet")
    
    # SUS VAR VS HSAR
    #plotting_keys = load_plot_keys(directory, "inner_keys.json")
    #plotting_keys = ["hsar", "sus_var"]
    #df = load_dataframe(directory, "pool_df-inf_var-0.000.parquet")

    # SUS VAR VS INF VAR
    plotting_keys = ["sus_var", "inf_var"]

    #df = load_dataframe(directory, "pool_df-hsar-0.20.parquet")
    #df = load_dataframe(directory, "pool_df-hsar-0.320.parquet")
    #hsar = 0.310
    hsar = 0.250
    df = load_dataframe(directory, f"pool_df-hsar-{hsar:.3f}.parquet")
    unspoken_parameters = {'hsar':hsar}
    print("UNIQUE INF VAR:", df['inf_var'].unique())
    print("UNIQUE SUS VAR:", df['sus_var'].unique())

    #df = load_dataframe(directory, "pool_df-hsar-0.500.parquet")
    #    
    
    sample_df, pool_df = split_to_sample_and_pool(df, sample_size=1000)

    use_by_mass_params = False
    if use_by_mass_params:
        # hardcoded axes for a minute
        axis = np.linspace(0.2, 0.9, 8)
        #import pdb; pdb.set_trace()
        #axis = np.linspace(0.2, 0.9, 22)
        axis = np.array([float("{:.2f}".format(x)) for x in axis])
        baseline_values = (0.2, 0.2)
        sample_df['sus_mass'] = np.array(list(axis) * int(len(sample_df['sus_var'])/len(axis)))
        sample_df['inf_mass'] = np.array([[x] * len(axis) for x in axis] * int(len(sample_df['sus_var'])/(len(axis)**2))).flatten()
        sample_df = sample_df.drop(['sus_var', 'inf_var'], axis=1)
        pool_df['sus_mass'] = np.array(list(axis) * int(len(pool_df['sus_var'])/len(axis)))
        pool_df['inf_mass'] = np.array([[x] * len(axis) for x in axis] * int(len(pool_df['sus_var'])/(len(axis)**2))).flatten()
        pool_df = pool_df.drop(['sus_var', 'inf_var'], axis=1)
        plotting_keys = ["sus_mass", "inf_mass"]

    print(plotting_keys)
    
    #dump_keys(directory, plotting_keys)

    #logl_df = load_dataframe(directory, "pool_df-inf-var-0.00_logl_df.parquet")
    #save_df(directory, "pool_df-inf-var-0.00_logl_df.parquet", logl_df)

    #figures = np.array(["confidence heatmap", "infection histograms", "many confidence heatmap", "trait histograms"]).reshape((2,2))
    figures = np.array(["logl contour plot", "infection histograms", "many confidence heatmap", "trait histograms"]).reshape((2,2))
    print(pool_df)
    #fancy_plotting.InteractiveFigure(pool_df, plotting_keys, figures, full_sample_df=sample_df)
    fancy_plotting.InteractiveFigure(pool_df, plotting_keys, figures, unspoken_parameters={'hsar':0.250})
