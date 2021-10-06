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


def load_plot_keys(path):
    with open(os.path.join(path, "keys.json"), 'r') as f:
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


if __name__ == "__main__":
    #directory = "/Users/thayer/covid_households/experiments/big-region-n-fold-all-size-6-09-27-02_59-sus-bimodal/"

    directory = sys.argv[1]
    df = load_dataframe(directory, "pool_df-inf-var-0.00.parquet")
    sample_df, pool_df = split_to_sample_and_pool(df, sample_size=1000)

    plotting_keys = load_plot_keys(directory)
    print(plotting_keys)
    plotting_keys = ["hsar", "sus_var"]
    #dump_keys(directory, plotting_keys)

    logl_df = load_dataframe(directory, "pool_df-inf-var-0.00_logl_df.parquet")
    logl_df = likelihood.logl_from_data(pool_df, sample_df, plotting_keys)
    #save_df(directory, "pool_df-inf-var-0.00_logl_df.parquet", logl_df)

    figures = np.array(["logl heatmap", "infection histograms", "logl contour plot", "trait histograms"]).reshape((2,2))
    import pdb; pdb.set_trace()

    fancy_plotting.InteractiveFigure(pool_df, sample_df, logl_df, plotting_keys, figures, baseline_values=(0.3, 1.0))

