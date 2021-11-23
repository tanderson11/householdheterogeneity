from genericpath import isfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
import json
import os
import sys

import likelihood

#empirical_df = "./empirical/BneiBrak/empirical_df.parquet"
#empirical_df = pq.read_table(empirical_df).to_pandas()

#target_prefix = "./final_push/new_pool/"
target_prefix = "./final_push/by_mass/"
#target_prefix = "./final_push/n_fold/"
#target_directories = [f for f in os.listdir(target_prefix) if os.path.isdir(os.path.join(target_prefix, f))]
sizes = range(2,15)
#import pdb; pdb.set_trace()
logl_dfs = []
key1 = "hsar"
key2 = "sus_mass"
#key2 = "sus_var"
key3 = "inf_mass"
#key3 = "inf_var"

outer_key = key1

#size_parts = ["low-sizes-low-sus", "high-sizes-low-sus"]
#size_parts = ["low-sizes-high-sus", "high-sizes-high-sus", "size-nine-high-sus"]

# For the calculations by mass
size_parts = ["high-sizes", "low-sizes-minus-5", "size-5"]
#size_parts = ["high-sizes-extra", "low-sizes-extra"]
#size_parts = ["low-sizes-minus-5"]


dfs = []
logl_dfs = []
frequency_dfs = []

for f in sorted(os.listdir(os.path.join(target_prefix, size_parts[0]))):
    if '.parquet' not in f:
        continue

    # surgery / inspection
    #if '0.28' not in f:
    #    continue

    print(f)
    dfs = []
    for size_group in size_parts:
        df = pq.read_table(os.path.join(target_prefix, size_group, f)).to_pandas()

        do_correction = target_prefix == "./final_push/by_mass/"
        axis = np.linspace(0.2, 0.9, 22)
        axis = np.array([float("{:.2f}".format(x)) for x in axis])
        total_len = len(df['sus_var'])
        axis_len = int(len(axis))
        df['sus_mass'] = list(np.array([[x] * int(total_len / (axis_len**2)) for x in axis]).flatten()) * axis_len
        df['inf_mass'] = np.array([[x] * int(total_len / (axis_len)) for x in axis]).flatten()
        df = df.drop(['sus_var', 'inf_var'], axis=1)
        #import pdb; pdb.set_trace()
        dfs.append(df)
    df = pd.concat(dfs)
    frequency_df = likelihood.frequencies_from_synthetic(df, [key1, key2, key3])

    frequency_dfs.append(frequency_df)

full_frequency_df = pd.concat(frequency_dfs)

if False and target_prefix == "./final_push/by_mass/":
    axis = np.linspace(0.2, 0.9, 22)
    axis = np.array([float("{:.2f}".format(x)) for x in axis])
    full_frequency_df = full_frequency_df.sort_index(level=['inf_var', 'sus_var', 'hsar', 'size', 'infections'])
    full_frequency_df = full_frequency_df.reset_index()
    #full_frequency_df['sus_mass'] = np.array(list(axis) * int(len(full_frequency_df['sus_var'])/len(axis)))
    total_len = len(full_frequency_df['sus_var'])
    axis_len = int(len(axis))
    full_frequency_df['sus_mass'] = list(np.array([[x] * int(total_len / (axis_len**2)) for x in axis]).flatten()) * axis_len
    full_frequency_df['inf_mass'] = np.array([[x] * int(total_len / (axis_len)) for x in axis]).flatten()
    full_frequency_df = full_frequency_df.drop(['sus_var', 'inf_var'], axis=1)
    full_frequency_df = full_frequency_df.set_index(['hsar', 'sus_mass', 'inf_mass', 'size', 'infections'])

full_frequency_table = pa.Table.from_pandas(pd.DataFrame(full_frequency_df))

pq.write_table(full_frequency_table, os.path.join(target_prefix, 'frequency_df.parquet'))