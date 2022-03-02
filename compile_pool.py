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
target_prefix = "./final_push/high_res_ontario/"
#target_prefix = "./final_push/slide_data/sample_size_new"
#target_prefix = "./final_push/by_mass/"
#target_prefix = "./final_push/n_fold/"
#target_directories = [f for f in os.listdir(target_prefix) if os.path.isdir(os.path.join(target_prefix, f))]
#sizes = range(2,15)
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
#size_parts = ["high-sizes", "low-sizes-minus-5", "size-5"]
#size_parts = ["high-sizes-extra", "low-sizes-extra"]
#size_parts = ["low-sizes-minus-5"]
size_parts = ["all-sizes"]
#size_parts = ["sar-part1", "sar-part2"]


dfs = []
logl_dfs = []
frequency_dfs = []

for f in sorted(os.listdir(os.path.join(target_prefix, size_parts[0]))):
    if '.parquet' not in f:
        continue

    print(f)
    dfs = []
    for size_group in size_parts:
        df = pq.read_table(os.path.join(target_prefix, size_group, f)).to_pandas()

        # old keys = ['hsar', 'inf_var', 'sus_var']
        keys = ['sus_var', 'hsar', 'inf_var']

        #do_correction = target_prefix == "./final_push/by_mass/"

        total_len = len(df['hsar'])

        inner_key = 'sus_mass'
        outer_key = 'inf_mass'
        inner_axis = np.linspace(0.2, 0.4, 21)
        inner_axis = np.array([float("{:.3f}".format(x)) for x in inner_axis])
        outer_axis = np.linspace(0.4, 0.6, 21)
        outer_axis = np.array([float("{:.3f}".format(x)) for x in outer_axis])
        #fixed_key_name = 'sus_mass'
        #fixed_key_value = 0.2
        remake_fixed_key = True
        fixed_key_name = 'hsar'
        fixed_key_value = float("{:.3f}".format(float(f[-13:-8])))
        print(fixed_key_value)
        #import pdb; pdb.set_trace()

        inner_len = len(inner_axis)
        outer_len = len(outer_axis)
        #import pdb; pdb.set_trace()
        if fixed_key_name in ['inf_mass', 'sus_mass'] or remake_fixed_key:
            df[fixed_key_name] = fixed_key_value
        if inner_key in ['inf_mass', 'sus_mass']:
            df[inner_key] = list(np.array([[x] * int(total_len / (outer_len * inner_len)) for x in inner_axis]).flatten()) * outer_len
        if outer_key in ['inf_mass', 'sus_mass']:
            df[outer_key] = np.array([[x] * int(total_len / (outer_len)) for x in outer_axis]).flatten()

        # inner = list(np.array([[x] * int(total_len / (outer_len * inner_len)) for x in inner_axis]).flatten()) * outer_len
        # outer = np.array([[x] * int(total_len / (outer_len)) for x in outer_axis]).flatten()
        #import pdb; pdb.set_trace()
        df = df.drop(['sus_var', 'inf_var'], axis=1)
        dfs.append(df)
    df = pd.concat(dfs)
    frequency_df = likelihood.frequencies_from_synthetic(df, [key1, key2, key3])

    frequency_dfs.append(frequency_df)

full_frequency_df = pd.concat(frequency_dfs)

full_frequency_table = pa.Table.from_pandas(pd.DataFrame(full_frequency_df))

pq.write_table(full_frequency_table, os.path.join(target_prefix, 'frequency_df.parquet'))