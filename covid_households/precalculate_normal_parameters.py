from multiprocessing import Pool
import tqdm
import pandas as pd

# Every conversion of an $(s_{80}, p_{80}, \text{SAR})$ to a $(\text{variance}, \text{variance}, \beta)$
# requires many numerical intergrations, it is slow. So we want to calculate these conversions once
# and save the result.

# This file when executed as `__main__` calculates these conversions for every coordinate
# in a specified `region`.

def to_normal_inputs(point):
    region, key1, key2, key3, v1, v2, v3, crib_region = point
    params = {}
    params[key1] = v1
    params[key2] = v2
    params[key3] = v3

    use_beta_crib = False
    if crib_region is not None:
        if (v1 in crib_region.axes_by_name[key1] and v2 in crib_region.axes_by_name[key2] and v3 in crib_region.axes_by_name[key3]):
            use_beta_crib=True

    default_parameters = region.parameter_class(**params).to_normal_inputs(use_beta_crib=use_beta_crib)
    return default_parameters

def calculate_region_parameters(region, crib_region=None):
    axis_data = list(region.axes_by_name.items())
    key1, axis1 = axis_data[0]
    key2, axis2 = axis_data[1]
    key3, axis3 = axis_data[2]
    
    def coordinate_stream(region, key1, key2, key3, axis1, axis2, axis3, crib_region):
        for value1 in axis1:
            for value2 in axis2:
                for value3 in axis3:
                    yield (region, key1, key2, key3, value1, value2, value3, crib_region)

    with Pool(4) as p:
        total = len(axis1) * len(axis2) * len(axis3)
        parameter_values = list(tqdm.tqdm(
            p.imap(to_normal_inputs, coordinate_stream(region, key1, key2, key3, axis1, axis2, axis3, crib_region)),
            total=total
        ))
    return parameter_values

if __name__ == '__main__':
    import recipes
    import utilities
    import numpy as np
    from typing import OrderedDict


    #s80_axis = np.linspace(0.02, 0.08, 4)
    #p80_axis = np.linspace(0.02, 0.80, 40)
    s80_axis = np.linspace(0.02, 0.80, 40)
    p80_axis = np.linspace(0.02, 0.80, 40)
    sar_axis = np.linspace(0.01, 0.60, 60)
    axes_by_key = OrderedDict({'s80':s80_axis, 'p80':p80_axis, 'SAR':sar_axis})
    region = recipes.SimulationRegion(axes_by_key, model_inputs.S80_P80_SAR_Inputs)

    known_good_s80_axis = np.linspace(0.36, 0.80, 23)
    known_good_p80_axis = np.linspace(0.36, 0.80, 23)
    known_good_SAR_axis = np.linspace(0.01, 0.60, 60)
    axes_by_key = OrderedDict({'s80':known_good_s80_axis, 'p80':known_good_p80_axis, 'SAR':known_good_SAR_axis})
    known_good_region = recipes.SimulationRegion(axes_by_key, model_inputs.S80_P80_SAR_Inputs)

    parameters = calculate_region_parameters(region, crib_region=known_good_region)

    beta_rows = []
    i = 0
    for v1 in s80_axis:
        for v2 in p80_axis:
            for v3 in sar_axis:
                beta_rows.append([(float(f"{v1:.3f}")), (float(f"{v2:.3f}")), (float(f"{v3:.3f}")), parameters[i]['household_beta']])
                i += 1
    beta_frame = pd.DataFrame(beta_rows, columns=('s80','p80','SAR','beta')).set_index(['s80','p80','SAR'])
    beta_frame.to_csv('./precalculated_beta_frame.csv')