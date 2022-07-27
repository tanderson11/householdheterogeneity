from multiprocessing import Pool
import tqdm
import pandas as pd

### Since every conversion of an $(s_{80}, p_{80}, \text{SAR})$ coordinate to a $(\text{variance}, \text{variance}, \beta)$ coordinate
### requires at many numerical intergrations, it is slow. So we want to calculate these conversions just once and save the results.

### This file when executed as `__main__` calculates these conversions for every coordinate in a specified `region`.

def to_normal_inputs(point):
    region, key1, key2, key3, v1, v2, v3 = point
    params = {}
    params[key1] = v1
    params[key2] = v2
    params[key3] = v3

    default_parameters = region.parameter_class(**params).to_normal_inputs()
    return default_parameters

def calculate_region_parameters(region):
    axis_data = list(region.axes_by_name.items())
    key1, axis1 = axis_data[0]
    key2, axis2 = axis_data[1]
    key3, axis3 = axis_data[2]
    
    def coordinate_stream(region, key1, key2, key3, axis1, axis2, axis3):
        for v1 in axis1:
            for v2 in axis2:
                for v3 in axis3:
                    yield (region, key1, key2, key3, v1, v2, v3)

    with Pool(4) as p:
        total = len(axis1) * len(axis2) * len(axis3)
        parameter_values = list(tqdm.tqdm(
            p.imap(to_normal_inputs, coordinate_stream(region, key1, key2, key3, axis1, axis2, axis3)),
            total=total
        ))
    return parameter_values

if __name__ == '__main__':
    import recipes
    import utilities
    import numpy as np

    #s80_axis = np.linspace(0.02, 0.08, 4)
    #p80_axis = np.linspace(0.02, 0.80, 40)
    s80_axis = np.linspace(0.10, 0.80, 36)
    p80_axis = np.linspace(0.02, 0.08, 4)
    sar_axis = np.linspace(0.01, 0.60, 60)

    #sar_axis = np.linspace(0.05, 0.60, 56)
    #s80_axis = np.linspace(0.2, 0.8, 3)
    #p80_axis = np.linspace(0.2, 0.8, 3)
    #sar_axis = np.linspace(0.15, 0.35, 3)
    from typing import OrderedDict
    axes_by_key = OrderedDict({'s80':s80_axis, 'p80':p80_axis, 'SAR':sar_axis})

    region = recipes.SimulationRegion(axes_by_key, utilities.S80_P80_SAR_Inputs)
    parameters = calculate_region_parameters(region)

    beta_rows = []
    i = 0
    for v1 in s80_axis:
        for v2 in p80_axis:
            for v3 in sar_axis:
                beta_rows.append([(np.float(f"{v1:.3f}")), (np.float(f"{v2:.3f}")), (np.float(f"{v3:.3f}")), parameters[i]['household_beta']])
                i += 1
    beta_frame = pd.DataFrame(beta_rows, columns=('s80','p80','SAR','beta')).set_index(['s80','p80','SAR'])