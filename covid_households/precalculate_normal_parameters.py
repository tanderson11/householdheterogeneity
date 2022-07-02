from multiprocessing import Pool
import tqdm


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

    s80_axis = np.linspace(0.10, 0.80, 36)
    p80_axis = np.linspace(0.10, 0.80, 36)
    sar_axis = np.linspace(0.05, 0.60, 56)
    #s80_axis = np.linspace(0.2, 0.8, 3)
    #p80_axis = np.linspace(0.2, 0.8, 3)
    #sar_axis = np.linspace(0.15, 0.35, 3)
    from typing import OrderedDict
    axes_by_key = OrderedDict({'s80':s80_axis, 'p80':p80_axis, 'SAR':sar_axis})

    region = recipes.SimulationRegion(axes_by_key, utilities.S80_P80_SAR_Inputs)
    parameters = calculate_region_parameters(region)