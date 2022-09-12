import model_inputs

# new instantaneous probability (manual integration done with dt timesteps)
tweaked_dprob_from_parts_dirs = [
    # part 1
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group1/experiment-03-30-01-25',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group1/experiment-03-31-01-57',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group1/experiment-04-01-02-25',
    # part 2
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-03-31-01-57',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-11-22-52',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-12-16-06',
]

tweaked_dprob_completed_dirs = [
    # part 1
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group1/experiment-04-04-02-27',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group1/experiment-04-05-21-13',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group1/experiment-04-13-01-29',
    # part 2
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-01-02-25',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-02-00-50',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-12-12-47',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-12-15-52',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-12-17-36',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-12-23-06',
    '/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability/group2/experiment-04-12-23-07'
]

# gillespie simulation dirs (exact forward simulation)
gillespie_from_parts_dirs = [
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-2-5/experiment-07-01-22-18',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-2-5/experiment-07-02-13-46',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-6-8/experiment-07-02-13-48',
]

gillespie_completed_dirs = [
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-2-5/experiment-07-02-16-34',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-6-8/experiment-07-02-17-25',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/expanded_extremes/experiment-07-25-15-28',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/expanded_extremes/experiment-07-25-20-43',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/expanded_extremes/experiment-07-26-15-00',
]

# applying beta corrections to above gillepsie simulations
gillespie_overwrite_dirs = [
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/corrections/experiment-08-12-19-00',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/corrections/experiment-08-12-20-38',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/corrections/experiment-08-13-13-32'
]

# high sizes using gillespie simulation
gillespie_high_size_dirs = [
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-9-12/from_parts/experiment-08-25-16-21',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-9-12/from_parts/experiment-08-26-17-35',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-9-12/from_parts/experiment-08-27-22-36',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-9-12/from_parts/experiment-08-29-19-39',
]

gillespie_high_size_completed_dirs = [
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/sizes-9-12/completed/experiment-08-30-20-32',
]

import recipes
# Load the results of many simulated trials
# Rebuild if we  want to stitch together the parts again
def rebuild(completed_dirs, from_parts_dirs, outpath, overwrite_dirs=None, filename='results', check_region=None, check_sizes=range(2,9), do_drop=False):
    r_objs = []
    for dir in from_parts_dirs:
        r = recipes.Results.load(dir, from_parts=True)
        r_objs.append(r)

    for dir in completed_dirs:
        r = recipes.Results.load(dir)
        r_objs.append(r)

    cumulative_r = None
    for r in r_objs:
        if cumulative_r is None:
            cumulative_r = r
        else:
            cumulative_r = cumulative_r.combine(r)
    
    if overwrite_dirs is not None:
        #import pdb; pdb.set_trace()
        overwrite_r = None
        for dir in overwrite_dirs:
            print(dir)
            r = recipes.Results.load(dir, from_parts=False)
            if overwrite_r is None:
                overwrite_r = r
            else:
                overwrite_r = overwrite_r.combine(r)
        #import pdb; pdb.set_trace()
        # if we want to overwite some values, we use method right to keep only the overwriting results
        cumulative_r = cumulative_r.combine(overwrite_r, method='right')
    #import pdb; pdb.set_trace()

    cumulative_r.save(outpath, filename=filename)
    results = cumulative_r

    if check_region is None:
        return results

    missing = results.check_sizes_on_axes(check_region.axes_by_name, check_sizes)
    if missing:
        raise MissingDataException(missing)

    if do_drop:
        # drop any rows where the parameter combinations result in beta that doesn't actually produce the right SAR
        results = recipes.Results(results.df.loc[(~model_inputs.S80_P80_SAR_Inputs.bad_combinations_crib['bad beta'])], results.metadata)
    return results

class MissingDataException(Exception):
    def __init__(self, missing) -> None:
        self.missing = missing
        super().__init__()

import numpy as np
s80_axis = np.linspace(0.02, 0.80, 40)
p80_axis = np.linspace(0.02, 0.80, 40)
sar_axis = np.linspace(0.01, 0.60, 60)
axes_by_key = {'s80':s80_axis, 'p80':p80_axis, 'SAR':sar_axis}
big_region = recipes.SimulationRegion(axes_by_key, model_inputs.S80_P80_SAR_Inputs)

try:
    high_size_results = rebuild(gillespie_high_size_completed_dirs, gillespie_high_size_dirs, '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/beta_corrections/high_sizes', check_region=big_region, check_sizes=range(9,14))
except MissingDataException as e:
    exception = e