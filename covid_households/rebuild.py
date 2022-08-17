import model_inputs

# new instantaneous probability:
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

# gillespie simulation dirs
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

gillespie_overwrite_dirs = [
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/corrections/experiment-08-12-19-00',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/corrections/experiment-08-12-20-38',
    '/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/corrections/experiment-08-13-13-32'
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
        raise Exception('some sizes are missing from some points in parameter space. Check `missing` object')

    if do_drop:
        # drop any rows where the parameter combinations result in beta that doesn't actually produce the right SAR
        results = recipes.Results(results.df.loc[(~model_inputs.S80_P80_SAR_Inputs.bad_combinations_crib['bad beta'])], results.metadata)
    return results