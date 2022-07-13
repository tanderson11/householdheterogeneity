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
]

import recipes
# Load the results of many simulated trials
# Rebuild if we  want to stitch together the parts again
def rebuild(completed_dirs, from_parts_dirs, outpath, filename='results', check_region=None):
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
    
    cumulative_r.save(outpath, filename=filename)
    results = cumulative_r

    if check_region is None:
        return results

    missing = results.check_sizes_on_axes(check_region.axes_by_name, range(2,9))
    if missing:
        raise Exception('some sizes are missing from some points in parameter space. Check `missing` object')
    else:
        return results