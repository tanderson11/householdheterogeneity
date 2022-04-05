import preparation
import fancy_plotting
import numpy as np
import recipes


test = 'test2'

if test == 'test1':
    key_ranges = {'hsar':(0.12, .41), 'sus_mass':(0.2,0.2), 'inf_mass':None, 'size':(5,5)}
    index = ['sus_mass', 'inf_mass', 'hsar', 'size', 'infections']
    desired_slice = preparation.extract_slice('/Users/thayer/covid_households/final_push/by_mass/combined_frequency_df2.parquet', key_ranges, index)
    desired_slice = desired_slice.droplevel('sus_mass')

    #figures = np.array(["logl contour plot", "infection histograms", "many confidence heatmap", "trait histograms"]).reshape((2,2))
    #figures = np.array(["probability contour plot", "infection histograms", "many confidence heatmap", "trait histograms"]).reshape((2,2))
    figures = np.array(["probability contour plot", "many confidence heatmap"]).reshape((1,2))
    plotting_keys = ["inf_mass", "hsar"]
    fancy_plotting.InteractiveFigure(
        None,
        plotting_keys,
        figures,
        frequency_df=desired_slice,
        unspoken_parameters={'sus_mass':0.2},
        simulation_sample_size=150,
        figsize=(10, 4.5))

if test == 'test2':
    figures = np.array(["probability contour plot", "many confidence heatmap", "infection histograms", "trait histograms"]).reshape((2,2))
    results = recipes.Results.load("/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8/full_results/")
    frequency_df = results.df
    #import pdb; pdb.set_trace()
    frequency_df = frequency_df[frequency_df.index.get_level_values('size') == 5]['frequency']
    frequency_df = frequency_df[frequency_df.index.get_level_values('SAR') <= 0.39]
    plotting_keys = ["p80", "SAR"]

    fancy_plotting.InteractiveFigure(
        plotting_keys,
        figures,
        frequency_df,
        unspoken_parameters={'s80':0.80},
        simulation_sample_size=150,
        figsize=(10, 4.5))

if test == 'test3':
    results = recipes.Results.load("/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8/full_results/")
    frequency_df = results.df
    frequency_df = frequency_df[frequency_df.index.get_level_values('size') == 5]['frequency']
    frequency_df = frequency_df[frequency_df.index.get_level_values('SAR') <= 0.39]
    plotting_keys = ["p80", "SAR"]

    # something like ...
    fancy_plotting.InfectionHistogram(frequency_df, point)