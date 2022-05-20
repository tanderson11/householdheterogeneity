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
    figures = np.array(["2D only probability contour plot", "infection histograms"]).reshape((1,2))
    figures = np.array(["2D only probability contour plot", "2D slice probability contour plot"]).reshape((1,2))
    figures = np.array(["2D only probability contour plot", "3D slice free parameter probability contour plot"]).reshape((1,2))

    if False:
        results = recipes.Results.load("/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8/full_results/")
        frequency_df = results.df
        frequency_df = frequency_df[frequency_df.index.get_level_values('SAR') <= 0.39]
    else:
        results = recipes.Results.load('/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8-tweaked-dprobability')
        results.find_frequencies()
        frequency_df = results.df
    
    #frequency_df = frequency_df[(frequency_df.index.get_level_values('size') == 4) | (frequency_df.index.get_level_values('size') == 8)]
    frequency_df = frequency_df['frequency']
    #import pdb; pdb.set_trace()
    #plotting_keys = ["p80", "SAR"]
    plotting_keys = ["p80", "SAR"]

    fancy_plotting.InteractiveFigure(
        plotting_keys,
        figures,
        frequency_df,
        unspoken_parameters={'s80':0.8},
        simulation_population={3: 2612, 4: 744, 5: 595, 6: 67, 7: 57, 8: 50},
        #simulation_population={4:2000, 8:1000},
        simulation_trials=1,
        figsize=(10, 4.5))

if test == 'test3':
    results = recipes.Results.load("/Users/thayer/covid_households/new_parameters/s80-p80-SAR-sizes-2-8/full_results/")
    frequency_df = results.df
    frequency_df = frequency_df[frequency_df.index.get_level_values('size') == 5]['frequency']
    frequency_df = frequency_df[frequency_df.index.get_level_values('SAR') <= 0.39]
    plotting_keys = ["p80", "SAR"]

    # something like ...
    fancy_plotting.InfectionHistogram(frequency_df, point)