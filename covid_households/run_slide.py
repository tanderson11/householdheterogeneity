def run_slide_hsar_vs_inf_var():
    import preparation

    key_ranges = {'hsar':(0.12, .41), 'sus_mass':(0.2,0.2), 'inf_mass':None, 'size':(5,5)}
    index = ['sus_mass', 'inf_mass', 'hsar', 'size', 'infections']
    desired_slice = preparation.extract_slice('/Users/thayer/covid_households/final_push/by_mass/combined_frequency_df2.parquet', key_ranges, index)
    desired_slice = desired_slice.droplevel('sus_mass')

    import fancy_plotting
    import numpy as np
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

def run_slide_sus_var_vs_inf_var():
    import preparation

    key_ranges = {'hsar':(0.25, 0.25), 'sus_mass':None, 'inf_mass':None, 'size':(5,5)}
    index = ['sus_mass', 'inf_mass', 'hsar', 'size', 'infections']
    desired_slice = preparation.extract_slice('/Users/thayer/covid_households/final_push/by_mass/combined_frequency_df2.parquet', key_ranges, index)
    desired_slice = desired_slice.droplevel('hsar')

    import fancy_plotting
    import numpy as np
    figures = np.array(["logl contour plot", "many confidence heatmap"]).reshape((2,2))
    plotting_keys = ["sus_mass", "inf_mass"]
    fancy_plotting.InteractiveFigure(
        None,
        plotting_keys,
        figures,
        frequency_df=desired_slice,
        unspoken_parameters={'hsar':0.25},
        simulation_sample_size=5000)

def run_slide_hsar_vs_sus_var():
    import preparation

    key_ranges = {'hsar':(0.12, .41), 'sus_mass':None, 'inf_mass':(0.2,0.2), 'size':(5,5)}
    index = ['sus_mass', 'inf_mass', 'hsar', 'size', 'infections']
    desired_slice = preparation.extract_slice('/Users/thayer/covid_households/final_push/by_mass/combined_frequency_df2.parquet', key_ranges, index)
    desired_slice = desired_slice.droplevel('inf_mass')

    import fancy_plotting
    import numpy as np
    #"logl contour plot",  
    #figures = np.array(["probability contour plot", "many confidence heatmap"]).reshape((1,2))
    figures = np.array(["infection histograms", "many confidence heatmap"]).reshape((1,2))
    plotting_keys = ["sus_mass", "hsar"]
    fancy_plotting.InteractiveFigure(
        None,
        plotting_keys,
        figures,
        frequency_df=desired_slice,
        unspoken_parameters={'inf_mass':0.2},
        simulation_sample_size=150,
        figsize=(10, 4.5))

def run_slide_histograms():
    import preparation

    key_ranges = {'hsar':None, 'sus_mass':None, 'inf_mass':(0.2,0.2), 'size':(5,5)}
    index = ['sus_mass', 'inf_mass', 'hsar', 'size', 'infections']
    desired_slice = preparation.extract_slice('/Users/thayer/covid_households/final_push/by_mass/combined_frequency_df2.parquet', key_ranges, index)
    desired_slice = desired_slice.droplevel('inf_mass')

    import fancy_plotting
    import numpy as np
    #"logl contour plot", "infection histograms", 
    figures = np.array(["many confidence heatmap", "trait histograms"]).reshape((1,2))
    plotting_keys = ["sus_mass", "hsar"]
    fancy_plotting.InteractiveFigure(
        None,
        plotting_keys,
        figures,
        frequency_df=desired_slice,
        unspoken_parameters={'inf_mass':0.2},
        simulation_sample_size=5000)

def run_slide_all_three():
    import preparation

    key_ranges = {'hsar':(0.12, .41), 'sus_mass':None, 'inf_mass':None, 'size':(5,5)}
    index = ['sus_mass', 'inf_mass', 'hsar', 'size', 'infections']
    desired_slice = preparation.extract_slice('/Users/thayer/covid_households/final_push/by_mass/combined_frequency_df2.parquet', key_ranges, index)
    #desired_slice = desired_slice.droplevel('inf_mass')

    import fancy_plotting
    import numpy as np
    #"logl contour plot", "infection histograms", 
    figures = np.array(["probability contour plot", "many confidence heatmap"]).reshape((1,2))
    plotting_keys = ["sus_mass", "hsar"]
    fancy_plotting.InteractiveFigure(
        None,
        plotting_keys,
        figures,
        frequency_df=desired_slice,
        unspoken_parameters={'inf_mass':0.4},
        simulation_sample_size=250,
        figsize=(10, 4.5))

import sys
if __name__ == '__main__':
    slide = sys.argv[1]

    if slide == 'inf-hsar':
        run_slide_hsar_vs_inf_var()
    elif slide == 'sus-hsar':
        run_slide_hsar_vs_sus_var()
    elif slide == 'two-variances':
        run_slide_sus_var_vs_inf_var()
    elif slide == 'trait-histograms':
        run_slide_histograms()
    elif slide == 'all-three':
        run_slide_all_three()
