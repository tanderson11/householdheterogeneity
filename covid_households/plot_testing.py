import preparation
import fancy_plotting
import numpy as np
import recipes

#figures = np.array(["2D only probability contour plot", "infection histograms"]).reshape((1,2))
#figures = np.array(["2D only probability contour plot", "2D slice probability contour plot"]).reshape((1,2))
figures = np.array(["3D slice free parameter probability contour plot"]).reshape((1,1))
#figures = np.array(["2D only probability contour plot", "3D slice free parameter probability contour plot"]).reshape((1,2))

results = recipes.Results.load('/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/drop_problems')
#results = recipes.Results.load('/Users/thayer/covid_households/new_parameters/gillespie-s80-p80-SAR/')
results.find_frequencies()
frequency_df = results.df

#frequency_df = frequency_df[(frequency_df.index.get_level_values('size') == 4) | (frequency_df.index.get_level_values('size') == 8)]
frequency_df = frequency_df['frequency']
#import pdb; pdb.set_trace()
#plotting_keys = ["p80", "SAR"]
plotting_keys = ["s80", "SAR"]

fancy_plotting.InteractiveFigure(
    plotting_keys,
    figures,
    frequency_df,
    unspoken_parameters={'p80':0.5},
    #simulation_population={3: 174, 4: 50, 5: 40, 6: 4, 7: 4, 8: 3}, #POP=1000
    simulation_population={3: 871, 4: 248, 5: 198, 6: 22, 7: 19, 8: 17}, #POP=5000
    #simulation_population={3: 2612, 4: 744, 5: 595, 6: 67, 7: 57, 8: 50}, #POP=15000
    #simulation_population={4:2000, 8:1000},
    simulation_trials=1,
    figsize=(6, 4.5))