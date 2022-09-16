import fancy_plotting
import numpy as np
import recipes
import os


#figures = np.array(["2D only probability contour plot", "infection histograms"]).reshape((1,2))
#figures = np.array(["2D only probability contour plot", "2D slice probability contour plot"]).reshape((1,2))
figures = np.array(["3D slice free parameter probability contour plot"]).reshape((1,1))
#figures = np.array(["2D only probability contour plot", "3D slice free parameter probability contour plot"]).reshape((1,2))

stem = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
results = recipes.Results.load(os.path.join(stem, 'new_parameters/gillespie-s80-p80-SAR/beta_corrections'))
results.find_frequencies()
frequency_df = results.df

#frequency_df = frequency_df[(frequency_df.index.get_level_values('size') == 4) | (frequency_df.index.get_level_values('size') == 8)]
frequency_df = frequency_df['frequency']
#plotting_keys = ["p80", "SAR"]
plotting_keys = ["s80", "SAR"]

fancy_plotting.InteractiveFigure(
    plotting_keys,
    figures,
    frequency_df,
    unspoken_parameters={'p80':0.8},
    simulation_population={3: 871, 4: 248, 5: 198, 6: 22, 7: 19, 8: 17}, #POP=5000
    simulation_trials=1,
    figsize=(6, 4.5))