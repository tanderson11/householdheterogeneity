import recipes
import utilities
import numpy as np
import os
s80_axis = np.linspace(0.2, 0.8, 6)
p80_axis = np.linspace(0.2, 0.8, 6)
sar_axis = np.linspace(0.15, 0.35, 5)
from typing import OrderedDict
axes_by_key = OrderedDict({'s80':s80_axis, 'p80':p80_axis, 'SAR':sar_axis})
region = recipes.SimulationRegion(axes_by_key, model_inputs.S80_P80_SAR_Inputs)

import datetime
date_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
dir_name = f"experiment_outputs/experiment-{date_str}"
os.makedirs(dir_name)
x = recipes.Model()
results = x.run_grid({4:100000}, region, dir_name, use_beta_crib=True)
results.save(dir_name, 'results')