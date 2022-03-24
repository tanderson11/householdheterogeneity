# A file that contains settings for using the module
from covid_households.constants import updated_constants as model_constants
from covid_households.constants import STATE

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")