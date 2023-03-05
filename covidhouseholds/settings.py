# A file that contains settings for using the module
from covidhouseholds.constants import updated_constants as model_constants
from covidhouseholds.constants import STATE

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
