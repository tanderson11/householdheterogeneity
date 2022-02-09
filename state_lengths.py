from settings import device
from settings import constants
from settings import STATE

import numpy as np
import torch

# Calculating the state lengths quickly with torch

#########
### GAMMA distribution
#########

# Get gamma distribution parameters

mean_vec = np.array(
      [1., constants.latent_period_duration_mean, constants.infectious_period_duration_mean, 1.])
### Standard deviations (not used if exponential waiting times)
std_vec=np.array(
      [1., constants.latent_period_duration_std, constants.infectious_period_duration_std, 1.])
shape_vec=(mean_vec/std_vec)**2 # This will contain shape values for each state
scale_vec=(std_vec**2)/mean_vec # This will contain scale values for each state
# beta is given in accordance with the line beta = delta_t/torch_scale_vec[state], so having this fraction makes sense

# some states have an infinite duration 
inf_waiting_states = [STATE.susceptible, STATE.removed]
shape_vec[inf_waiting_states] = np.inf
scale_vec[inf_waiting_states] = np.inf
mean_vec[inf_waiting_states] = np.inf

# numpy arrays
numpy_shape_vec = np.array(shape_vec)
numpy_scale_vec = np.array(scale_vec)
numpy_mean_vec = np.array(mean_vec)

numpy_stationary_states = np.array(inf_waiting_states)

# torch's gamma distributions are parametrized with concentration and rate, but the documentation confirms concentration=alpha and rate=beta
torch_shape_vec = torch.from_numpy(numpy_shape_vec).to(device) ## move to device
torch_scale_vec = torch.from_numpy(numpy_scale_vec).to(device)

gamma_DISTS = [torch.distributions.gamma.Gamma(torch.tensor([alpha]).to(device), torch.tensor([1.0]).to(device)) for alpha in torch_shape_vec]

def gamma_state_length_sampler(new_state, entrants): #state is the constant of the state people are entering. entrants is the vector of individuals entering that state
    if new_state == STATE.removed or new_state == STATE.susceptible:
        return torch.full_like(entrants, 1e10, dtype=torch.double)

    dist = gamma_DISTS[new_state]
    samples = dist.sample(entrants.shape)
    beta = constants.delta_t/torch_scale_vec[new_state]
    samples = torch.round(samples / beta)
    return torch.squeeze(1+samples) # Time must be at least 1.

#########
### LOGNORMAL distribution
#########

from traits import LognormalTrait

lognormal_DISTS = {
    STATE.infectious.value: LognormalTrait.from_natural_mean_variance(constants.infectious_period_duration_mean, constants.infectious_period_duration_std**2),
    STATE.exposed.value: LognormalTrait.from_natural_mean_variance(constants.latent_period_duration_mean, constants.latent_period_duration_std**2),
}

torch_lognormal_DISTS = {
    STATE.infectious.value: torch.distributions.log_normal.LogNormal(lognormal_DISTS[STATE.infectious.value].mu, lognormal_DISTS[STATE.infectious.value].sigma),
    STATE.exposed.value: torch.distributions.log_normal.LogNormal(lognormal_DISTS[STATE.exposed.value].mu, lognormal_DISTS[STATE.exposed.value].sigma),
}

def lognormal_state_length_sampler(new_state, entrants):
    if new_state == STATE.removed or new_state == STATE.susceptible:
        return torch.full_like(entrants, 1e10, dtype=torch.double).to(device)
    
    dist = torch_lognormal_DISTS[new_state]
    samples = dist.sample(entrants.shape)
    samples = torch.round(samples / constants.delta_t)
    return torch.squeeze(1+samples).double()

    