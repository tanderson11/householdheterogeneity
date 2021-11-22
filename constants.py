# Constants used to refer to states
import numpy as np
import enum

class STATE(enum.IntEnum):
      susceptible = 0
      exposed = 1
      infectious = 2
      removed = 3

# Model parameter values
delta_t = 0.1

### Means
LatentPeriod=4  #Latent period, days (1 day less than incubation period, to include presymptomatic transmission)
DurMildInf=7 #Duration of mild infections, days (Equal to infectious period)

std_LatentPeriod=4  #Latent period, days (1 day less than incubation period, to include presymptomatic transmission)
std_DurMildInf=4 #Duration of mild infections, days

# Get gamma distribution parameters

mean_vec = np.array(
      [1., LatentPeriod, DurMildInf, 1.])
### Standard deviations (not used if exponential waiting times)
std_vec=np.array(
      [1., std_LatentPeriod, std_DurMildInf, 1.])
shape_vec=(mean_vec/std_vec)**2# This will contain shape values for each state
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

# For our "Bernie Sanders" parameters, we're looking at the total fraction of the trait contained in the top 20% of trait-havers. How do we convert that back to actual variance?
# If we want to do it quickly, we can use this magic lookup table as a hard-coded record of our best solutions
by_mass_axis = np.linspace(0.2, 0.9, 22)
variance_axis = np.array([0.0, 0.013, 0.051, 0.109, 0.188, 0.286, 0.403, 0.538, 0.694, 0.870, 0.107, 1.294, 1.554, 1.840, 2.172, 2.544, 2.980, 3.348, 4.070, 4.810, 5.745, 6.911])
mass_to_variance = {float(f"{x:.2f}"):y for x,y in zip(by_mass_axis, variance_axis)}