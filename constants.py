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

# For our "Bernie Sanders" parameters, we're looking at the total fraction of the trait contained in the top 20% of trait-havers. How do we convert that back to actual variance?
# If we want to do it quickly, we can use this magic lookup table as a hard-coded record of our best solutions
by_mass_axis = np.linspace(0.2, 0.9, 22)
variance_axis = np.array([0.0, 0.013, 0.051, 0.109, 0.188, 0.286, 0.403, 0.538, 0.694, 0.870, 1.07, 1.294, 1.554, 1.840, 2.172, 2.544, 2.980, 3.348, 4.070, 4.810, 5.745, 6.911])
mass_to_variance = {float(f"{x:.2f}"):y for x,y in zip(by_mass_axis, variance_axis)}

by_mass_axis2 = np.linspace(0.2, 0.8, 30)
variance_axis2 = np.array([0., 0.00524225, 0.02028586, 0.04428567, 0.07656225,
       0.11687054, 0.16455132, 0.21944574, 0.28133561, 0.35196922,
       0.42844795, 0.51222519, 0.60250828, 0.70594292, 0.81398352,
       0.93214325, 1.05698274, 1.19579864, 1.33992803, 1.4991609 ,
       1.67108227, 1.8493954 , 2.05348345, 2.27296443, 2.5050355 ,
       2.76446551, 3.04996878, 3.35501725, 3.70071365, 4.08810643])
mass_to_variance.update({float(f"{x:.2f}"):float(f"{y:.2f}") for x,y in zip(by_mass_axis2, variance_axis2)})

# We refer to variables by short names for easy typing in our code
# when we want to print them prettily in a plot, we use the following key
pretty_names = {
      'hsar'    : 'SAR',
      'inf_mass': 'infectivity in top 20%',
      'sus_mass': 'susceptibility in top 20%'
}