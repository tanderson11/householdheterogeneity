# Constants used to refer to states
import copy
from dataclasses import dataclass
import enum

class STATE(enum.IntEnum):
      susceptible = 0
      exposed = 1
      infectious = 2
      removed = 3

# Model parameter values
delta_t = 0.1

@dataclass
class Constants:
      delta_t: float

      infectious_period_duration_mean: float
      infectious_period_duration_std: float
      latent_period_duration_mean: float
      latent_period_duration_std: float

      def as_dict(self):
            return copy.deepcopy(self.__dict__)

### Constants used for Anjalika's evictions paper
#Latent period, days (1 day less than incubation period, to include presymptomatic transmission)
LatentPeriod=4
#Duration of mild infections, days (Equal to infectious period)
DurMildInf=7

#Latent period, days (1 day less than incubation period, to include presymptomatic transmission)
std_LatentPeriod=4
#Duration of mild infections, days
std_DurMildInf=4

evictions_paper_constants = Constants(
      delta_t,
      DurMildInf,
      std_DurMildInf,
      LatentPeriod,
      std_LatentPeriod
)

### Updated constants
updated_constants = Constants(
      delta_t,
      infectious_period_duration_mean=6.,
      infectious_period_duration_std=2.5,
      latent_period_duration_mean=3.5,
      latent_period_duration_std=2.5,
)