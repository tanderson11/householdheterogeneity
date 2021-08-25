import enum
from typing import NamedTuple
import numpy as np
import pandas as pd

import constants
import traits
from torch_forward_simulation import torch_forward_time, torch_state_length_sampler

class StateLengthConfig(enum.Enum):
    gamma_state_lengths = 'gamma state lengths'
    constant_state_lengths = 'constant state lengths'

class InitialSeedingConfig(enum.Enum):
    seed_one_by_susceptibility = 'seed one by susceptibility'
    no_initial_infections = 'seed none'

class ImportationRegime(NamedTuple):
    duration: int
    importation_rate: float

class Model(NamedTuple):
    # simulation parameters
    state_lengths: str = StateLengthConfig.gamma_state_lengths.value
    initial_seeding: str = InitialSeedingConfig.seed_one_by_susceptibility.value
    importation: ImportationRegime = None
    secondary_infections: bool = True # for debugging / testing

    def run_trials(self, household_beta, trials=1, population=None, sizes=None, sus=traits.ConstantTrait("sus"), inf=traits.ConstantTrait("inf")):
        #import pdb; pdb.set_trace()
        if population is None:
            assert sizes is not None
            expanded_sizes = {size:count*trials for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations
            population = Population(expanded_sizes, sus, inf)

        df = population.simulate_population(household_beta, *self)
        
        dfs = []
        grouped = df.groupby("size")
        for size, group in grouped:
            count = sizes[size]
            trialnums = [t for t in range(trials) for i in range(count)]
            group["trialnum"] = trialnums
            dfs.append(group)
        return pd.concat(dfs)

class Population:
    def __init__(self, household_sizes, susceptibility=traits.ConstantTrait("sus"), infectivity=traits.ConstantTrait("inf")):
        unpacked_sizes = [[size]*number for size, number in household_sizes.items()]
        flattened_unpacked_sizes = [x for l in unpacked_sizes for x in l]

        self.df = pd.DataFrame({"size":flattened_unpacked_sizes},
            columns = ["size"])

        total_households = len(self.df)
        max_size = self.df["size"].max()

        # the giant numpy array of individuals holds 'households' of identical size but we use is_occupied to enforce the actual strcuture
        self.is_occupied = np.array([[1. if i < hh_size else 0. for i in range(max_size)] for hh_size in self.df["size"]]) # 1. if individual exists else 0.
        self.susceptibility = np.expand_dims(susceptibility.draw_from_distribution(self.is_occupied), axis=2) # ideally we will get rid of expand dims at some point
        self.infectiousness = np.transpose(np.expand_dims(infectivity.draw_from_distribution(self.is_occupied), axis=2), axes=(0,2,1))

        # not adjacent to yourself
        nd_eyes = np.stack([np.eye(max_size,max_size) for i in range(total_households)]) # we don't worry about small households here because it comes out in a wash later
        adjmat = 1 - nd_eyes # this is used so that an individual doesn't 'infect' themself
        
        # reintroduce vaccines TK

        self.connectivity_matrix = (self.susceptibility @ self.infectiousness) * adjmat

    def seed_one_by_susceptibility(self):
        n_hh = len(self.df["size"])
        initial_state = self.is_occupied * constants.STATE.susceptible
        
        sus_p = [np.squeeze(self.susceptibility[i,:,:]) for i in range(n_hh)]
        choices = [np.random.choice(range(len(sus)), 1, p=sus/np.sum(sus)) for sus in sus_p] # susceptibility/total_sus chance of getting seed --> means this works with small households
        
        #import pdb; pdb.set_trace()
        choices = np.array(choices).reshape(n_hh)

        initial_state[np.arange(n_hh), choices] = constants.STATE.exposed
        return initial_state

    def seed_none(self):
        initial_state = self.is_occupied * constants.STATE.susceptible
        return initial_state

    def simulate_population(self, household_beta, state_lengths, initial_seeding, importation, **kwargs):
        initial_seeding = InitialSeedingConfig(initial_seeding)

        if initial_seeding == InitialSeedingConfig.seed_one_by_susceptibility:
            initial_state = self.seed_one_by_susceptibility()
        elif initial_seeding == InitialSeedingConfig.seed_none:
            initial_state = self.seed_none()
        else:
            raise Exception(f"unimplimented initial seeding name {initial_seeding}")

        initial_state = np.expand_dims(initial_state, axis=2)

        if importation is None:
            importation_probability = 0. * self.susceptibility
        else:
            if initial_state.any():
                print("WARNING: importation rate is defined while initial infections were seeded. Did you intend this?")
        
            importation_probability = importation.importation_rate * self.susceptibility

        # select the appropriate function for state lengths based on config str:
        state_length_config = StateLengthConfig(state_lengths)
        if state_length_config == StateLengthConfig.gamma_state_lengths:
            state_length_sampler = torch_state_length_sampler
        elif state_length_config == StateLengthConfig.constant_state_lengths:
            raise Exception('unimplemented constant state lengths')
            state_length_sampler = None
        else:
            raise Exception('unimplemented')

        # calling our simulator
        infections = torch_forward_time(initial_state, state_length_sampler, household_beta, self.connectivity_matrix, importation_probability, **kwargs)
                        
        num_infections = pd.Series(np.sum(infections, axis=1).squeeze())
        num_infections.name = "infections"
        #self.df["infections"] = num_infections
        #assert (self.df["infections"] <= self.df["size"]).all(), "Saw more infections than the size of the household"
        #import pdb; pdb.set_trace()
        return pd.concat([self.df, pd.Series(num_infections)], axis=1)

#x = Model()
#x.run_trials(0.05, sizes={10:1000})
