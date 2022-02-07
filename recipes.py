import enum
from typing import NamedTuple
import numpy as np
import pandas as pd

from settings import constants
import traits
from torch_forward_simulation import torch_forward_time, torch_state_length_sampler

class StateLengthConfig(enum.Enum):
    gamma = 'gamma'
    constant = 'constant'
    lognormal = 'lognormal'

class InitialSeedingConfig(enum.Enum):
    seed_one_by_susceptibility = 'seed one by susceptibility'
    no_initial_infections = 'seed none'

class ImportationRegime(NamedTuple):
    duration: int
    importation_rate: float

class Model(NamedTuple):
    # simulation configuration
    state_lengths: str = StateLengthConfig.gamma.value
    initial_seeding: str = InitialSeedingConfig.seed_one_by_susceptibility.value
    importation: ImportationRegime = None
    secondary_infections: bool = True # for debugging / testing

    def run_trials(self, household_beta=None, trials=1, population=None, sizes=None, sus=traits.ConstantTrait(), inf=traits.ConstantTrait()):
        assert household_beta is not None
        #import pdb; pdb.set_trace()
        if population is None:
            assert sizes is not None
            expanded_sizes = {size:count*trials for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations
            population = PopulationStructure(expanded_sizes, sus, inf)

        df = population.simulate_population(household_beta, *self)

        dfs = []
        grouped = df.groupby("size")
        for size, group in grouped:
            if trials > 1:
                count = sizes[size]
                trialnums = [t for t in range(trials) for i in range(count)]
            dfs.append(group)
        return pd.concat(dfs)

class Population(NamedTuple):
    is_occupied: np.ndarray
    sus: np.ndarray
    inf: np.ndarray
    connectivity_matrix: np.ndarray

    def seed_one_by_susceptibility(self):
        # Let's say a household has susceptibilites like 0.5, 1.0, 2.0
        # we want to map these onto a 'die' that is rolled between 0 and 1
        # like 1/7, 2/7, 4/7 -> 1/7, 3/7, 7/7. Then we roll our die in [0,1] and look where it 'landed'
        # ie: maps the susceptibilities onto their share of the unit interval
        roll_mapping = np.cumsum(np.squeeze(self.sus)/np.sum(self.sus, axis=1), axis=1)
        roll = np.random.random((roll_mapping.shape[0],1))
        hits = np.where(roll_mapping > roll) # our first `False` in each row of this array corresponds to where we want to seed the infection
        # we want only the unique values in the row indices so if the roll was 2/7 we want just 3/7 not also 7/7
        row_hits, one_index_per_row = np.unique(hits[0], return_index=True)
        col_hits = hits[1][one_index_per_row]

        initial_state = self.is_occupied * constants.STATE.susceptible
        initial_state[row_hits, col_hits] = constants.STATE.exposed
        return initial_state

    def seed_none(self):
        initial_state = self.is_occupied * constants.STATE.susceptible
        return initial_state

    def make_initial_state(self, initial_seeding):
        initial_seeding = InitialSeedingConfig(initial_seeding)

        if initial_seeding == InitialSeedingConfig.seed_one_by_susceptibility:
            initial_state = self.seed_one_by_susceptibility()
        elif initial_seeding == InitialSeedingConfig.seed_none:
            initial_state = self.seed_none()
        else:
            raise Exception(f"unimplimented initial seeding name {initial_seeding}")

        initial_state = np.expand_dims(initial_state, axis=2)
        return initial_state

class PopulationStructure:
    def __init__(self, household_sizes, susceptibility=traits.ConstantTrait(), infectivity=traits.ConstantTrait()):
        assert isinstance(household_sizes, dict)
        self.household_sizes_dict = household_sizes
        assert isinstance(susceptibility, traits.Trait)
        self.susceptibility = susceptibility
        assert isinstance(infectivity, traits.Trait)
        self.infectivity = infectivity

        unpacked_sizes = [[size]*number for size, number in household_sizes.items()]
        flattened_unpacked_sizes = [x for l in unpacked_sizes for x in l]
        self.sizes_table = pd.Series(flattened_unpacked_sizes, name="size")

        self.max_size = self.sizes_table.max()
        #import pdb; pdb.set_trace()
        self.is_occupied = np.array([[True if i < hh_size else False for i in range(self.max_size)] for hh_size in self.sizes_table], dtype=bool) # 1. if individual exists else 0.
        
        self.total_households = len(self.sizes_table)

        self._nd_eyes = np.stack([np.eye(self.max_size,self.max_size) for i in range(self.total_households)]) # we don't worry about small households here because it comes out in a wash later
        self._adjmat = 1 - self._nd_eyes # this is used so that an individual doesn't 'infect' themself

    def make_population(self):
        # creates a real draw from the trait distributions to represent
        # one instance of the abstract structure
        sus = np.expand_dims(self.susceptibility.draw_from_distribution(self.is_occupied), axis=2) # ideally we will get rid of expand dims at some point
        inf = np.transpose(np.expand_dims(self.infectivity.draw_from_distribution(self.is_occupied), axis=2), axes=(0,2,1))

        connectivity_matrix = (sus @ inf) * self._adjmat

        return Population(self.is_occupied, sus, inf, connectivity_matrix)

    def simulate_population(self, household_beta, state_lengths, initial_seeding, importation, secondary_infections=True):
        ###################
        # Make population #
        ###################

        pop = self.make_population()

        ################################
        # Process and verify arguments #
        ################################

        initial_state = pop.make_initial_state(initial_seeding)

        if importation is None:
            importation_probability = 0. * pop.sus
        else:
            if initial_state.any():
                print("WARNING: importation rate is defined while initial infections were seeded. Did you intend this?")
            importation_probability = importation.importation_rate * pop.sus

        # select the appropriate function for state lengths based on config str:
        state_length_config = StateLengthConfig(state_lengths)
        if state_length_config == StateLengthConfig.gamma:
            state_length_sampler = torch_state_length_sampler
        elif state_length_config == StateLengthConfig.constant:
            raise Exception('unimplemented constant state lengths')
        else:
            raise Exception('unimplemented state length configuration')

        ############
        # Simulate #
        ############

        infections = torch_forward_time(initial_state, state_length_sampler, household_beta, pop.connectivity_matrix, importation_probability, secondary_infections=secondary_infections)
                        
        num_infections = pd.Series(np.sum(infections, axis=1).squeeze())
        num_infections.name = "infections"

        return pd.concat([self.sizes_table, pd.Series(num_infections)], axis=1)

if __name__ == '__main__':
    x = Model()
    x.run_trials(0.05, sizes={5:1, 4:1}, sus=traits.BiModalTrait(2.0))
    x = PopulationStructure({5:10000, 10:10000})
    pop = x.make_population()
    pop.seed_one_by_susceptibility()
    #pop.seed_one_by_susceptibility()