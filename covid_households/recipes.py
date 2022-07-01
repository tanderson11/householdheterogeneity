import enum
from typing import NamedTuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import json

from settings import model_constants
from settings import STATE
import state_lengths as state_length_module
import traits as traits
from torch_forward_simulation import torch_forward_time
from gillespie_forward_simulation import gillespie_simulation
from interventions import Intervention

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
    state_lengths: str = StateLengthConfig.lognormal.value
    initial_seeding: str = InitialSeedingConfig.seed_one_by_susceptibility.value
    importation: ImportationRegime = None
    secondary_infections: bool = True # for debugging / testing
    intervention: Intervention = None

    def run_trials(self, household_beta=None, trials=1, population=None, sizes=None, sus=traits.ConstantTrait(), inf=traits.ConstantTrait(), as_counts=False, nice_index=False):
        assert household_beta is not None
        #import pdb; pdb.set_trace()
        if population is None:
            assert sizes is not None
            expanded_sizes = {size:count*trials for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations
            population = PopulationStructure(expanded_sizes, sus, inf)

        df = population.simulate_population(household_beta, *self)

        # restore a notion of 'trials'
        # (we flattened the households into one large population for efficiency, now add the labels back to restructure)
        dfs = []
        grouped = df.groupby("size")
        for size, group in grouped:
            if trials > 1:
                count = sizes[size]
                trialnums = [t for t in range(trials) for i in range(count)]
                group['trial'] = trialnums
            dfs.append(group)

        df = pd.concat(dfs)
        if as_counts:
            grouping = ['size','infections']
            if trials > 1: grouping.append('trial')
            df = pd.DataFrame(df.groupby(grouping).size())
            df.rename(columns={0:'count'}, inplace=True)

        return df

    def run_grid(self, sizes, region, progress_path=None):
        axis_data = list(region.axes_by_name.items())
        key1, axis1 = axis_data[0]
        key2, axis2 = axis_data[1]
        key3, axis3 = axis_data[2]

        metadata = Metadata(model_constants.as_dict(), self, sizes, list(region.axes_by_name.keys()))
        if progress_path is not None:
            metadata.save(progress_path)

        two_d_dfs = []
        for v1 in axis1:
            one_d_dfs = []
            for v2 in axis2:
                for v3 in axis3:
                    params = {}
                    params[key1] = v1
                    params[key2] = v2
                    params[key3] = v3

                    default_parameters = region.parameter_class(**params).to_normal_inputs()

                    sus_dist = default_parameters['sus']
                    inf_dist = default_parameters['inf']
                    beta = default_parameters['household_beta']

                    point_results = self.run_trials(beta, sizes=sizes, sus=sus_dist, inf=inf_dist, as_counts=True)

                    trait_parameter_name, trait_parameter_value = sus_dist.as_column()
                    point_results[f'sus_{trait_parameter_name}'] = np.float(f"{trait_parameter_value:.3f}")
                    trait_parameter_name, trait_parameter_value = inf_dist.as_column()
                    point_results[f'inf_{trait_parameter_name}'] = np.float(f"{trait_parameter_value:.3f}")
                    point_results['beta'] = np.float(f"{beta:.3f}")

                    point_results[key1] = v1
                    point_results[key2] = v2
                    point_results[key3] = v3

                    one_d_dfs.append(point_results)
            two_d_df = pd.concat(one_d_dfs)
            #return two_d_df
            two_d_dfs.append(two_d_df)
            if progress_path is not None:
                parquet_df = pa.Table.from_pandas(two_d_df)
                pq.write_table(parquet_df, os.path.join(progress_path, f"pool_df-{key1}-{v1:.3f}.parquet"))
            two_d_df = None
        three_d_df = pd.concat(two_d_dfs)

        df = three_d_df.reset_index().set_index(metadata.parameters + ['size', 'infections'])
        return Results(df, metadata)

class Population(NamedTuple):
    is_occupied: np.ndarray
    sus: np.ndarray
    inf: np.ndarray

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

        initial_state = self.is_occupied * STATE.susceptible
        initial_state[row_hits, col_hits] = STATE.exposed
        return initial_state

    def seed_none(self):
        initial_state = self.is_occupied * STATE.susceptible
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
    
    def make_connectivity_matrix(self, adjmat):
        """Makes a matrix of *relative* probabilities with ith row jth column corresponding to the relative probability that ith individual is infected by the jth individual

        Args:
            adjmat (np.ndarray): an array of True and False where ij = True means that i could be infected by j. (IE: they live in the same household, both exist, but they're not the same people)
        
        Returns:
            [np.ndarray] -- connectivity matrix of relative probabilities infection i <-- j
        """
        # matrix of 
        connectivity_matrix = (self.sus @ self.inf) * adjmat
        return connectivity_matrix

    @classmethod
    def apply_intervention(cls, intervention_scheme, population, initial_state):
        """Applies an invention to a population given its initial state and returns a new population (new set of traits).

        Args:
            population (Population): a realized population of individuals in households with susceptibilities (sus) and infectivies (inf).
            intervention_scheme (Intervention): an intervention object that can `.apply` itself to traits given the initial state.
            initial_state (np.ndarray): the initial state of the population at time t=0. Values correspond to `constants.STATE`

        Returns:
            Population: a population where the trait values of individual (might) have been modified by an intervention.
        """
        sus, inf = intervention_scheme.apply(population.sus, population.inf, initial_state)
        return cls(population.is_occupied, sus, inf)

from utilities import ModelInputs
from typing import OrderedDict
class SimulationRegion(NamedTuple):
    axes_by_name: OrderedDict
    parameter_class: ModelInputs

class Metadata(NamedTuple):
    constants: dict
    model: Model
    population: dict
    parameters: type

    def save(self, root):
        with open(os.path.join(root, 'metadata.json'), 'w') as f:
            json.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)

        #import pdb; pdb.set_trace()
        data[cls._fields.index('model')] = Model(*data[cls._fields.index('model')])
        metadata = cls(*data)
        return metadata

    def check_compatibility(self, m2):
        for i, field in enumerate(self._fields):
            if field != 'population' and self[i] != m2[i]:
                return False
        return True

class Results(NamedTuple):
    df: pd.DataFrame
    metadata: Metadata

    def save(self, root, filename):
        parquet_df = pa.Table.from_pandas(self.df)
        pq.write_table(parquet_df, os.path.join(root, filename + "_df.parquet"))
        self.metadata.save(root)

    @classmethod
    def load(cls, root, filename='results_df.parquet', from_parts=False):
        if not from_parts:
            df = pq.read_table(os.path.join(root, filename)).to_pandas()
        else:
            print("Compiling results df from parts. Treating all files with extension .parquet as parts.")
            parts = []
            observed_sizes = []
            for f in sorted(os.listdir(root)):
                if '.parquet' not in f:
                    continue
                part_df = pq.read_table(os.path.join(root, f)).to_pandas()
                #import pdb; pdb.set_trace()
                observed_sizes.append(part_df.reset_index()['size'].unique())
                parts.append(part_df)
            for x in observed_sizes:
                assert (observed_sizes[0] == x).all(), "Df parts usually come from a process that ensures equal sizes."
            df = pd.concat(parts)

        metadata = Metadata.load(os.path.join(root, 'metadata.json'))
        df = df.reset_index().set_index(metadata.parameters + ['size', 'infections'])
        df = df.rename(columns={"0":"count"}) # old tables might have improperly labeled count column

        return cls(df, metadata)

    def combine(self, r2, decimal_places=3):
        if not self.metadata.check_compatibility(r2.metadata):
            raise ValueError("Tried to combine two results objects with incompatible metadata.")
        # combine dfs: concat and then update intersecting components to be the sum of the two components
        df3 = pd.concat([self.df, r2.df])
        #print("Duplicates?", df3.index.duplicated().any())
        df3 = df3[~(df3.index.duplicated(keep='first'))]
        #print("Duplicates?", df3.index.duplicated().any())
        intersection = (self.df["count"] + r2.df["count"]).dropna()
        df3.loc[intersection.index, "count"] = intersection

        df3 = df3.rename(index=lambda val: round(val, decimal_places))
        df3 = df3.sort_index()

        # update the sizes dictionary to include the new min and max # of households at each size
        size_mins = df3.groupby(["s80", "p80", "SAR", "size"]).sum()['count'].groupby('size').min()
        size_maxs = df3.groupby(["s80", "p80", "SAR", "size"]).sum()['count'].groupby('size').max()
        size_dict = self.metadata.population.copy()
        for s in df3.index.unique(level="size"):
            # simultaneously update min,max sizes and also check that every size is present in every slice
            mi = size_mins.loc[s]
            ma = size_maxs.loc[s]
            size_dict[str(s)] = (int(mi),int(ma)) if mi != ma else mi
        # need to update sizes dictionary based on results
        # need to work out an example where the combined parts don't share all sizes
        return self.__class__(df3, self.metadata._replace(population=size_dict))

    def find_frequencies(self, minimum_size=20000, inplace=True):
        for _, minimum in self.metadata.population.items():
            if isinstance(minimum, tuple) or isinstance(minimum, list):
                minimum, _ = minimum
            assert minimum > minimum_size, "number of households was below minimum required in at least one part of df"

        frequencies = self.df["count"]/(self.df.groupby(self.metadata.parameters+["size"]).sum()["count"])
        if inplace:
            self.df['frequency'] = frequencies
        return frequencies

    def resample(self, parameter_point, population_sizes, trials=1):
        self.find_frequencies()
        dfs = []
        for i in range(trials):
            x = self.resample_once(parameter_point, population_sizes)
            x = pd.concat({i: x}, names=['trial'])
            dfs.append(x)
        samples = pd.concat(dfs)
        samples = samples.reindex(samples.index.rename(['trial'] + ["sample " + key for key in self.metadata.parameters] + ["size", "infections"]))
        return samples

    def resample_once(self, parameter_point, population_sizes):
        outer_index_names = self.df.index.names[:len(parameter_point)]
        point_df = self.df.loc[parameter_point].copy()

        # use the frequencies of different # of infections (for different size households)
        # get a new df that treats the frequencies as probabilities of occurence

        # presumptively: 0 for the counts of each infection until we update
        defaults = {(s,i):0 for s in population_sizes.keys() for i in range(1, s+1)}
        for size, number in population_sizes.items():
            size_df = point_df.loc[size]
            die_faces = tuple(size_df.index)
            die_weights = tuple(size_df['frequency'])
            infections, counts = np.unique(np.random.choice(die_faces, number, p=die_weights), return_counts=True)
            # count up the occurences in a way that we can then insert into our indexed df
            occurences = {(size, i):c for i,c in zip(infections, counts)}
            defaults.update(occurences)

        point_df["count"].update(pd.Series(defaults))

        # compute which sizes aren't included in our new population
        drop_sizes = []
        for s in point_df.index.unique(level='size'):
            if s not in population_sizes.keys():
                drop_sizes.append(s)

        # if we didn't resample for a certain size, drop it
        point_df = point_df.drop(drop_sizes)
        # frequency, which should exist, is no longer meaningful / accurate: drop it
        point_df = point_df.drop("frequency", axis=1)
        # add back the indices levels that we loc'd through at the start
        for name, value in zip(reversed(outer_index_names), reversed(parameter_point)):
            point_df = pd.concat({value: point_df}, names=[str(name)])

        return point_df

    @staticmethod
    def repair_missing_counts_in_group(group):
        sizes = list(group.index.unique(level='size'))
        initial_labels = group.index.names
        assert len(sizes) == 1
        size = sizes[0]
        # we want all the different number of infections to be present
        new_index = range(1, size+1)

        reindexed = group.reset_index().set_index('infections').reindex(new_index)
        # count is 0 everywhere it was missing before
        reindexed['count'].fillna(value=0., inplace=True)
        try:
            # same with frequency
            reindexed['frequency'].fillna(value=0., inplace=True)
        except KeyError:
            pass
        # otherwise ffill
        reindexed = reindexed.ffill()
        # because we've set and reset the index, the groups aren't being combined properly upstream
        # we drop the first few levels of the index to correct for this (they'll be present after aggregation)
        reindexed = reindexed.drop(initial_labels[:-1], axis=1)

        return reindexed

    def repair_missing_counts(self):
        grouped = self.df.groupby(self.metadata.parameters + ["size"])
        repaired = grouped.apply(lambda g: self.repair_missing_counts_in_group(g))
        return Results(repaired, self.metadata)

    def check_sizes_on_region(self, region, desired_sizes):
        return self.check_sizes_on_axes(region.axes_by_key, desired_sizes)

    def check_sizes_on_axes(self, axes_by_key, desired_sizes):
        '''Checks that every size household is present at all combination of parameter values over the specified axes.

        Returns a mapping of combinations of parameters values that are missing sizes ---> sizes that are missing at that combination.'''
        key_names = self.df.index.names[:3]
        missing = {}
        desired_sizes = set(desired_sizes)
        for x in axes_by_key[key_names[0]]:
            x = float(f'{x:.3f}')
            for y in axes_by_key[key_names[1]]:
                y = float(f'{y:.3f}')
                for z in axes_by_key[key_names[2]]:
                    z = float(f'{z:.3f}')
                    try:
                        slc = self.df.loc[x,y,z]
                        present_sizes = set(np.unique(slc.index.get_level_values('size')))
                    except KeyError:
                        present_sizes = set()
                    if present_sizes != desired_sizes:
                        missing_sizes =  desired_sizes - set(present_sizes)
                        missing[(x,y,z)] = missing_sizes
        return missing

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

        return Population(self.is_occupied, sus, inf)

    def simulate_population(self, household_beta, state_lengths, initial_seeding, importation, secondary_infections=True, intervention=None):
        ###################
        # Make population #
        ###################

        pop = self.make_population()

        ################################
        # Process and verify arguments #
        ################################

        initial_state = pop.make_initial_state(initial_seeding)
        if intervention is not None:
            pop = pop.apply_intervention(intervention, initial_state)

        if importation is None:
            importation_probability = 0. * pop.sus
        else:
            if initial_state.any():
                print("WARNING: importation rate is defined while initial infections were seeded. Did you intend this?")
            importation_probability = importation.importation_rate * pop.sus

        # select the appropriate function for state lengths based on config str:
        state_length_config = StateLengthConfig(state_lengths)
        if state_length_config == StateLengthConfig.gamma:
            state_length_sampler = state_length_module.gamma_state_length_sampler
        elif state_length_config == StateLengthConfig.lognormal:
            state_length_sampler = state_length_module.lognormal_state_length_sampler
        elif state_length_config == StateLengthConfig.constant:
            raise Exception('unimplemented constant state lengths')
        else:
            raise Exception('unimplemented state length configuration')

        ############
        # Simulate #
        ############

        infections = torch_forward_time(initial_state, household_beta, state_length_sampler, pop.sus, pop.inf, self._adjmat, np_importation_probability=importation_probability, secondary_infections=secondary_infections)

        num_infections = pd.Series(np.sum(infections, axis=1).squeeze())
        num_infections.name = "infections"

        return pd.concat([self.sizes_table, pd.Series(num_infections)], axis=1)

if __name__ == '__main__':
    x = Model()
    x.run_trials(0.05, sizes={5:1, 4:1}, sus=traits.BiModalTrait(2.0))
    x = PopulationStructure({5:10000, 10:10000})
    pop = x.make_population()
    pop.seed_one_by_susceptibility()