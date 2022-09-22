import enum
from typing import NamedTuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import json
from typing import OrderedDict

from model_inputs import ModelInputs
from settings import model_constants
from settings import STATE
import state_lengths as state_length_module
import traits
from torch_forward_simulation import torch_forward_time
from gillespie_forward_simulation import gillespie_simulation
from interventions import Intervention

### ENUMERATIONS
# We use `enum.Enum`s (enumerations) to create settings that are constrained to a finite set of options.
# This provides easy validation because if we have a string `setting_name`
# we can try SettingEnumeration(`setting_name`), which throws an error if that is an invalid setting.

class StateLengthConfig(enum.Enum):
    lognormal = 'lognormal'
    gamma = 'gamma'
    constant = 'constant'

class ForwardSimulationConfig(enum.Enum):
    gillespie = 'gillespie'
    finite_timesteps = 'finite timesteps'

class InitialSeedingConfig(enum.Enum):
    seed_one_by_susceptibility = 'seed one by susceptibility'
    no_initial_infections = 'seed none'

class ImportationRegime(NamedTuple):
    duration: int
    importation_rate: float

class Model(NamedTuple):
    """The Model class contains configuration and settings related to how forward simulation is performed.

    Fields:
        state_lengths: what distribution of durations should be used for the infectious and exposed state. Defaults to 'lognormal'.
        forward_simulation: what method of forward simulation should be used. Defaults to 'gillespie'.
        importation: a tuple of (# days, importation probability) that adds a constant rate of risk of importation into a household. Defaults to None.
        secondary_infections: DEBUG only: whether infected secondary contacts spread chains of infection. Defaults to True.
        intervention: an optional Intervention object that acts on the relative susceptibilities, infectivies, and the initial state to represent an intervention. Defaults to None.
    """
    state_lengths: str = StateLengthConfig.lognormal.value
    forward_simulation: str = ForwardSimulationConfig.gillespie.value
    initial_seeding: str = InitialSeedingConfig.seed_one_by_susceptibility.value
    importation: ImportationRegime = None
    secondary_infections: bool = True # for debugging / testing
    intervention: Intervention = None

    def run_trials(self, household_beta=None, trials=1, population=None, sizes=None, sus=traits.ConstantTrait(), inf=traits.ConstantTrait(), as_counts=True):
        """Simulate a group of households forward in time many times and collect the outcome of infections / household grouped by trial.

        Args:
            household_beta (float, optional): the probability/time of an infection passing
                from infectious --> susceptible. Defaults to None.
            trials (int, optional): how many differently seeded outbreaks to simulate
                for the same population. Defaults to 1.
            population (recipes.Population, optional): a PopulationStructure object
                that specifies the sizes of households.
                Either population or sizes must be specified. Defaults to None.
            sizes (dict, optional): a dictionary that describes the cohort of households
                by mapping household size --> # households of that size.
                Either population or sizes must be specified. Defaults to None.
            sus (traits.Trait, optional): a Trait object that describes the distribution of relative susceptibility
                in the population. Defaults to traits.ConstantTrait().
            inf (traits.Trait, optional): a Trait object that describes the distribution of relative infectivity
                in the population. Defaults to traits.ConstantTrait().
            as_counts (bool, optional): if True, group the outcomes by household size.
                If False, return outcomes in each household separately. Defaults to True.

        Returns:
            pandas.DataFrame: a table of outcome infections where 'infections' indicates how many infections where present
                at fixation (counting the index case) and 'trial' indicates which trial those infections occurred in.
                The shape of the data depends on the `use_counts` argument.
        """
        assert household_beta is not None
        if population is None:
            assert sizes is not None
            expanded_sizes = {size:count*trials for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations
            population = PopulationStructure(expanded_sizes, sus, inf)

        df = population.simulate_population(household_beta, *self, sus=sus, inf=inf)
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

    def run_index(self, sizes, index, parameter_class, progress_path=None, progress_frequency=1500, use_beta_crib=False):
        keys = index.names
        metadata = Metadata(model_constants.as_dict(), self, sizes, list(keys))
        if progress_path is not None:
            metadata.save(progress_path)

        population = None
        results_list = []
        for i,point in enumerate(index):
            # treat the point as parameters
            params = OrderedDict({key: value for key,value in zip(keys,point)})
            # convert the point in the region to inputs that are recognizable for simulation
            default_parameters = parameter_class(**params).to_normal_inputs(use_beta_crib=use_beta_crib)

            # for the first point, we need to create the PopulationStructure object, which knows who lives in which household and therefore all the connections between individuals
            # the random elements will all be calculated when the Population object is realized in the `run_trials` method below
            if population is None:
                expanded_sizes = {size:count*1 for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations
                population = PopulationStructure(expanded_sizes, default_parameters['sus'], default_parameters['inf'])

            point_results = self.run_point(sizes, population, keys, point, default_parameters)
            results_list.append(point_results)
            if i != 0 and i % progress_frequency == 0:
                partial_df = pd.concat(results_list)
                parquet_df = pa.Table.from_pandas(partial_df)
                pq.write_table(parquet_df, os.path.join(progress_path, f"pool_df-through-index-{i}.parquet"))

        df = pd.concat(results_list).reset_index().set_index(keys + ['size', 'infections'])
        return Results(df, metadata)

    def run_grid(self, sizes, region, progress_path=None, use_beta_crib=False):
        """Simulate a cohort of households forward in time for every combination or parameter values in the specified region using the configuration of this Model object.

        Args:
            sizes (dict): a mapping of household size --> # households of that size,
                which describes the population whose initial state is to be simulated forward in time.
            region (recipes.SimulationRegion): a convex 3D region in parameter space over which to simulate.
            progress_path (str, optional): the path to a directory in which to save incremental progress
                and final results. Defaults to None.
            use_beta_crib (bool, optional): if True, use a precalculated mapping of s80,p80,SAR parameter values
                --> necessary inputs for simulation to save time. Defaults to False.

        Returns:
            recipes.Results: the results at each combination of parameters
                with attributes `df` (the table of infections for the households)
                    and `metadata` (information about the settings used to execute simulation).
        """
        axis_data = list(region.axes_by_name.items())
        # extract the names of each parameter and the range of each parameter
        key1, axis1 = axis_data[0]
        key2, axis2 = axis_data[1]
        key3, axis3 = axis_data[2]

        metadata = Metadata(model_constants.as_dict(), self, sizes, list(region.axes_by_name.keys()))
        if progress_path is not None:
            metadata.save(progress_path)

        population = None

        # for each point in the region, simulate infections and hold onto the results
        two_d_dfs = []
        for v1 in axis1:
            one_d_dfs = []
            for v2 in axis2:
                for v3 in axis3:
                    params = OrderedDict()
                    params[key1] = v1
                    params[key2] = v2
                    params[key3] = v3
                    keys = list(params.keys())
                    values = list(params.values())
                    # convert the point in the region to inputs that are recognizable for simulation
                    default_parameters = region.parameter_class(**params).to_normal_inputs(use_beta_crib=use_beta_crib)

                    # for the first point, we need to create the PopulationStructure object, which knows who lives in which household and therefore all the connections between individuals
                    # the random elements will all be calculated when the Population object is realized in the `run_trials` method below
                    if population is None:
                        expanded_sizes = {size:count*1 for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations
                        population = PopulationStructure(expanded_sizes, default_parameters['sus'], default_parameters['inf'])

                    point_results = self.run_point(sizes, population, keys, values, default_parameters)
                    # we collect results for each 'line' in the 3D grid, then add those lines together to make planes, and then compile all the planes into a cube
                    one_d_dfs.append(point_results)
            two_d_df = pd.concat(one_d_dfs)
            two_d_dfs.append(two_d_df)
            # save progress incrementally for each plane
            if progress_path is not None:
                parquet_df = pa.Table.from_pandas(two_d_df)
                pq.write_table(parquet_df, os.path.join(progress_path, f"pool_df-{key1}-{v1:.3f}.parquet"))
            two_d_df = None
        three_d_df = pd.concat(two_d_dfs)

        # let the custom labels of the regions become indices into the DataFrame
        df = three_d_df.reset_index().set_index(metadata.parameters + ['size', 'infections'])
        return Results(df, metadata)

    def run_point(self, sizes, population, keys, values, default_parameters):
        # extract the three desired quantities from the default parameters
        sus_dist = default_parameters['sus']
        inf_dist = default_parameters['inf']
        beta = default_parameters['household_beta']
        # simulate to find resultant infections at this point in parameter space
        point_results = self.run_trials(
            beta, population=population, sizes=sizes, sus=sus_dist, inf=inf_dist, as_counts=True
        )

        # add labels to the dataframe based on the parameters truly used to simulate
        trait_parameter_name, trait_parameter_value = sus_dist.as_column()
        point_results[f'sus_{trait_parameter_name}'] = np.float(f"{trait_parameter_value:.3f}")
        trait_parameter_name, trait_parameter_value = inf_dist.as_column()
        point_results[f'inf_{trait_parameter_name}'] = np.float(f"{trait_parameter_value:.3f}")
        point_results['beta'] = np.float(f"{beta:.3f}")
        # and add labels based on the parameters that are custom for this region
        point_results[keys[0]] = np.float(f"{values[0]:.3f}")
        point_results[keys[1]] = np.float(f"{values[1]:.3f}")
        point_results[keys[2]] = np.float(f"{values[2]:.3f}")
        return point_results

class Population(NamedTuple):
    is_occupied: np.ndarray
    sus: np.ndarray
    inf: np.ndarray

    def seed_one_by_susceptibility(self):
        """Introduce one infection to each household in proportion to the susceptibility of individuals.

        Returns:
            np.ndarray: the initial state of infections in each household in the population. Values correspond to constants.STATE enumeration values.
        """
        # Let's say a household has susceptibilites like 0.5, 1.0, 2.0
        # We want to map these onto a 'die' that is rolled between 0 and 1
        # Like 1/7, 2/7, 4/7 -> 1/7, 3/7, 7/7. Then we roll our die in [0,1] and look where it 'landed'
        # IE: maps the susceptibilities onto their share of the unit interval
        # This line of code does that:
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
        """Using the specified initial_seeding protocol, generate initial infections in each household.

        Args:
            initial_seeding (str): a valid choice of initial seeding protocol.

        Raises:
            Exception: if the initial_seeding string is not recognized as a valid configuration.

        Returns:
            np.ndarray: the initial state of infections in each household in the population.
                Values correspond to constants.STATE enumeration values.
        """
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
        """Makes a matrix of *relative* probabilities with ith row jth column
            corresponding to the relative probability that the ith individual is infected by the jth individual

        Args:
            adjmat (np.ndarray): an array of True and False where ij = True means that i could be infected by j.
                (IE: they live in the same household, both exist, but they're not the same people)

        Returns:
            [np.ndarray] -- connectivity matrix of relative probabilities infection i <-- j
        """
        connectivity_matrix = (self.sus @ self.inf) * adjmat
        return connectivity_matrix

    def apply_intervention(self, intervention_scheme):
        """Applies an invention to a population given its initial state and returns a new population (new set of traits).

        Args:
            population (Population): a realized population of individuals in households
                with susceptibilities (population.sus) and infectivies (population.inf).
            intervention_scheme (Intervention): an intervention object that can `.apply`
                itself to traits given the initial state.

        Returns:
            Population: a population where the trait values of individual (might) have been modified by an intervention.
        """
        sus, inf = intervention_scheme.apply(self.sus, self.inf)
        return self.__class__(self.is_occupied, sus, inf)

class SimulationRegion(NamedTuple):
    """A region in parameter space that describes combinations of parameter values to be simulated forwards in time.

    Fields:
        axes_by_name (OrderedDict): an ordered dictionary (in to-be-simulated order) that maps 'name of parameter' --> 'numpy vector of parameter values that are of interest.' for each parameter of interest.
        parameter_class (utilities.ModelInputs): an object that can map a point in the region to the necessary values for simulation (consisting of the susceptibility distribution, the infectivity distribution, and the probability/time of infection).
    """
    axes_by_name: OrderedDict
    parameter_class: ModelInputs

class NpEncoder(json.JSONEncoder):
    """A custom json encoder used to save numpy integers as python integers for better serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NpEncoder, self).default(obj)

class Metadata(NamedTuple):
    constants: dict
    model: Model
    population: dict
    parameters: type

    def save(self, root, name='metadata'):
        """Save this metatdata object as a json file at the root directory.

        Args:
            root (str): the path to the directory in which to save the metadata.
            metadata (str, optional): the name of the json file. Defaults to 'metadata'.
        """
        with open(os.path.join(root, f'{name}.json'), 'w') as f:
            json.dump(self, f, cls=NpEncoder)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)

        data[cls._fields.index('model')] = Model(*data[cls._fields.index('model')])
        metadata = cls(*data)
        return metadata

    def check_compatibility(self, m2):
        """Compare this Metadata object to another to see if the they are compatible
            (ie, represent an identical simulation approach so different approaches aren't accidentally combined).

        Args:
            m2 (recipes.Metadata): the Metadata to test compatibility with.

        Returns:
            bool: True if compatible, False if not.
        """
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

    def combine(self, r2, method='add', decimal_places=3):
        """Return the combined data about infections carried by this Results object and a second Results object — if they are compatible.

        Args:
            r2 (recipes.Results): another Results object to combine with.
            method (str, optional): either 'add' to sum entries or 'left' or 'right' to keep only the values from that side of the combine. Defaults to 'add'.
            decimal_places (int, optional): the decimal precision to use for the index to find shared values in the two indices. Defaults to 3.

        Raises:
            ValueError: raised if the two Results objects do not have compatible Metadata. Or if method argument is not recognized.

        Returns:
            recipes.Results: the combined results of the two Results objects.
        """
        if not self.metadata.check_compatibility(r2.metadata):
            raise ValueError("Tried to combine two results objects with incompatible metadata.")
        # combine dfs: concat and then update intersecting components to be the sum of the two components
        df3 = pd.concat([self.df, r2.df])
        #print("Duplicates?", df3.index.duplicated().any())
        df3 = df3[~(df3.index.duplicated(keep='first'))]
        #print("Duplicates?", df3.index.duplicated().any())
        if method == 'add':
            intersection = (self.df["count"] + r2.df["count"]).dropna()
        elif method == 'left':
            intersection = self.df["count"]
        elif method == 'right':
            intersection = r2.df["count"]
        else:
            raise ValueError(f"method of {method} is not recognized by combine.")
        
        df3.loc[intersection.index, "count"] = intersection
        # if we join on the right, we need to relabel in accordance with the parameters used on the right
        if method=='right':
            #import pdb; pdb.set_trace()
            df3.loc[intersection.index, "beta"] = r2.df.loc[intersection.index, "beta"]
            if 'inf_variance' in r2.df.columns:
                df3.loc[intersection.index, "inf_variance"] = r2.df.loc[intersection.index, "inf_variance"]
            if 'sus_variance' in r2.df.columns:
                df3.loc[intersection.index, "sus_variance"] = r2.df.loc[intersection.index, "sus_variance"]
            if 'inf_constant_value' in r2.df.columns:
                df3.loc[intersection.index, "inf_constant_value"] = r2.df.loc[intersection.index, "inf_constant_value"]
            if 'sus_constant_value' in r2.df.columns:
                df3.loc[intersection.index, "sus_constant_value"] = r2.df.loc[intersection.index, "sus_constant_value"]

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
        """Find the frequencies (ratio of observed/total occurrences) for different numbers of infections at each point in the index. In other words, convert results from a cohort of households into probabilities of those occurences.

        Args:
            minimum_size (int, optional): a minimum # of households at each point in the index. Since we generally want frequencies to correspond to a probability calculated in reference to many households, this warns us if we have few households in our results. Defaults to 20000.
            inplace (bool, optional): if True, add the calculated frequencies to the Results.df attribute as a new column. Defaults to True.

        Returns:
            pandas.Series: the frequency/probability of observing X infections in households of size Y.
        """
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
        """Checks that every size household is present at all combination of parameter values over the specified axes.

        Returns a mapping of combinations of parameters values that are missing sizes ---> sizes that are missing at that combination."""
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
        if len(missing) != 0:
            import pdb; pdb.set_trace()
        return missing

class PopulationStructure:
    def __init__(self, household_sizes, susceptibility=traits.ConstantTrait(), infectivity=traits.ConstantTrait(), is_occupied=None, adjmat=None):
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

        self.total_households = len(self.sizes_table)

        if is_occupied is None:
            self.is_occupied = np.array([[True if i < hh_size else False for i in range(self.max_size)] for hh_size in self.sizes_table], dtype=bool) # 1. if individual exists else 0.
        else:
            self.is_occupied = is_occupied

        if adjmat is None:
            _nd_eyes = np.stack([np.eye(self.max_size,self.max_size) for i in range(self.total_households)]) # we don't worry about small households here because it comes out in a wash later
            self._adjmat = 1 - _nd_eyes # this is used so that an individual doesn't 'infect' themself
        else:
            self._adjmat = adjmat

    def make_population(self, susceptibility=None, infectivity=None):
        # if we get overrides for susceptibility and infectivity, we use those
        susceptibility = susceptibility if susceptibility is not None else self.susceptibility
        infectivity = infectivity if infectivity is not None else self.infectivity

        # creates a real draw from the trait distributions to represent
        # one instance of the abstract structure
        # ideally we will get rid of expand dims at some point
        sus = np.expand_dims(susceptibility.draw_from_distribution(self.is_occupied), axis=2)
        inf = np.transpose(np.expand_dims(infectivity.draw_from_distribution(self.is_occupied), axis=2), axes=(0,2,1))

        return Population(self.is_occupied, sus, inf)

    def simulate_population(self, household_beta, state_lengths, forward_simulation, initial_seeding, importation, secondary_infections=True, intervention=None, sus=None, inf=None):
        ###################
        # Make population #
        ###################

        # a population is a realization of this PopulationStructure. People are connected in the way described by the structure, but their relative traits are randomly decided for this specific population.
        pop = self.make_population(sus, inf)

        if intervention is not None:
            pop = pop.apply_intervention(intervention)
        ################################
        # Process and verify arguments #
        ################################

        # create an initial state based on the seeding protocol
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
            state_length_sampler = state_length_module.gamma_state_length_sampler
        elif state_length_config == StateLengthConfig.lognormal:
            state_length_sampler = state_length_module.lognormal_state_length_sampler
        elif state_length_config == StateLengthConfig.constant:
            raise Exception('unimplemented constant state lengths')
        else:
            raise Exception('unimplemented state length configuration')

        forward_simulation_config = ForwardSimulationConfig(forward_simulation)
        if forward_simulation_config == ForwardSimulationConfig.finite_timesteps:
            simulator = torch_forward_time
        elif forward_simulation_config == ForwardSimulationConfig.gillespie:
            simulator = gillespie_simulation
        else:
            raise Exception('unimplemented forward simulation configuration')

        ############
        # Simulate #
        ############

        infections = simulator(initial_state, household_beta, state_length_sampler, pop.sus, pop.inf, self._adjmat, np_importation_probability=importation_probability, secondary_infections=secondary_infections)

        num_infections = pd.Series(np.sum(infections, axis=1).squeeze())
        num_infections.name = "infections"

        df = pd.concat([self.sizes_table, pd.Series(num_infections)], axis=1)

        if intervention is not None and intervention.track_colors:
            intervention_and_infection = np.where(
                intervention._intervention_mask,
                infections,
                np.zeros_like(infections)
            )
            df['intervention and infection'] = np.sum(intervention_and_infection, axis=1).squeeze()
            df['total interventions'] = np.sum(intervention._intervention_mask, axis=1).squeeze()

        return df

if __name__ == '__main__':
    x = Model()
    x.run_trials(0.05, sizes={5:1, 4:1}, sus=traits.BiModalTrait(2.0))
    x = PopulationStructure({5:10000, 10:10000})
    pop = x.make_population()
    pop.seed_one_by_susceptibility()
