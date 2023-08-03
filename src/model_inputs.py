import abc
import pandas as pd
from pathlib import Path

import src.utilities as utilities
import src.traits as traits

class ModelInputs(abc.ABC):
    @abc.abstractmethod
    def to_normal_inputs(self):
        '''
        Returns a dictionary of the form:

        {
            'household_beta': float,
            'sus': Trait,
            'inf': Trait,
        }
        '''
        pass

class Lognormal_Variance_Variance_Beta_Inputs(ModelInputs):
    def __init__(self, sus_variance, inf_variance, household_beta) -> None:
        self.sus_varaince = sus_variance
        self.inf_variance = inf_variance
        self.household_beta = household_beta

    def to_normal_inputs(self, use_beta_crib=False, use_trait_crib=False):
        return {
            'household_beta': self.household_beta,
            'sus': traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=self.sus_varaince),
            'inf': traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=self.inf_variance),
        }

class S80_P80_SAR_Inputs(ModelInputs):
    key_names = ['s80', 'p80', 'SAR']
    # cribs are files that list precalculated mappings between these parameters and the ones required for simulation
    # they speed up simulation by avoiding the costly numerical solving step
    s80_crib  = pd.read_csv(Path(__file__).parent / "../data/s80_lookup.csv").set_index('s80')
    p80_crib  = pd.read_csv(Path(__file__).parent / "../data/p80_lookup.csv").set_index('p80')
    beta_crib = pd.read_csv(Path(__file__).parent / "../data/beta_lookup.csv").set_index(['s80', 'p80', 'SAR'])
    bad_combinations_crib = pd.read_csv(Path(__file__).parent / "../data/problematic_parameter_combinations.csv").set_index(['s80', 'p80', 'SAR'])

    def __init__(self, s80, p80, SAR) -> None:
        self.s80 = s80
        self.p80 = p80
        self.SAR = SAR

    def to_normal_inputs(self, use_beta_crib=False, use_trait_crib=True):
        if self.s80 == 0.8:
            sus_dist = traits.ConstantTrait()
        else:
            if use_trait_crib:
                sus_dist = traits.LognormalTrait(*tuple(self.s80_crib.loc[(float(f"{self.s80:.3f}"))]))
            else:
                sus_variance = utilities.lognormal_s80_solve(self.s80)
                assert(sus_variance.success is True)
                sus_variance = sus_variance.x[0]
                sus_dist = traits.LognormalTrait.from_natural_mean_variance(1., sus_variance)

        if self.p80 == 0.8:
            inf_dist = traits.ConstantTrait()
        else:
            if use_trait_crib:
                inf_dist = traits.LognormalTrait(*tuple(self.p80_crib.loc[(float(f"{self.p80:.3f}"))]))
            else:
                inf_variance = utilities.lognormal_p80_solve(self.p80)
                assert(inf_variance.success is True)
                inf_variance = inf_variance.x[0]
                inf_dist = traits.LognormalTrait.from_natural_mean_variance(1., inf_variance)

        if use_beta_crib:
            beta = float(self.beta_crib.loc[((float(f"{self.s80:.3f}")), (float(f"{self.p80:.3f}")), (float(f"{self.SAR:.3f}")))])
        else:
            beta = utilities.beta_from_sar_and_lognormal_traits(self.SAR, sus_dist, inf_dist)

        return {
            'household_beta': beta,
            'sus': sus_dist,
            'inf': inf_dist,
        }

    @classmethod
    def wrapped_beta_from_point(cls, point):
        return cls(*point).to_normal_inputs()['household_beta']

    def as_dict(self):
        return {
            's80':self.s80,
            'p80':self.p80,
            'SAR':self.SAR,
        }

    def __repr__(self) -> str:
        rep = "ModelInputs"
        return rep + f"({self.as_dict()}\n{self.to_normal_inputs()})"

parameterization_by_keys = {}
parameterization_by_keys[frozenset(S80_P80_SAR_Inputs.key_names)] = S80_P80_SAR_Inputs

def residual_wrapper(point, skip_old=True):
    if skip_old and (point in S80_P80_SAR_Inputs.bad_combinations_crib.index):
        return S80_P80_SAR_Inputs.bad_combinations_crib.loc[point]['residuals']
    return calculate_residual(point)

def calculate_residual(point):
    s80, p80, sar = point
    s80 = float(f"{s80:.3f}")
    p80 = float(f"{p80:.3f}")
    sar = float(f"{sar:.3f}")
    if s80 == 0.8:
        sus_dist = traits.ConstantTrait()
    else:
        mu, sigma = S80_P80_SAR_Inputs.s80_crib.loc[s80]
        sus_dist = traits.LognormalTrait(mu, sigma)
    if p80 == 0.8:
        inf_dist = traits.ConstantTrait()
    else:
        mu, sigma = S80_P80_SAR_Inputs.p80_crib.loc[p80]
        inf_dist = traits.LognormalTrait(mu, sigma)

    beta = S80_P80_SAR_Inputs.beta_crib.loc[s80, p80, sar]
    generalized_rv = utilities.lognormal_calculate_generalized_period(sus_dist, inf_dist)
    objective_function = utilities.lognormal_SAR_objective_function_crafter(sar, generalized_rv)
    return objective_function(beta)