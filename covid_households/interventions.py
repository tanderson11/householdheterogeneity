import abc
import numpy as np

class Intervention():
    def __init__(self, intervention_shape):
        self.intervention_shape = intervention_shape
    
    def apply(self, sus, inf):
        intervention_mask = self.where_intervene(sus, inf)
        sus = np.where(
            intervention_mask,
            self.intervene_on_sus(sus),
            sus
        )
        
        inf = np.where(
            np.swapaxes(intervention_mask, 1, 2),
            self.intervene_on_inf(inf),
            inf
        )

        return sus, inf

    def where_intervene(self, sus, inf):
        return self.intervention_shape.intervention_mask(sus, inf)

    def intervene_on_sus(self, sus):
        """Applies the intervention to the susceptibilities of individuals.

        Args:
            sus (np.ndarray): a vector of susceptibilities for individuals in the population.
        
        Returns:
            sus (np.ndarray): the new susceptibilities
        """
        return sus
    
    def intervene_on_inf(self, inf):
        """Applies the intervention to the infectivities of individuals.

        Args:
            inf (np.ndarray): a vector of infectivities for individuals in the population. (transposed relative to sus)
        
        Returns:
            inf (np.ndarray): the new infectivities
        """
        return inf

class ConstantFactorIntervention(Intervention):
    def __init__(self, intervention_shape, sus_factor, inf_factor):
        super().__init__(intervention_shape)
        self.sus_factor = sus_factor
        self.inf_factor = inf_factor

    def intervene_on_sus(self, sus):
        return self.sus_factor * sus

    def intervene_on_inf(self, inf):
        return self.inf_factor * inf

class InterventionShape(abc.ABC):
    @abc.abstractmethod
    def intervention_mask(self, sus, inf):
        return np.full_like(sus, True)

class InterveneOnFirst(InterventionShape):
    def intervention_mask(self, sus, inf):
        #import pdb; pdb.set_trace()
        mask = np.full_like(sus, False)
        mask[:, 0] = True

        return mask

class InterveneOnFraction(InterventionShape):
    def __init__(self, fraction_affected) -> None:
        self.fraction_affected = fraction_affected
    
    def intervention_mask(self, sus, inf):
        shape = sus.shape
        mask = np.zeros(shape)
        affected_in_household = np.arange(np.around(shape[1] * self.fraction_affected)).astype('int32') #axis 1 corresponds to the individuals in a household
        if np.around(shape[1] * self.fraction_affected) != shape[1] * self.fraction_affected:
            print("Warning, attempted to vaccinate fraction={0} of household with size={1}. Vaccinated {2}".format(self.fraction_affected, shape[1], np.around(shape[1] * self.fraction_affected)))
        mask[:, affected_in_household] = 1
        return mask 