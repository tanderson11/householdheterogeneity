import abc

class Intervention(abc.ABC):
    def __init__(self, name):
        self.name = name
    
    @abc.abstractmethod
    def apply(self, sus, inf, initial_state):
        """Applies the intervention to a population given its initial state.

        Args:
            sus (np.ndarray): a vector of susceptibilities for individuals in the population.
            inf (np.ndarray): a vector of infectivities for individuals in the population. (transposed relative to sus)
            initial_state (np.ndarray): the initial state of the population at time t=0. Values correspond to `constants.STATE`
        
        Returns:
            sus (np.ndarray): the new susceptibilities
            inf (np.ndarray): the new infectivities
        """
        return sus, inf

    def __repr__(self) -> str:
        return f'Intervention named {self.name}'

class ConstantFactor(Intervention):
    def __init__(self, name, factor=1.0):
        super().__init__(name)
        self.factor = factor

    def apply(self, sus, inf, initial_state):
        return sus * self.factor, inf

    def __repr__(self) -> str:
        return f'Multiply susceptibility by {self.factor}'