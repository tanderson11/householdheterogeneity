import matplotlib.pyplot as plt
import numpy as np
import abc

class Trait(abc.ABC):
    def __call__(self, occupants):
        return self.draw_from_distribution(occupants)
    
    def plot(self, samples=1000, **kwargs):
        output = np.array(self(samples))
        sample_mean = np.mean(output)
        sample_var = np.var(output)

        plt.hist(output, **kwargs)
        plt.title("{0}.\nSample mean {1:.2f} and sample var {2:.2f}".format(self, sample_mean, sample_var))
        plt.xlabel("relative magnitude of {0}".format(self.name))
        plt.ylabel("# people")

    @abc.abstractmethod
    def draw_from_distribution(self, occupants):
        '''Takes:
                is_occupied:pd.DataFrame a table of households where values are masked
                True=individual exists and False=individual not present in household
            Returns:
                np.array of households with the trait values for individuals
                if that individual is present and 0s otherwise to pad the size to max_size
        '''
        pass

class ConstantTrait(Trait):
    def __init__(self, trait_value=1.0):
        super().__init__()
        self.trait_value = trait_value

    def draw_from_distribution(self, is_occupied):
        return is_occupied * self.trait_value

    def __repr__(self):
        return "Constant trait named {0} with value {1:.2f}".format(self.name, self.trait_value)

class GammaTrait(Trait):
    def __init__(self, mean, variance):
        # copy fields
        super().__init__()
        self.mean = mean
        self.variance = variance

    def draw_from_distribution(self, is_occupied):
        if self.variance == 0:
            return is_occupied * self.mean
        else:
            occupants = is_occupied.copy()
            filtered = occupants[occupants != 0]
            filtered = np.random.gamma(self.mean**2/self.variance, scale=self.variance/self.mean, size=filtered.shape) # reassign values based on gamma dist
            occupants[occupants != 0] = filtered

            #import pdb; pdb.set_trace()
            return occupants

    def __repr__(self):
        return "Gamma distributed trait named {0} with mean {1:.2f} and variance {2:.2f}".format(self.name, self.mean, self.variance)
    

