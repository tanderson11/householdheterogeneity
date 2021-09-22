import matplotlib.pyplot as plt
import numpy as np
import abc

class Trait(abc.ABC):
    def __call__(self, occupants):
        return self.draw_from_distribution(occupants)
    
    def plot(self, samples=1000, **kwargs):
        shaped_array = np.full((samples,), True)
        output = np.array(self(shaped_array))
        sample_mean = np.mean(output)
        sample_var = np.var(output)
        ninetieth_percentile = np.percentile(output, 90)

        plt.hist(output, **kwargs)
        plt.title("{0}.\nSample mean {1:.2f} and sample var {2:.2f}\n 90th percentile {3:.2f}".format(self, sample_mean, sample_var, ninetieth_percentile))
        plt.xlabel("relative magnitude")
        plt.ylabel("# people")

        plt.axvline(np.percentile(output, 10))
        plt.axvline(np.percentile(output, 50))
        plt.axvline(np.percentile(output, 90))

        return sample_mean, sample_var

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
        return "Constant trait named {0} with value {1:.2f}".format(self.trait_value)

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
            values = np.full_like(is_occupied, 0., dtype=float)
            filtered = is_occupied[is_occupied != False]
            # reassign values based on gamma dist
            filtered = np.random.gamma(self.mean**2/self.variance, scale=self.variance/self.mean, size=filtered.shape)
            values[is_occupied != False] = filtered
            #import pdb; pdb.set_trace()
            return values

    def __repr__(self):
        return "Gamma distributed trait with mean {0:.2f} and variance {1:.2f}".format(self.mean, self.variance)
    

#t = GammaTrait(mean=1.0, variance=1.0)
#t.plot(samples=100)