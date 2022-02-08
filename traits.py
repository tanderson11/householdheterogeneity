import matplotlib.pyplot as plt
import numpy as np
import abc
import scipy.stats

class Trait(abc.ABC):
    def __call__(self, occupants):
        return self.draw_from_distribution(occupants)

    def plot(self, samples=1000, **kwargs):
        shaped_array = np.full((samples,), True)
        output = np.array(self(shaped_array))
        sample_mean = np.mean(output)
        sample_var = np.var(output)
        sample_median = np.median(output)
        ninetieth_percentile = np.percentile(output, 90)

        ax = plt.hist(output, **kwargs, stacked=True)
        #plt.title("{0}.\nSample mean {1:.2f} and sample var {2:.2f}\n median {3:.2f}\n 90th percentile {3:.2f}".format(self, sample_mean, sample_var, sample_median, ninetieth_percentile))
        plt.xlabel("relative magnitude")
        plt.ylabel("# people")

        #plt.axvline(np.percentile(output, 10))
        #plt.axvline(np.percentile(output, 50))
        #plt.axvline(np.percentile(output, 90))
        return ax

    @abc.abstractmethod
    def draw_from_distribution(self, is_occupied):
        '''Takes:
                is_occupied:pd.DataFrame a table of households where values are masked
                True=individual exists and False=individual not present in household
            Returns:
                np.array of households with the trait values for individuals
                if that individual is present and 0s otherwise to pad the size to max_size
        '''
        pass

    def as_dict(self):
        self_dict = {
            'distribution_type':self.distribution_type,
        }
        return self_dict

class ConstantTrait(Trait):
    distribution_type = 'constant'
    def __init__(self, trait_value=1.0):
        super().__init__()
        self.trait_value = trait_value

    def draw_from_distribution(self, is_occupied):
        return is_occupied * self.trait_value

    def __repr__(self):
        return "Constant trait named {0} with value {1:.2f}".format(self.trait_value)

    def as_dict(self):
        self_dict = super().as_dict()
        self_dict['trait_value'] = self.trait_value
        return self_dict

    def as_column(self):
        '''For when we log this trait in a pandas dataframe.'''
        assert(self.mean == 1)
        return ('constant_value', self.trait_value)

class GammaTrait(Trait):
    distribution_type = 'gamma'
    def __init__(self, mean=1.0, variance=None):
        # copy fields
        super().__init__()
        self.mean = mean
        self.variance = variance

    def draw_from_distribution(self, is_occupied):
        if self.variance == 0:
            return is_occupied * self.mean
        else:
            values = np.where(is_occupied, np.random.gamma(self.mean**2/self.variance, scale=self.variance/self.mean, size=is_occupied.shape), 0)
            #import pdb; pdb.set_trace()
            return values

    def __repr__(self):
        return "Gamma distributed trait with mean {0:.2f} and variance {1:.2f}".format(self.mean, self.variance)

    def as_dict(self):
        self_dict = super().as_dict()
        self_dict.update({'mean': self.mean, 'variance': self.variance})
        return self_dict

    def as_column(self):
        '''For when we log this trait in a pandas dataframe.'''
        assert(self.mean == 1)
        return ('variance', self.variance)


class BiModalTrait(Trait):
    distribution_type = 'bimodal'
    def __init__(self, n_fold_difference):
        self.n_fold = n_fold_difference
        # P_high * n + (1 - P_high) / n = 1.0
        # P_high * n^2 + 1 - P_high = n
        # P_high * n^2 - n + (1 - P_high) = 0
        # P_high (n^2 - 1) - n + 1 = 0
        # P_high = n-1 / (n^2 - 1)
        if self.n_fold == 1.0:
            self.probability_of_high_value = 1.0
        else:
            self.probability_of_high_value = (self.n_fold - 1)/(self.n_fold**2 - 1)

        mean = self.probability_of_high_value * self.n_fold + (1 - self.probability_of_high_value) / self.n_fold
        assert np.abs(mean - 1.0) < 0.001, mean

    def draw_from_distribution(self, occupants):
        #values = np.full_like(occupants, 0., dtype=float)
        is_high = np.random.random(occupants.shape) < self.probability_of_high_value
        values = np.where(occupants & is_high, self.n_fold * occupants, occupants/self.n_fold)

        return values

    def as_dict(self):
        self_dict = super().as_dict()
        self_dict.update({'n_fold_difference': self.n_fold})
        return self_dict

class LognormalTrait(Trait):
    distribution_type = 'lognormal'
    def __init__(self, mu, sigma, mean=None, variance=None) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

        self.mean = mean
        self.variance = variance

        self.distribution = scipy.stats.lognorm(s=sigma, scale=np.exp(mu))

    def draw_from_distribution(self, is_occupied):
        return np.where(is_occupied, self.distribution.rvs(is_occupied.shape), 0)

    def as_dict(self):
        self_dict = super().as_dict()
        self_dict.update({'mu': self.mu, 'variance': self.sigma})
        return self_dict

    def as_column(self):
        '''For when we log this trait in a pandas dataframe.'''
        assert(self.mean == 1)
        return ('variance', self.variance)

    @classmethod
    def from_natural_mean_variance(cls, mean, variance):
        sigma = np.sqrt(np.log(variance/(mean**2) + 1))
        mu = np.log(mean**2 / np.sqrt(variance + mean**2))

        return cls(mu, sigma, mean=mean, variance=variance)

    def __repr__(self) -> str:
        return f"LognormalTrait({self.as_dict()})"

if __name__ == '__main__':
    t = GammaTrait(mean=1.0, variance=1.0)
    t.plot(samples=1000)
    plt.show()

    #BiModalTrait(2.6).plot(samples=1000)
    #plt.show()