import matplotlib.pyplot as plt
import numpy as np

class Trait:
    def __init__(self):
        pass

    def __call__(self, shape):
        return self.distribution(shape)
    
    def plot(self, samples=1000):
        output = np.array(self(10000))
        sample_mean = np.mean(output)
        sample_var = np.var(output)

        plt.hist(output)
        plt.title("Histogram of {0}. Sample mean {0:.2f} and sample var {1:.2f}".format(self, sample_mean, sample_var))
        plt.xlabel("relative magnitude of trait")
        plt.ylabel("# people")
        plt.show()

class ConstantTrait(Trait):
    def __init__(self, trait_value=1.0):
        self.trait_value = trait_value
        self.distribution = return lambda shape: np.ones(shape) * trait_value

    def __repr__(self):
        return "Constant trait with value {0:.2f}".format(self.trait_value)

class GammaTrait(Trait):
    def __init__(self, mean, variance):
        # copy fields
        self.mean = mean
        self.variance = variance

        # assign the underlying distribution
        if variance == 0:
            self.distribution = return lambda shape: np.ones(shape) * mean
        else:
            self.distribution = lambda shape: np.random.gamma(mean**2/variance, scale=variance/mean, size=shape)

    def __repr__(self):
        return "Gamma distributed trait with mean {0:.2f} and variance {1:.2f}".format(self.mean, self.variance)
    

