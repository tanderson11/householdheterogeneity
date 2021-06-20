import matplotlib.pyplot as plt
import numpy as np

class Trait:
    def __init__(self):
        pass

    def __call__(self, shape):
        return self.distribution(shape)
    
    def plot(self, samples=1000, **kwargs):
        output = np.array(self(samples))
        sample_mean = np.mean(output)
        sample_var = np.var(output)

        plt.hist(output, **kwargs)
        plt.title("{0}.\nSample mean {1:.2f} and sample var {2:.2f}".format(self, sample_mean, sample_var))
        plt.xlabel("relative magnitude of {0}".format(self.name))
        plt.ylabel("# people")

class ConstantTrait(Trait):
    def __init__(self, name, trait_value=1.0):
        self.name = name
        self.trait_type="constant"
        self.trait_value = trait_value
        self.distribution = lambda shape: np.ones(shape) * trait_value

    def __repr__(self):
        return "Constant trait named {0} with value {1:.2f}".format(self.name, self.trait_value)

class GammaTrait(Trait):
    def __init__(self, name, mean, variance):
        # copy fields
        self.name = name
        self.mean = mean
        self.variance = variance
        self.trait_type="gamma"

        # assign the underlying distribution
        if variance == 0:
            self.distribution = lambda shape: np.ones(shape) * mean
        else:
            self.distribution = lambda shape: np.random.gamma(mean**2/variance, scale=variance/mean, size=shape)

    def __repr__(self):
        return "Gamma distributed trait named {0} with mean {1:.2f} and variance {2:.2f}".format(self.name, self.mean, self.variance)
    

