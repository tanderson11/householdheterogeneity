import torch_forward_simulation as tf
import constants
import matplotlib.pyplot as plt

s = constants.EXPOSED_STATE
entrants = np.zeros(1000)
lengths = tf.torch_state_length_sampler(s, entrants)

plt.hist(lengths)
plt.show()