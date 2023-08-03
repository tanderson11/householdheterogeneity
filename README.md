# covidhouseholds

## Getting started

This python package provides tools for studying the spread of Covid-19 (or another infection) in small, well-mixed subpopulations. For more information about the infection model and supported population structure, see the [[Methods link]]

There are also IPython notebooks contained within this repository. Each notebook uses the underlying python package, but is an endpoint that is not used by other notebooks or parts of the program. If you're only interested in one aspect of this work / problem space, then that notebook (ex. MinimalForwardSimulation.ipynb) will work as a self-contained environment for studying it. These notebooks are designed with Google Colab in mind; Colab provides Google users access to cloud-based IPython instances.

If you want to work with a notebook in Google Colab, rather than using `git clone` to make a local copy of this package on your computer, instead select the notebook that you want to work with, click the "Open in Colab" button, and follow the notebook's instructions for making a clone of this repository in your Google drive.

Notebooks:
- `MinimalForwardSimulation.ipynb`: a notebook that provides the minimal code necessary to run a forward simulation and an example of how to plot infections in households with different amounts of heterogeneity.
- `ViolinsAndPowerCalc.ipynb`: a notebook for benchmarking the precision and bias of fits (ie: looking at the distribution of MLEs over many observations/simulations).
- `EmpiricalFits.ipynb`: a notebook for calculating the MLE and confidence intervals from a fit to a particular empirical data set.
- `MassForwardSimulation.ipynb`: a notebook used for simulating infections over a large region in parameter space.


### Getting started locally

While it's recommended that you follow the steps above to run the notebooks in this module through Colab, there are reasons to download this package locally. For example, you might have a faster or more reliable local environment than Google provides. Or you might want to use interactive figures made possible through matplotlib's widget interface, which isn't available in Colab. Or you might want to run this package on a set of distributed computers. Or you might want to make changes to the code to better suit your needs.

In any case, to make a local version, take the following steps:

1. Open Terminal and clone this module (`git clone https://github.com/tanderson11/householdheterogeneity.git`).
2. Install poetry (https://python-poetry.org/docs/#installation), and then run the command `poetry install` to install all the necessary dependencies for this package.
3. From inside the `src/` directory, run the command `poetry run jupyter-lab` which will open the IPython server in your browser. From there, navigate to the notebook that interests you. (See Jupyter's documentation if you are new to using `IPython` notebooks: https://jupyter.readthedocs.io/en/latest/install.html)

## The python package

The provided notebooks are useful for engaging in specific tasks, but a lot of functionality lives in the underlying python module. To help access this functionality and to help you make changes to suit your needs, here's a brief list of important files and features.

### Files
- `gillespie_forward_simulation.py`: this file hosts the `gillespie_simulation` function, which simulates an initial state of infections in a group of households forward in time using an exact stochastic simulation approach derived from a modified Gillespie simulation. This function is programmed to use the python module `torch` to execute calculations on the GPU when possible. The `device` variable in `settings.py` determines if the CPU or GPU will be used.
- `recipes.py`: this file provides objects and functions to run common tasks. These include the `Model` class, which configures additional settings of the underlying SEIR model, with the `run_trials` method. The `PopulationStructure` class, which can instantiate individual `Populations` --- which represent a group of households whose individuals might vary with respect to susceptibility and infectivity. And the `Results` class, which wraps around a [Pandas](https://pandas.pydata.org/docs/user_guide/index.html#user-guide) `DataFrame` and helps aggregate the outcomes of many different forward simulations.
- `constants.py`: the `Constants` class lives here. A `Constants` objects specifies the mean and std of the time spent in the latent state and the infectious state. The variable `dt`, the time step used in forward simulation, is also specified for when non-exact forward simulation techniques are used. The prepackaged choices of constants are intended to represent facts about SARS-CoV2. You should create a new `Constants` object and point to it in `settings.py` if you intend to study a different disease.
- `traits.py`: this file implements the `Trait` class, which represents a quantity that varies between individuals in a population. The two traits of interest in our model are the susceptibility and infectivity of individuals. Various kinds of traits, such as `ConstantTrait`s, `GammaTrait`s, and `LognormalTrait`s are implemented as subclasses of the abstract base class. These `Trait` objects wrap around a `draw_from_distribution` method that draws samples from the underlying random variable. The `Trait` objects can be passed to various methods to specify the population distributions of traits.
- `utilities.py`: there are common tasks that must be performed (1) before we simulate infections and (2) as we process the data from completed simulations. The file `utilities` groups together some helper functions and objects associated with these tasks.
- `model_inputs.py`: the forward simulation technology assumes that parameters will be given as two `Trait` objects that wrap around random variables (for the susceptibility and infectivity) and a `beta` (a probability/time of infection passing from infectious individual to susceptible individual). But often times,  we want to specify a bespoke scheme of parameters and conduct many simulations for different values of these parameters. For example, in the research associated with this project, I specified the values `s80` (fraction of susceptibility among the bottom 80% of susceptible individuals), `p80` (fraction of expected infections caused by individuals from the bottom 80% of infectivity), and `SAR` (the household secondary attack rate given the distributions of susceptibility and infectivity in the population). This file implements the `ModelInputs` abstract base class, which is used for converting from a custom scheme to the ordinary scheme. Objects of this class are initialized with some data, representing a custom way of providing parameter values, and have a method, `to_normal_inputs` that produces a dictionary of ordinary inputs to the foward simulation tool based on the custom data held in the object. The `S80_P80_SAR_Inputs` class converts an `s80`, `p80`, and `SAR` into the expected inputs for `recipes`.

### Figures

| Figure  | Created by |
| ------------- | ------------- |
| Fig 1b: visual representation of `s80` and `p80`  | `notebooks/TraitFigures.ipynb`  |
| Fig 2: effects of heterogeneity  | `notebooks/MinimalForwardSimulation.ipynb`  |
| Fig 3 --- left: contours of likelihood | `src/plot_testing.py` and `src/fancy_plotting.py`|
| Fig 3 --- right: violins of MLEs  | `notebooks/ViolinsAndPowerCalc.ipynb`  |
| Fig 4b and c: best fits for empirical data  | `notebooks/EmpiricalFits.ipynb`  |
| Table 1: best fits for empirical data  | `notebooks/EmpiricalFits.ipynb`  |
| Table 2: power to detect intervention  | `notebooks/ViolinsAndPowerCalc.ipynb`  |

### References

1. The Gillespie exact simulation technique.

- Original paper: Gillespie, D. T. Exact Stochastic Simulation of Coupled Chemical Reactions. The journal of physical chemistry 1977, 81 (25), 2340–2361.
- Skeleton of Python code: Justin Bois's and Michael Elowitz's [course notes](http://be150.caltech.edu/2019/handouts/12_stochastic_simulation_all_code.html) shared under a [Creative Commons Attribution License CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
