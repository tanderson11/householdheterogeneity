# Household Heterogeneity

This repository is used to simulate the spread of infection in small subpopulations where the susceptibility and infectiousness of individuals may differ.

Conact: Alison Hill [](alhill@jhmi.edu) and Thayer Anderson [](tanderson11@gmail.com).

Paper: [Quantifying individual-level heterogeneity in infectiousness and susceptibility through household studies](https://www.sciencedirect.com/science/article/pii/S1755436523000464).

## Project summary

Heterogeneity between individuals governs the spread of many pathogens. "Superspreading" (heterogeneity in infectiousness) is a key feature of transmission dynamics for SARS, MERS, smallpox, Ebola, tuberculosis, HIV, and SARS-CoV-2. To quantify superspreading, researchers commonly turn to contact tracing studies, which may be prone to bias because larger chains of infection are more likely to observed than smaller chains of infection. We propose that superspreading (and other heterogeneities, such as differences in susceptibility) might be more easily quantified by analyzing routine household studies that are often conducted to measure the risk one infected household member poses to their household contacts. Households are themselves important settings where pathogens spread (and therefore important settings for the containment of a spreading pathogen), and heterogeneity influences household transmission and affects the interpretation of the risk between household members.

This project connects the problem of quantifying heterogeneity to the problem of accurately understanding household transmission of a pathogen, and provides the means to solve these two problems jointly by a combined forward simulation + maximum likelihood estimation method. Our project has the following core aims:

1. To simulate the spread of infections within households (or other small, well-mixed populations) based on a flexibly specified model of transmission that can include heterogeneity.
2. To estimate the secondary attack risk (SAR; a measure of household transmission) as well the amount of heterogeneity from household final size data using maximum likelihood estimation based on the results of forward simulation of the specified model.
3. To analyze of household studies by estimating model parameters, measure confidence intervals, and create figures.

## Getting started (with Colab)

This package provides tools for studying the spread of Covid-19 (or another infection) in small, well-mixed subpopulations. For more information about the default infection model and the supported population structure, see the [Methods](https://www.sciencedirect.com/science/article/pii/S1755436523000464#sec2) of the related paper.

We have provided several IPython notebooks as self-contained laboratories for particular components of this project. These notebooks are designed with Google Colab in mind; Colab provides Google users access to cloud-based IPython instances.

To get started with project, open the notebook that you want to work with on github, click the "Open in Colab" button, and follow the notebook's instructions for making a clone of this repository in your Google drive.

Notebooks:
- `MinimalForwardSimulation.ipynb`: a notebook that provides the minimal code necessary to run a forward simulation and an example of how to plot infections in households with different amounts of heterogeneity.
- `ViolinsAndPowerCalc.ipynb`: a notebook for benchmarking the precision and bias of fits (ie: looking at the distribution of MLEs over many observations/simulations).
- `EmpiricalFits.ipynb`: a notebook for calculating the MLE and confidence intervals from a fit to a particular empirical data set.
- `MassForwardSimulation.ipynb`: a notebook used for simulating infections over a large region in parameter space.

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

### Getting started (locally)

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
- `model_inputs.py`: the forward simulation technology assumes that parameters will be given as two `Trait` objects that wrap around random variables (for the susceptibility and infectivity) and a `beta` (a probability/time of infection passing from infectious individual to susceptible individual). But often times,  we want to specify a bespoke scheme of parameters and conduct many simulations for different values of these parameters. For example, in the research associated with this project, we specified the values `s80` (fraction of susceptibility among the bottom 80% of susceptible individuals), `p80` (fraction of expected infections caused by individuals from the bottom 80% of infectivity), and `SAR` (the household secondary attack rate given the distributions of susceptibility and infectivity in the population). This file implements the `ModelInputs` abstract base class, which is used for converting from a custom scheme to the ordinary scheme. Objects of this class are initialized with some data, representing a custom way of providing parameter values, and have a method, `to_normal_inputs` that produces a dictionary of ordinary inputs to the foward simulation tool based on the custom data held in the object. The `S80_P80_SAR_Inputs` class converts an `s80`, `p80`, and `SAR` into the expected inputs for `recipes`.

### References

1. The Gillespie exact simulation technique.

- Original paper: Gillespie, D. T. Exact Stochastic Simulation of Coupled Chemical Reactions. The journal of physical chemistry 1977, 81 (25), 2340–2361.
- Skeleton of Python code: Justin Bois's and Michael Elowitz's [course notes](http://be150.caltech.edu/2019/handouts/12_stochastic_simulation_all_code.html) shared under a [Creative Commons Attribution License CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
