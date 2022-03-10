# covid_households

## Getting started

This python package provides tools for studying the spread of Covid-19 (or another infection) in small, well-mixed subpopulations. For more information about the infection model and supported population structure, see the [[Methods link]]

There are also IPython notebooks contained within this repository. Each notebook uses the underlying python package, but is an endpoint that is not used by other notebooks or parts of the program. If you're only interested in one aspect of this work / problem space, then that notebook (ex. VaccineLab.ipynb) will work as a self-contained environment for studying it. These notebooks are designed with Google Colab in mind; Colab provides Google users access to cloud-based IPython instances.

If you want to work with a notebook in Google Colab, rather than using `git clone` to make a local copy of this package on your computer, instead select the notebook that you want to work with, click the "Open in Colab" button, and follow the notebook's instructions for making a clone of this repository in your Google drive.

Notebooks:
- `Minimal Forward Simulation.ipynb`: a notebook that provides the minimal code necessary to run a forward simulation.

### Getting started locally

While it's recommended that you follow the steps above to run the notebooks in this module through Colab, there are reasons to download this package locally. For example, you might have a faster or more reliable local environment than Google provides. Or, you might want to use interactive figures made possible through matplotlib's widget interface, which isn't available in Colab. Or, you might want to run this package on a set of distributed computers. 

In any case, to make a local version, open Terminal and clone this module (`git clone https://github.com/tanderson11/covid_households.git`). Then install Jupyter lab, (see: https://jupyter.readthedocs.io/en/latest/install.html).

From inside the `covid_households/` directory, run the command `jupyter-lab` which will open the IPython server in your browser. From there, navigate to the notebook that interests you.

## The python package

The provided notebooks are useful for engaging in specific tasks, but a lot of functionality lives in the underlying python module. To help access this functionality and to make changes to suit your needs, here's a brief list of important files and features.

### Files
- `torch_forward_simulation.py`: this file hosts the `torch_forward_time` function, which simulates an initial state of infections in a group of households forward in time using a stochastic SEIR model. The prefix `torch` refers to the fact that this code uses the python module `torch` to execute calculations on the GPU when possible.
- `recipes.py`: this file provides objects and functions to run common tasks. These include the `Model` class, which configures additional settings of the underlying SEIR model, with the `run_trials` method. The `PopulationStructure` class, which can instantiate individual `Populations` --- which represent a group of households whose individuals might vary with respect to susceptibility and infectivity. And the `Results` class, which wraps around a [Pandas](https://pandas.pydata.org/docs/user_guide/index.html#user-guide) `DataFrame` and helps aggregate the outcomes of many different forward simulations.
- `constants.py`: the `Constants` class lives here. A `Constants` objects specifies the mean and std of the time spent in the latent state and the infectious state as well as `dt`, the time step used in forward simulation. The prepackaged choices of constants are intended to represent facts about SARS-CoV2. You should create a new `Constants` object and point to it in `settings.py` if you intend to study a different disease.