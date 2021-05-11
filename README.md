# covid_households

## Getting started

This python package and set of IPython notebooks provide tools for studying the spread of Covid-19 (or another infection) in small, well-mixed subpopulations. For more information about the infection model and supported population structure, see the Infection Model heading.

IPython notebooks are a beloved tool for scientific uses of python because they designed to be used interactively and can incorporate written text, LaTeX, and figures.

Each notebook uses the underlying python package, but is an endpoint that is not used by other notebooks or parts of the program. If you're only interested in one aspect of this work / problem space (ex. vaccination), then that notebook (ex. VaccineLab.ipynb) will work as a self-contained environment for studying it. These notebooks are designed with Google Colab in mind; Colab provides Google users access to cloud-based IPython instances.

Rather than using `git clone` to make a local copy of this package on your computer, instead select the notebook that you want to work with, click the "Open in Colab" button, and follow the notebook's instructions for making a clone of this repository in your Google drive.

Notebooks:
- `VaccineLab.ipynb`: a notebook for developing clinical studies of vaccine effects.
- `ParameterInferenceLaboratory.ipynb`: a notebook for trying to back out a vaccine's effect on true model parameters.
- `SuperspreadingLaboratory.ipynb`: a notebook for studying trait-based variance in individuals' susceptibilities and infectivities.

### Getting started locally

While it's recommended that you follow the steps above to run the notebooks in this module through Colab, there are reasons to download this package locally. For example, you might have a faster or more reliable local environment than Google provides. Or, you might want to use interactive figures made possible through matplotlib's widget interface, which isn't available in Colab. 

In any case, to make a local version, open Terminal and clone this module (`git clone https://github.com/tanderson11/covid_households.git`). Then install Jupyter lab, (see: https://jupyter.readthedocs.io/en/latest/install.html).

From inside the `covid_households/` directory, run the command `jupyter-lab` which will open the IPython server in your browser. From there, navigate to the notebook that interests you.
