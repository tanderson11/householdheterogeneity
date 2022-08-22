# Initialization
from typing import NamedTuple
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import operator

import likelihood
import utilities
import model_inputs
import traits
import recipes

plt.rcParams['pdf.fonttype']=42
EMPIRICAL_TRIAL_ID = -1
pretty_names = {'sus_mass': 'sus mass', 'inf_mass': 'inf mass', 'hsar': 'hsar', 'SAR': 'SAR', 's80': 's80', 'p80': 'p80'}

class SelectedPoint:
    def __init__(self, parameter_coordinates, color, is_baseline=False):
        self.parameter_coordinates = parameter_coordinates
        self.color = color

        self.is_baseline = is_baseline

    def __str__(self):
        return "Selected point at {0}".format(self.parameter_coordinates)

def find_most_likely(logl_df, keys):
    width = logl_df.unstack().columns.size # verify that this still works

    idx = logl_df.reset_index()["logl"].argmax() # idiom for finding position of largest value
    #import pdb; pdb.set_trace()
    x_min = idx % width
    y_min = idx // width
    baseline_1 = logl_df.unstack().reset_index()[keys[0]].to_list()[y_min]
    baseline_2 = logl_df.unstack().columns.to_list()[x_min] #columns associated with key2, index with key1
    return baseline_1, baseline_2

class InteractiveFigure:
    @staticmethod
    def restrict_to_dimensions_of_interest(df, unspoken_parameters):
        '''Takes a dataframe with arbitrarily many index dimensions, and reduces it to the two relevant dimensions for plotting.
        
        Unspoken parameters is a dictionary of parameters-->values for each of the dimensions that /won't/ be plotted in order to find the right 2D slice to plot.'''
        if unspoken_parameters is None:
            return df

        for k,v in unspoken_parameters.items():
            try:
                df = df[df.index.get_level_values(k) == v]
            except KeyError:
                pass
        # drop the unnecessary index levels
        df.index = df.index.droplevel(list(unspoken_parameters.keys()))
        return df

    def get_nD_sample_df(self):
        nD_sample_df = self.sample_df.copy()
        # rebuild each of the layers that were removed to get to a 2D plotting sample DF
        for k,v in self.unspoken_parameters.items():
            nD_sample_df = pd.concat({v: self.sample_df}, names=[k])
        return nD_sample_df

    def solve_likelihood(self, frequency_df, sample_df, keys, **likelihood_kwargs):
        return likelihood.logl_from_frequencies_and_counts(frequency_df, sample_df, keys, **likelihood_kwargs)


    def __init__(
            self,
            keys,
            subplots,
            frequency_df,
            full_sample_df=None,
            is_empirical=False,
            baseline_values=None,
            empirical_path=False,
            simulation_population=None,
            simulation_trials=1,
            unspoken_parameters={},
            patch_palette='sequential',
            figsize=(15,7),
            baseline_color='red'):

        self.figsize = figsize
        self.is_empirical = is_empirical

        assert frequency_df is not None
        # a copy that will be reduced to the appropriate 2 dimensions for plotting
        self.frequency_df = frequency_df
        # a copy of the maximum dimension used for some probability plots
        self.nD_frequency_df = frequency_df.copy()
        # if we want to fix a parameter outside of our 2D slice,
        # we need to ensure its value is constant in the frequency_df
        self.unspoken_parameters = unspoken_parameters
        self.frequency_df = self.restrict_to_dimensions_of_interest(frequency_df, unspoken_parameters)

        self.reset_freqs = self.frequency_df.reset_index()
        self.keys = keys
        self.key1, self.key2 = keys

        self.full_sample_df = full_sample_df
        self.simulation_population = simulation_population
        if is_empirical:
            self.trials = 1
        else:
            self.trials = simulation_trials

        self.logl_df = None
        if self.is_empirical:
            self.logl_df = self.solve_likelihood(self.frequency_df, full_sample_df, keys)
            assert baseline_values is None, "baseline specified but empirical dataset selected"
            # if empirical, we want to set the baseline to be at the point of maximum likelihood so we can display boostrapped points:

            baseline_1, baseline_2 = find_most_likely(self.logl_df, keys)
            # add labels to the empirical df so its position can be identified
            full_sample_df[self.key1] = baseline_1
            full_sample_df[self.key2] = baseline_2
            baseline_coordinates = {self.key1:baseline_1, self.key2:baseline_2}

            self.sample_df = full_sample_df
        else:
            # -- Selecting the location of the baseline point --
            if baseline_values is None:
                baseline_coordinates = {self.key1:self.reset_freqs[self.key1].iloc[0], self.key2:self.reset_freqs[self.key2].iloc[0]}
            else:
                baseline_coordinates = {self.key1:baseline_values[0], self.key2:baseline_values[1]}
            self.sample_df = self.baseline_at_point(baseline_coordinates)
            print("SAMPLE_DF:\n", self.sample_df)

            if self.full_sample_df is None:
                self.logl_df = self.solve_likelihood(self.frequency_df, self.sample_df, keys)
            else:
                self.logl_df = self.solve_likelihood(self.frequency_df, self.full_sample_df, keys)

        self.sample_model_dict = {}

        self.baseline_color = baseline_color
        self.baseline_point = SelectedPoint(baseline_coordinates, self.baseline_color, is_baseline=True)

        self.selected_points = [self.baseline_point]

        # --
        if patch_palette == 'categorical':
            self.available_colors = ["red", "blue", "green", "violet"]
        elif patch_palette == 'sequential':
            cmap = plt.get_cmap("Oranges")
            self.available_colors = [cmap(x) for x in np.linspace(0.2, 0.7, 4)]
            #self.available_colors = [cmap(x) for x in np.linspace(0.4, 0.9, 2)]

        self.subplots = subplots
        #import pdb; pdb.set_trace()
        self.make_figure()

    @staticmethod
    def simulate_at_point(keys, sizes, trials, unspoken_parameters=None, model=recipes.Model()):
        # keys = dictionary {key: value}
        beta = None
        hsar = keys.get('hsar', None)
        hsar = hsar if hsar is not None else keys.get('SAR', None)
        if hsar is None: beta = keys['beta']

        parameterization = model_inputs.parameterization_by_keys[frozenset(keys.keys())]
        params = parameterization(**keys)

        df = model.run_trials(**params.to_normal_inputs(), trials=trials, sizes=sizes, as_counts=True)
        for k,v in keys.items():
            df[k] = np.float("{0:.2f}".format(v))
        #import pdb; pdb.set_trace()
        df = df.reset_index().set_index(list(keys.keys()) + ['size', 'infections'])

        if unspoken_parameters is not None:
            df = df.set_index(df.index.droplevel(list(unspoken_parameters.keys())))

        class SimulationResult(NamedTuple):
            df: pd.DataFrame
            beta: float
        return SimulationResult(df, beta)

    def baseline_at_point(self, baseline_coordinates):
        if self.full_sample_df is None:
            # in this case we have to simulate
            keys = {**self.unspoken_parameters, **baseline_coordinates}
            baseline_df = self.simulate_at_point(keys, self.simulation_population, self.trials, self.unspoken_parameters).df
        else:
            baseline_df = self.full_sample_df[(self.full_sample_df[self.key1] == baseline_coordinates[self.key1]) & (self.full_sample_df[self.key2] == baseline_coordinates[self.key2])]

        try: # for legacy dfs that don't always include a trial column
            print(baseline_df["trial"])
        except KeyError:
            baseline_df["trial"] = 0

        return baseline_df

    def toggle(self, parameter_coordinates, **kwargs):
        i_to_remove = len(self.selected_points) # don't remove anything; add to the list
        for i,p in enumerate(self.selected_points):
            if p.is_baseline == False:
                #same_point = [p.parameter_coordinates[k] == v for k,v in parameter_coordinates.items()]
                if parameter_coordinates[self.key1] == p.parameter_coordinates[self.key1] and parameter_coordinates[self.key2] == p.parameter_coordinates[self.key2]: # if the point already exists
                    i_to_remove = i
        if i_to_remove == len(self.selected_points):
            self.select(parameter_coordinates, **kwargs)
        else:
            removed = self.selected_points.pop(i_to_remove)

            for ax in self.ax.ravel():
                if ax.associated_subfigure.has_patches == True:
                    color = ax.associated_subfigure.remove_patch(removed) #remove the patch from each subfigure that uses patches

            self.available_colors.append(color)

        self.draw_after_toggle()

    def draw_after_toggle(self):
        for ax in self.ax.ravel():
            sf = ax.associated_subfigure
            if sf.has_patches: # subfigures that have patches need to update the patches
                sf.draw_patches(self)
            elif isinstance(sf, SelectionDependentSubfigure): # subfigures that depend on the selection need to redraw
                sf.draw(self)
        self.fig.canvas.blit(self.fig.bbox)

    def select(self, parameter_coordinates, **kwargs):
        try:
            color = self.available_colors.pop()
        except IndexError:
            print("Error. Tried to select point but no colors are available.")
            return False

        point = SelectedPoint(parameter_coordinates, color, **kwargs)
        self.selected_points.append(point)

        self.draw_after_toggle()
        return point

    def make_figure(self):
        # -- Associating Subfigure objects with axes objects --
        self.fig, self.ax = plt.subplots(*self.subplots.shape, figsize=self.figsize)
        if self.subplots.shape == (1,1):
            self.ax = np.array([self.ax])
        for subfigure_type, _ax in zip(self.subplots.ravel(), self.ax.ravel()):
            _ax.associated_subfigure = subfigure_factory(subfigure_type, _ax, self) # adds a subfigure parameter to each plt axes instance

        # -- Drawing --
        self.draw()
        plt.tight_layout()

        self.new_baseline_flag = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        slide_name = f"slide-ss{self.simulation_population}"
        for k,v in self.baseline_point.parameter_coordinates.items():
            slide_name += f'-{k[0]}{v}'
        #plt.savefig(f'mass_produced/{slide_name}.png', dpi=400),# bbox_inches='tight')
        plt.show()

    def reset_baseline(self, parameter_coordinates):
        print("Resetting the baseline point")
        print(parameter_coordinates[self.key1], parameter_coordinates[self.key2])
        self.sample_df = self.baseline_at_point(parameter_coordinates)
        if self.full_sample_df is None:
            self.logl_df = self.solve_likelihood(self.frequency_df, self.sample_df, [self.key1, self.key2])
        self.baseline_point = SelectedPoint(parameter_coordinates, self.baseline_color, is_baseline=True)

        self.selected_points[0] = self.baseline_point

        plt.close()
        self.make_figure()

    def draw(self):
        for ax in self.ax.ravel():
            ax.associated_subfigure.draw(self) # draws each subfigure

    def on_click(self, event):
        print(event.inaxes.associated_subfigure)
        if self.new_baseline_flag:
            click_type = "reset baseline"
        else:
            click_type = "select"

        event.inaxes.associated_subfigure.click(self, event.xdata, event.ydata, click_type) # pass clicks to the subfigure objects

    def on_key(self, event):
        if event.key == " ":
            self.new_baseline_flag = True
        elif event.key == "x":
            self.fig.savefig('./batched_figs/contours/interactive_figure.pdf', dpi=400)

def subfigure_factory(plot_type, ax, interactive):
    keys = (interactive.key1, interactive.key2)
    trial = 0
    #import pdb; pdb.set_trace()
    logl_df = interactive.logl_df.loc[trial]

    if plot_type == 'logl heatmap':
        # new code with the full logl df
        if interactive.is_empirical:
            #import pdb; pdb.set_trace()
            sizes=interactive.sample_df.groupby("size")["model"].count().to_dict() # idiom for getting the # hh at different sizes
            title = ""
            #title = "Log likelihood of observing empirical dataset\nsizes={}\n fixed values={}".format(sizes, interactive.fixed_values)
        else:
            #if interactive.sample_model_dict["importation_rate"] == 0:
            if True:
                title = ""
                #title = "Log likelihood of observing {0} baseline (size={1}) versus {2} and {3}\n seeding={4}, daily importation=0".format(interactive.baseline_color, sum(interactive.baseline_sizes.values()), interactive.key1, interactive.key2, interactive.sample_model_dict["seeding"]["name"])
            else:
                title = ""
                #title = "Log likelihood of observing {0} baseline (size={1}) versus {2} and {3}\n seeding={4}, daily importation={4:.4f}, and duration={5}".format(interactive.baseline_color, sum(interactive.baseline_sizes.values()), interactive.key1, interactive.key2, interactive.sample_model_dict["seeding"]["name"], interactive.sample_model_dict["importation_rate"], interactive.sample_model_dict["duration"])

        subfigure = Heatmap(ax, logl_df, title, scatter_stars=True)

    elif plot_type == 'confidence heatmap':
        title = ""
        subfigure = InOrOutConfidenceIntervalHeatmap(ax, keys, logl_df, title)

    elif plot_type == 'many confidence heatmap':
        title = "Membership in cumulative probability density contours"
        subfigure = ManyMasksConfidenceHeatmap(ax, keys, logl_df, title)

    elif plot_type == 'logl contour plot':
        subfigure = ContourPlot(ax, keys, logl_df, "Contours of loglikelihood with default levels", color_label="logl")

    elif 'probability contour plot' in plot_type:
        kwargs = {'scatter_stars':False}
        if '2D slice' in plot_type:
            full_keys = list(interactive.unspoken_parameters.keys()) + interactive.keys
            logl_df = interactive.solve_likelihood(
                interactive.nD_frequency_df,
                interactive.get_nD_sample_df(),
                full_keys,
            )
            prob_df = utilities.normalize_logl_as_probability(logl_df)
            max_prob = prob_df.max()
            print(f"Global probability maximum: {max_prob}")
            print(logl_df.idxmax())

            # now we have to reduce the dimensions again to the ones that are appropriate for plotting
            dropped_parameters = interactive.unspoken_parameters
            prob_df = interactive.restrict_to_dimensions_of_interest(prob_df, dropped_parameters)
            prob_df = prob_df.loc[trial]
            kwargs.update({'vmin':0.0, 'vmax':max_prob})
        elif '3D slice free parameter' in plot_type:
            full_keys = list(interactive.unspoken_parameters.keys()) + interactive.keys
            logl_df = interactive.solve_likelihood(
                interactive.nD_frequency_df,
                interactive.get_nD_sample_df(),
                full_keys,
            )
            # now we sum over the free parameter at each point with the other parameters fixed
            # so that the 2D plot will capture the uncertainties caused by lack of clarity about the free parameter
            prob_df = utilities.normalize_logl_as_probability(logl_df)
            prob_df = prob_df.groupby(['trial'] + interactive.keys).sum()
            prob_df = prob_df.loc[trial]

            max_prob = prob_df.max()
            print(f"Global probability maximum: {max_prob}")
            print(logl_df.idxmax())
            #import pdb; pdb.set_trace()
            #if logl_df.idxmax()[3] < 0.02:
            #    import pdb; pdb.set_trace()

            kwargs.update({'vmin':0.0, 'vmax':max_prob})
        elif '2D only' in plot_type:
            prob_df = utilities.normalize_logl_as_probability(logl_df)
            #import pdb; pdb.set_trace()
        else:
            raise Exception(f"unknown plot type {plot_type}")
        
        subfigure = ProbabilityContourPlot(ax, keys, prob_df, "", color_label="probability", **kwargs)

    elif plot_type == 'infection histograms':
        subfigure = InfectionHistogram(ax, keys)

    elif plot_type == 'trait histograms':
        subfigure = TraitHistograms(ax, keys)
    else:
        raise Exception(f"unknown plot type {plot_type}")

    return subfigure

class Subfigure:
    def __init__(self, ax, keys, **kwargs):
        self.ax = ax
        self.has_patches = False
        self.key1, self.key2 = keys

        self.kwargs = kwargs

    def draw(self, interactive):
        plt.sca(self.ax)
        plt.cla()

class SelectionDependentSubfigure(Subfigure):
    def frequency_df_at_point(self, interactive, point, drop_level=None):
        labels = [point.parameter_coordinates[self.key1], point.parameter_coordinates[self.key2]] # ie we could use .values if the dict were sorted, TK
        if point.is_baseline: # the baseline dataframe is the data at the baseline point, so we pass it along as is
            restriction = interactive.sample_df.copy()
            # choose only the first trials
            #import pdb; pdb.set_trace()
            restriction = restriction[restriction['trial'] == 0]
            # Only works if we have one size of household in our data
            restriction['freq'] = restriction['count'] / restriction['count'].sum()
            restriction = restriction['freq']
        else: # otherwise, we select only the comparison distributions at the relevant points
            masks = [interactive.reset_freqs[self.key1] == labels[0], interactive.reset_freqs[self.key2] == labels[1]] # is true precisely where our two keys are at the chosen parameter values
            mask = functools.reduce(operator.and_, masks)
            mask.index = interactive.frequency_df.index
            restriction = interactive.frequency_df[mask]
            restriction.name = 'freq'
            #import pdb; pdb.set_trace()
        if drop_level:
            restriction.index = restriction.index.droplevel(drop_level)
            if self.key1 == drop_level:
                labels = labels[1:]
            if self.key2 == drop_level:
                labels = labels[:1]
            #pass
        return labels, restriction

class TraitHistograms(SelectionDependentSubfigure):
    def __init__(self, ax, keys):
        SelectionDependentSubfigure.__init__(self, ax, keys)

    def draw(self, interactive):
        SelectionDependentSubfigure.draw(self, interactive)
        possible_traits = ["sus_var", "inf_var", "sus_mass", "inf_mass", "s80", "p80"]
        if self.key1 in possible_traits and self.key2 in possible_traits:
            print("WARNING: both keys are traits. Graphing only key1")
        if self.key1 in possible_traits:
            graph_key = self.key1
        elif self.key2 in possible_traits:
            graph_key = self.key2
        else:
            return None

        #bins = np.linspace(0., 6., 30)
        bins = np.linspace(0., 7., 20)
        colors = []
        outputs = []
        labels = []
        for p in interactive.selected_points:
            #trait=traits.GammaTrait("{0}".format(graph_key), mean=1.0, variance=p.parameter_coordinates[graph_key])
            parameter_value = p.parameter_coordinates[graph_key]
            if graph_key == 'sus_var' or graph_key == 'inf_var':
                trait=traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=parameter_value)
            elif graph_key == 's80' or graph_key == 'p80':
                variance = utilities.lognormal_p80_solve(parameter_value).x[0]
                trait=traits.LognormalTrait.from_natural_mean_variance(mean=1.0, variance=variance)
            color = p.color
            #import pdb; pdb.set_trace()
            if p.is_baseline:
                alt_cmap = plt.get_cmap("Oranges")
                color = [alt_cmap(x) for x in np.linspace(0.2, 0.7, 4)][2]
            colors.append(color)
            samples = 40000
            shaped_array = np.full((samples,), True)
            output = np.array(trait(shaped_array))
            outputs.append(output)
            labels.append(p.parameter_coordinates[interactive.key1])
            #ax = trait.plot(samples=10000, color=color, bins=bins, normed=1, histtype='bar')

        #ax = plt.hist(outputs, histtype='bar', bins=bins, color=colors, label=labels)
        ax = plt.hist(outputs, histtype='stepfilled',bins=bins, color=colors, label=labels, alpha=0.8, edgecolor='black')
        self.ax.legend(prop={'size': 12})
        plt.ylabel("# people")
        plt.xlabel("relative magnitude")
        #plt.title("Gamma distributed {0}".format(graph_key))

class InfectionHistogram(SelectionDependentSubfigure):
    def __init__(self, ax, keys, drop_level=None, highlight_special_value=True):
        SelectionDependentSubfigure.__init__(self, ax, keys)
        self.drop_level = drop_level
        self.drop_level = 'SAR'
        self.highlight_special_value = highlight_special_value

    def draw(self, interactive):
        SelectionDependentSubfigure.draw(self, interactive)

        color_dict = {}
        average_dict = {}
        represented_coordinates = set()
        restrictions = []
        #import pdb; pdb.set_trace()
        for p in interactive.selected_points:
            labels, restriction = self.frequency_df_at_point(interactive, p, drop_level=self.drop_level)
            if len(labels) > 1:
                labels = tuple(labels)
            else:
                labels = labels[0]
            color_dict[labels] = p.color
            # SPECIAL
            if self.highlight_special_value:
                if p.parameter_coordinates['SAR'] == 0.25:
                    alt_cmap = plt.get_cmap("Oranges")
                    color_dict[0.2] = [alt_cmap(x) for x in np.linspace(0.2, 0.7, 4)][2]

            reset_restriction = restriction.reset_index()
            average_dict[labels] = (reset_restriction['infections'] * reset_restriction['freq']).sum()
            if p.is_baseline:
                baseline_restriction = restriction
                baseline_coordinates = labels
            else:
                restrictions.append(restriction)
                represented_coordinates.add(labels)
        # only display the baseline if we haven't separately chosen to view that cell
        if baseline_coordinates not in represented_coordinates:
            restrictions.append(baseline_restriction)

        restricted_df = pd.concat(restrictions)
        keys = [interactive.key1, interactive.key2]
        if self.drop_level:
            try:
                keys.remove(self.drop_level)
            except ValueError:
                pass
        #import pdb; pdb.set_trace()
        ax = utilities.simple_bar_chart(restricted_df, key=keys, color=color_dict, title="", ax=self.ax, ylabel="fraction observed", xlabel="household size, observed number infected")
        #ax.legend(title='top 20% portion')
        #ax.legend(title='household secondary attack rate')
        offsets = (0.04*i for i in range(0, len(restrictions)))
        #for labels, average in average_dict.items():
        #    offset = next(offsets)
        #    ax.text(0.25, 0.37-offset, f"average={average:.2f}", size=12, color=color_dict[labels])

class OnAxesSubfigure(Subfigure):
    '''A class that holds a figure and can manage coordinates, plot patches, and other aspects of interactive selection.

    Needs to be specialized to different frameworks via the subclasses OnMatplotlibAxes and OnSeabornAxes, which handle the different ways of converting coordinates to xy.'''
    def __init__(self, ax, keys, **plotting_kwargs):
        Subfigure.__init__(self, ax, keys, **plotting_kwargs)
        self.has_patches = True
        self.patches = {}

    def click(self, interactive, event_x, event_y, click_type):
        parameter_coordinates = self.click_to_coordinates(event_x, event_y)

        print(parameter_coordinates)
        if click_type == "select":
            interactive.toggle(parameter_coordinates)
        elif click_type == "reset baseline":
            interactive.reset_baseline(parameter_coordinates)

    def draw_patches(self, interactive):
        for point in interactive.selected_points:
            x,y = self.parameter_coordinates_to_xy(point.parameter_coordinates)

            try:
                self.patches[(x,y)]
            except KeyError:
                patch = self.draw_patch(point, x,y)
                self.patches[(x,y)] = patch

    def scatter_point_estimates(self, interactive, **kwargs):
        width = self.stacked_df.unstack().columns.size

        x_mins = []
        y_mins = []
        colors = []
        for trial,_logl_df in interactive.logl_df.groupby("trial"): # go to the baseline coordinates, then look at all the trials
            idx = _logl_df.reset_index()["logl"].argmax() # idiom for finding position of largest value / not 100% sure all the reseting etc. is necessary
            x_min_index = idx % width
            y_min_index = idx // width
            x_min, y_min = self.index_to_xy(x_min_index, y_min_index)

            x_min = x_min + self.center_offset# + (np.random.rand(1)/2 - 0.25)*self.patches_x_scale # the middle of the cell with random fuzzing so there's no overlap
            y_min = y_min + self.center_offset# + (np.random.rand(1)/2 - 0.25)*self.patches_x_scale

            x_mins.append(x_min)
            y_mins.append(y_min)

            if trial == EMPIRICAL_TRIAL_ID:
                colors.append("red")
                #colors.append("orange")
            else:
                colors.append("red")
                #colors.append("orange")

        self.ax.scatter(x_mins, y_mins, s=4, c=colors, **kwargs)

    def remove_patch(self, point):
        x,y = self.parameter_coordinates_to_xy(point.parameter_coordinates)
        self.patches.pop((x,y)).remove()

        return point.color

class OnMatplotlibAxes(OnAxesSubfigure):
    def __init__(self, ax, keys, **plotting_kwargs):
        OnAxesSubfigure.__init__(self, ax, keys, **plotting_kwargs)
        self.center_offset = 0.0

        self.x_grid_values = self.df.columns.to_numpy()
        self.y_grid_values = self.df.index.to_numpy()

        self.patches_do_offset, self.patches_x_scale, self.patches_y_scale = 0, (self.x_grid_values[-1]-self.x_grid_values[0]) / (len(self.x_grid_values)), (self.y_grid_values[-1]-self.y_grid_values[0]) / (len(self.y_grid_values))

    def click_to_coordinates(self, event_x, event_y):
        parameter_coordinates = self.xy_to_parameter_coordinates(event_x, event_y)

        return parameter_coordinates

    def index_to_xy(self, x_index, y_index):
        return self.x_grid_values[x_index], self.y_grid_values[y_index]

    def xy_to_parameter_coordinates(self, x, y): # is this reversed in terms of x and y relative to seaborn?
        x = x+1*self.patches_x_scale
        y = y+1*self.patches_y_scale

        # we need to snap continuous x and y values onto the grid defined by the dataframe
        x_where_lower = np.ma.MaskedArray(self.x_grid_values, self.x_grid_values > x) # we want to go to the closest point that's up and to the right so things snap to grid in a consistent way across figures
        #print(self.x_grid_values < x, x_where_lower, x-x_where_lower)
        x_idx = (x-x_where_lower).argmin() # nearest value where lower

        y_where_lower = np.ma.MaskedArray(self.y_grid_values, self.y_grid_values > y)
        #print(self.y_grid_values > y, y_where_lower, y-y_where_lower)
        y_idx = (y-y_where_lower).argmin()


        parameter_coordinates = {self.key1:self.y_grid_values[y_idx], self.key2:self.x_grid_values[x_idx]}
        #print(parameter_coordinates)
        #import pdb; pdb.set_trace()
        return parameter_coordinates

    def parameter_coordinates_to_xy(self, parameter_coordinates):
        # this is easy because the xy on the matplotlib axes just are parameter coordinates
        x,y = parameter_coordinates[self.key2], parameter_coordinates[self.key1] # key order inverted as always
        #import pdb; pdb.set_trace()
        return x,y

    def draw_patch(self, point, x,y):
        # refactor this so it calls a draw_patch method of the subfigure

        if point.is_baseline:
            patch = self.ax.add_patch(patches.Ellipse((x,y), 0.5*self.patches_x_scale, 0.5*self.patches_y_scale, fill=False, edgecolor=point.color, lw=1))
            #patch = self.ax.add_patch(patches.Ellipse((x+0.15*self.patches_x_scale,y+0.15*self.patches_y_scale), 0.2*self.patches_x_scale, 0.2*self.patches_y_scale, fill=False, edgecolor=point.color, lw=1))
            #patch = self.ax.add_patch(patches.Circle((x+0.05*self.patches_x_scale,y+0.05*self.patches_y_scale), 0.05*self.patches_x_scale,fill=True, edgecolor=point.color, lw=2))
        else:
            patch = self.ax.add_patch(patches.Rectangle((x-0.5*self.patches_x_scale,y-0.5*self.patches_y_scale), 1*self.patches_x_scale, 1*self.patches_y_scale, fill=False, edgecolor=point.color, lw=2))

        return patch

class OnSeabornAxes(OnAxesSubfigure):
    def __init__(self, ax, keys):
        self.center_offset = 0.5
        self.patches_x_scale = 1.
        self.patches_y_scale = 1.
        OnAxesSubfigure.__init__(self, ax, keys)

    def click_to_coordinates(self, event_x, event_y):
        x = np.floor(event_x)
        y = np.floor(event_y)

        parameter_coordinates = self.xy_to_parameter_coordinates(x, y)

        return parameter_coordinates

    def index_to_xy(self, x_index, y_index):
        # an index like "5th row/column" gets passed straight to xy on seaborn axes
        return float(x_index), float(y_index)

    def xy_to_parameter_coordinates(self, x, y):
        key1_value = self.df.index.to_list()[int(y)]
        key2_value = self.df.columns.to_list()[int(x)] #columns associated with key2, index with key1

        parameter_coordinates = {self.key1:key1_value, self.key2:key2_value}

        return parameter_coordinates

    def parameter_coordinates_to_xy(self, parameter_coordinates):
        # parameter coordinates is like {key1_name: key1_value ...}
        x = float(self.df.columns.to_list().index(parameter_coordinates[self.key2])) #columns associated with key2, index with key1
        y = float(self.df.index.to_list().index(parameter_coordinates[self.key1]))

        return x,y

    def draw_patch(self, point, x,y):
        # refactor this so it calls a draw_patch method of the subfigure

        if point.is_baseline:
            patch = self.ax.add_patch(patches.Ellipse((x+self.center_offset,y+self.center_offset), 1, 1, fill=False, edgecolor=point.color, lw=2))
        else:
            patch = self.ax.add_patch(patches.Rectangle((x,y), 2*self.center_offset, 2*self.center_offset, fill=False, edgecolor=point.color, lw=2))

        return patch

class ContourPlot(OnMatplotlibAxes):
    def __init__(self, ax, keys, df, title, color_label, scatter_stars=True, **kwargs):
        self.df = df.unstack()
        self.stacked_df = df
        #alias of df
        self.Z = self.df
        OnMatplotlibAxes.__init__(self, ax, keys, **kwargs)

        self.title = title
        self.color_label = color_label
        self.scatter_stars = scatter_stars

    def draw(self, interactive):
        Subfigure.draw(self, interactive)

        #X,Y = np.meshgrid(np.linspace(0.2, 0.9, 8), np.linspace(0.2, 0.9, 8))
        X,Y = np.meshgrid(self.Z.columns, self.Z.index)
        print(self.kwargs)
        contourf = plt.contourf(X, Y, self.Z, **self.kwargs)

        #self.ax.set_ylim(self.ax.get_ylim()[::-1]) # invert the y-axis

        cbar = plt.colorbar(contourf)
        cbar.ax.set_ylabel(self.color_label)
        # Add the contour line levels to the colorbar
        #cbar.add_lines(contourf)

        if self.scatter_stars:
            self.scatter_point_estimates(interactive)

        plt.title(self.title)
        plt.xlabel(pretty_names[interactive.key2])
        plt.ylabel(pretty_names[interactive.key1])

        plt.sca(self.ax)
        self.draw_patches(interactive)

class ProbabilityContourPlot(ContourPlot):
    def __init__(self, ax, keys, df, title, color_label, scatter_stars=True, **kwargs):
        #import pdb; pdb.set_trace()
        kwargs.update({'levels':5})
        super().__init__(ax, keys, df, title, color_label, scatter_stars=scatter_stars, **kwargs)
        #self.kwargs['levels'] = 5

class Heatmap(OnSeabornAxes):
    def __init__(self, ax, keys, df, title, scatter_stars=True):
        OnSeabornAxes.__init__(self, ax, keys)
        self.stacked_df = df
        self.df = df.unstack()
        self.title = title

        self.scatter_stars = scatter_stars

    def draw(self, interactive):
        Subfigure.draw(self, interactive)

        ax = sns.heatmap(self.df, mask=np.isinf(self.df), ax=self.ax, cbar=True, cmap=sns.cm.rocket_r) # need .unstack() here if we don't do it by default
        ax.invert_yaxis()

        if self.scatter_stars:
            self.scatter_point_estimates()#color='blue')

        plt.title(self.title)

        self.draw_patches()

class ConfidenceHeatmap(Heatmap):
    def draw(self, interactive):
        Subfigure.draw(self, interactive)
        cmap = sns.color_palette("Greens", self.n_masks)
        #cmap.set_bad("black")
        in_no_group_mask = np.where(self.df==0., True, False)
        ax = sns.heatmap(self.df, mask=in_no_group_mask, ax=self.ax, cbar=True, cmap=cmap) # need .unstack() here if we don't do it by default
        ax.invert_yaxis()
        #ax.set_facecolor("wheat")
        colorbar = ax.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([colorbar.vmin + r / self.n_masks * (0.5 + i) for i in range(self.n_masks)])
        colorbar.set_ticklabels(list(self.labels))
        #colorbar.ax.invert_yaxis()

        #ax.xaxis.set_major_locator(ticker.MultipleLocator(9))
        #plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        #ax.set_xticks(range(len(self.df)))
        #ax.set_xticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

        if self.scatter_stars:
            self.scatter_point_estimates(interactive)

        plt.title(self.title)

        self.draw_patches(interactive)

class InOrOutConfidenceIntervalHeatmap(ConfidenceHeatmap):
    def __init__(self, ax, keys, df, title, scatter_stars=True):
        super().__init__(ax, keys, df, title, scatter_stars=scatter_stars)
        #import pdb; pdb.set_trace()
        #self.df = self.normalize_probability(self.df.unstack()).unstack().T
        self.n_masks = 1
        self.labels = ["95%"]
        self.df = likelihood.confidence_mask_from_logl(self.df.unstack()).unstack().T
        #import pdb; pdb.set_trace()

class ManyMasksConfidenceHeatmap(ConfidenceHeatmap):
    def __init__(self, ax, keys, df, title, scatter_stars=True):
        super().__init__(ax, keys, df, title, scatter_stars=scatter_stars)
        self.df = likelihood.confidence_mask_from_logl(self.df.unstack(), percentiles=(0.99, 0.95, 0.90, 0.85)).unstack().T
        self.n_masks = 4
        self.labels = reversed(["85%", "90%", "95%", "99%"])