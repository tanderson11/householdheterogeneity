# Initialization
import importlib
import vaccine
import population
import likelihood
import utilities
import pandas as pd
import numpy as np
import functools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traits
import seaborn as sns
import operator
import os
import json
import itertools

# Manager
# maintains registry of axes -> object
# maintains ledger of selected points
# handles click events by sending them to underlying objects

# Underlying figures
# handle click event and broadcast to the manager any public data that needs to change (manager then tells other subfigures to mirror those changes)

# how to represent that 1 dataframe might be the anchor (ie logl dataframe for infection histograms)?
#    underlying figures access the anchoring frame through the manager. point being, they can do hard look ups because the infection figure
#    JK!! they should only need the baseline and comparison raw dfs and the public ledger of highlighted points

class SelectedPoint:
    def __init__(self, parameter_coordinates, color, is_baseline=False):
        #self.patch = patch # TK who should be the rightful owner of patch objects? probably heatmaps because other figures need to access selected points and they don't know about the patches at all
        self.parameter_coordinates = parameter_coordinates
        self.color = color

        self.patches = []

        self.is_baseline = is_baseline

    def __str__(self):
        return "Selected point at {0}".format(self.parameter_coordinates)
        
    def draw(self, ax, x, y):
        if self.is_baseline:
            patch = ax.add_patch(patches.Circle((x+0.5,y+0.5), 0.5, fill=False, edgecolor=self.color, lw=2))
        else:
            patch = ax.add_patch(patches.Rectangle((x,y), 1, 1, fill=False, edgecolor=self.color, lw=2))

        self.patches.append(patch) 
        return patch

    def remove(self):
        for patch in self.patches:
            patch.remove()

        return self.color

class InteractiveFigure:
    def __init__(self, path, plt_handles, subfigure_types):
        # -- Reading the files --
        os.chdir(path)
        
        self.comparison_df = pd.read_hdf('comparison_df.hdf')
        self.baseline_df = pd.read_hdf('baseline_df.hdf')

        with open('keys.json', 'r') as handle:
            self.key1, self.key2 = json.load(handle)

        with open("./baseline_model.json", 'r') as handle:
            self.baseline_model_dict = json.load(handle) # we load the baseline parameters from a json file

        print(self.baseline_model_dict)
        # -- Selecting the location of the baseline point --
        assert len(self.baseline_df[self.key1].unique())==1, "Expected the baseline df to have one unique value for key1"
        baseline_key1_value = self.baseline_df[self.key1].unique()[0]

        assert len(self.baseline_df[self.key2].unique())==1, "Expected the baseline df to have one unique value for key2"
        baseline_key2_value = self.baseline_df[self.key2].unique()[0]

        baseline_coordinates = {self.key1:baseline_key1_value, self.key2:baseline_key2_value}
        self.baseline_color = "blue"
        self.baseline_point = SelectedPoint(baseline_coordinates, self.baseline_color, is_baseline=True)
        
        self.selected_points = [self.baseline_point]
        
        # --
        self.available_colors = ["orange", "red", "green", "violet"]


        # -- Associating Subfigure objects with axes objects --
        self.fig, self.ax = plt_handles
        print(self.ax.ravel())
        for subfigure_type, _ax in zip(subfigure_types, self.ax.ravel()):
            _ax.associated_subfigure = subfigure_factory(subfigure_type, _ax, self) # adds a subfigure parameter to each plt axes instance

        # -- Drawing --
        self.draw()
        plt.tight_layout()

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        #self.fig.canvas.mpl_connect('key_press_event', self.onkey)

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

            self.available_colors.append(removed.remove())
            for ax in self.ax.ravel():
                sf = ax.associated_subfigure
                if isinstance(sf, Heatmap):
                    sf.draw_patches()
                else:
                    sf.draw()

        self.fig.canvas.draw()        
        
    def select(self, parameter_coordinates, **kwargs):
        try:
            color = self.available_colors.pop()
        except IndexError:
            print("Error. Tried to select point but no colors are available.")
            return False
        
        point = SelectedPoint(parameter_coordinates, color,  **kwargs)
        self.selected_points.append(point)

        for ax in self.ax.ravel():
            sf = ax.associated_subfigure
            if isinstance(sf, Heatmap):
                sf.draw_patches()
            else:
                sf.draw()
        
        return point
        
    def draw(self):
        for ax in self.ax.ravel():
            ax.associated_subfigure.draw() # draws each subfigure

    def onclick(self, event):
        print(event.inaxes.associated_subfigure)
        event.inaxes.associated_subfigure.click(event.xdata, event.ydata) # pass clicks to the subfigure objects


def subfigure_factory(plot_type, ax, interactive):
    possible_plot_types = ['logl heatmap', 'average heatmap', 'infection histograms', 'two point likelihoods', 'trait histograms']
    assert plot_type in possible_plot_types, "No plot of type {} is known to exist".format(plot_type)

    if plot_type == 'logl heatmap':
        logl_df = interactive.comparison_df.groupby([interactive.key1, interactive.key2]).apply(lambda g: likelihood.log_likelihood(["size", "infections"], interactive.baseline_df, g)).unstack() # may or may not want to unstack at this point
        print("LOGL DF\n", logl_df)
        
        if interactive.baseline_model_dict["importation_rate"] == 0:
            title = "Log likelihood of observing {0} baseline versus {1} and {2}\n seeding={3}, daily importation=0".format(interactive.baseline_color, interactive.key1, interactive.key2, interactive.baseline_model_dict["seeding"]["name"])
        else:
            title = "Log likelihood of observing {0} baseline versus {1} and {2}\n seeding={3}, daily importation={4:.4f}, and duration={5}".format(interactive.baseline_color, interactive.key1, interactive.key2, interactive.baseline_model_dict["seeding"]["name"], interactive.baseline_model_dict["importation_rate"], interactive.baseline_model_dict["duration"])
                                                                                                
        subfigure = Heatmap(ax, interactive, logl_df, title)
            
    elif plot_type == 'average heatmap':
        average_df = interactive.comparison_df.groupby([interactive.key1,interactive.key2])["infections"].apply(lambda g: g.mean()).unstack()
        title = "Average infections per household"
        subfigure = Heatmap(ax, interactive, average_df, title)

    elif plot_type == 'infection histograms':
        subfigure = InfectionHistogram(ax, interactive)

    elif plot_type == 'two point likelihoods':
        subfigure = TwoPointLikelihoods(ax, interactive)

    elif plot_type == 'trait histograms':
        subfigure = TraitHistograms(ax, interactive)

    return subfigure

class Subfigure:
    def __init__(self, ax, interactive):
        self.ax = ax
        self.interactive = interactive

    def draw(self):
        plt.sca(self.ax)
        plt.cla()
        print("clearing", type(self))
        #sns.cla()

class SelectionDependentSubfigure(Subfigure):
    def df_at_point(self, point):
        labels = [point.parameter_coordinates[self.interactive.key1], point.parameter_coordinates[self.interactive.key2]] # ie we could use .values if the dict were sorted, TK
        if point.is_baseline: # the baseline dataframe is the data at the baseline point, so we pass it along as is
            restriction = self.interactive.baseline_df
        else: # otherwise, we select only the comparison distributions at the relevant points
            masks = [self.interactive.comparison_df[self.interactive.key1] == labels[0], self.interactive.comparison_df[self.interactive.key2] == labels[1]] # is true precisely where our two keys are at the chosen parameter values
            mask = functools.reduce(operator.and_, masks)
            restriction = self.interactive.comparison_df[mask]

        return labels, restriction

class TwoPointLikelihoods(SelectionDependentSubfigure):
    def __init__(self, ax, interactive):
        Subfigure.__init__(self, ax, interactive)

    def draw(self):
        Subfigure.draw(self)
        point_dfs = []
        color_dict = {}
        for p in self.interactive.selected_points:
            if not p.is_baseline:
                labels, df = self.df_at_point(p)
                point_dfs.append(df)
                color_dict[tuple(labels)] = p.color
                
        #point_dfs = [self.df_at_point(p)[1] for p in self.interactive.selected_points if not p.is_baseline] # 1 index beccause df_at_point returns (parameter_coord_labels, df); ignore baseline because it doesn't make sense for this figure
        

        if len(point_dfs) > 0:
            df = pd.concat(point_dfs)
            #print(df)
            #import pdb; pdb.set_trace()

            
            
            grouped = df.groupby([self.interactive.key1, self.interactive.key2]) #.apply(lambda g: list(itertools.combinations(g.values,2)))
            rows = []
            index = []
            for k1,g1 in grouped:
                column_names=[]
                row = []
                for k2,g2 in grouped:
                    logl = likelihood.log_likelihood(["size", "infections"], g1, g2) # observed, frequencies
                    print(k1, k2)
                    print(logl)
                    row.append(logl)
                    column_names.append(k2) # this is a little inefficient, we only really need to do it once.
                
                # possibly try to create more subplots and vertically stack linear sns heatmaps
                
                rows.append(row)
                print("k1", k1)
                index.append(k1)

            print(index)
            grid = np.array(rows)
            logl_df = pd.DataFrame(grid, columns=column_names, index=index)

            print(logl_df)
            #mask = np.zeros_like(grid, dtype=np.bool)
            #mask[np.tril_indices_from(mask, k=-1)] = True
            
            sns.heatmap(logl_df, ax=self.ax, annot=True, cbar=False)
            plt.ylabel("as observed")
            plt.xlabel("as frequencies")

            plt.title("Two point likelihoods using only simulated data")

            #colors = [p.color for p in self.interactive.selected_points if not p.is_baseline]
            #print(colors)
            #print([t for t in self.ax.xaxis.get_ticklabels()])
            [t.set_color(color_dict[tup]) for t,tup in zip(self.ax.xaxis.get_ticklabels(), index)]
            [t.set_color(color_dict[tup]) for t,tup in zip(self.ax.yaxis.get_ticklabels(), column_names)]
            
            #[t.set_color('red') for t in self.ax.xaxis.get_ticklabels()]
            
            
            #plt.pcolormesh(logl_df)
            #plt.yticks(np.arange(0.5, len(logl_df.index), 1), logl_df.index)
            #plt.xticks(np.arange(0.5, len(logl_df.columns), 1), logl_df.columns)

class TraitHistograms(SelectionDependentSubfigure):
    def __init__(self, ax, interactive):
        SelectionDependentSubfigure.__init__(self, ax, interactive)

    def draw(self):
        SelectionDependentSubfigure.draw(self)

        possible_traits = ["sus_var", "inf_var"]
        if self.interactive.key1 in possible_traits and self.interactive.key2 in possible_traits:
            print("WARNING: both keys are traits. Graphing only key1")

        if self.interactive.key1 in possible_traits:
            graph_key = self.interactive.key1
        elif self.interactive.key2 in possible_traits:
            graph_key = self.interactive.key2
        else:
            return None

        bins = np.linspace(0., 6., 30)
        for p in self.interactive.selected_points:
            trait=traits.GammaTrait("{0}".format(graph_key), mean=1.0, variance=p.parameter_coordinates[graph_key])
            trait.plot(samples=10000, color=p.color, alpha=0.5, bins=bins)
        plt.title("Gamma distributed {0}".format(graph_key))

class InfectionHistogram(SelectionDependentSubfigure):
    def __init__(self, ax, interactive):
        SelectionDependentSubfigure.__init__(self, ax, interactive)

    def draw(self):
        SelectionDependentSubfigure.draw(self)

        color_dict = {}
        restrictions = []
        for p in self.interactive.selected_points:

            labels, restriction = self.df_at_point(p)
            #labels = [p.parameter_coordinates[self.interactive.key1], p.parameter_coordinates[self.interactive.key2]] # ie we could use .values if the dict were sorted, TK
            color_dict[tuple(labels)] = p.color

            restrictions.append(restriction)

        restricted_df = pd.concat(restrictions)
        utilities.bar_chart_new(restricted_df, key=[self.interactive.key1, self.interactive.key2], color=color_dict, title="Normalized histogram of infections", ax=self.ax, ylabel="fraction observed", xlabel="household size, observed number infected")


class Heatmap(Subfigure):
    def __init__(self, ax, interactive, df, title):
        Subfigure.__init__(self, ax, interactive)
        self.df = df
        self.title = title

        #import pdb; pdb.set_trace()
        
        #self.patches = {}

    def xy_to_parameter_values(self, x, y):
        key1_value = self.df.index.to_list()[int(y)]
        key2_value = self.df.columns.to_list()[int(x)] #columns associated with key2, index with key1

        parameter_coordinates = {self.interactive.key1:key1_value, self.interactive.key2:key2_value}

        return parameter_coordinates

    
    def click(self, event_x, event_y):
        x = np.floor(event_x)
        y = np.floor(event_y)

        parameter_coordinates = self.xy_to_parameter_values(x, y)
        self.interactive.toggle(parameter_coordinates)
        

    def draw(self):
        Subfigure.draw(self)
        sns.heatmap(self.df, ax=self.ax) # need .unstack() here if we don't do it by default
        plt.title(self.title)

        self.draw_patches()

    def parameter_values_onto_grid(self, parameter_coordinates):
        # parameter coordinates is like {key1_name: key1_value ...}
        x = np.float(self.df.columns.to_list().index(parameter_coordinates[self.interactive.key2])) #columns associated with key2, index with key1
        y = np.float(self.df.index.to_list().index(parameter_coordinates[self.interactive.key1]))

        return x,y

    def draw_patches(self):
        for point in self.interactive.selected_points:
            x,y = self.parameter_values_onto_grid(point.parameter_coordinates)
            point.draw(self.ax, x, y)
            

fig, ax = plt.subplots(2,2)
#figures = ["logl heatmap", "infection histograms", "average heatmap", "two point likelihoods"]
figures = ["logl heatmap", "infection histograms", "average heatmap", "trait histograms"]


#path = "./experiments/inf_var-hsar-seed_one-no_importation-04-30-14_06" # hsar=0.25
path = "./experiments/inf_var-hsar-seed_one-no_importation-05-05-13_59/" # hsar=0.3

#path = "./experiments/sus_var-hsar-seed_one-no_importation-05-05-14_31/" # hsar=0.3

interactive = InteractiveFigure(path, (fig, ax), figures)

plt.show()
