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

def subfigure_factory(plot_type, ax, interactive):
    assert plot_type in possible_plot_types, "No plot of type {} is known to exist".format(plot_type)

    if plot_type == 'logl heatmap':
        subfigure = LoglHeatmap(ax, interactive)
            
    elif plot_type == 'average heatmap':
        subfigure = AverageHeatmap(ax, interactive)

    elif plot_type == 'infection histograms':
        subfigure = InfectionHistograms(ax, interactive)

    elif plot_type == 'two point likelihoods':
        subfigure = TwoPointLikelihoods(ax, interactive)

    elif plot_type == 'trait histograms':
        subfigure = TraitHistograms(ax, interactive)

class Subfigure:
    def __init__(self, ax, interactive):
        self.ax = ax
        self.interactive = interactive

class Heatmap(Subfigure):
    def __init__(ax, interactive):
        Subfigure.__init__(ax, interactive)

class Histogram(Subfigure):
    def __init__(ax, interactive):
        Subfigure.__init__(ax, interactive)

    
class Subfigure:
    def __init__(self, ax, interactive, plot_type):
        possible_plot_types = ['logl heatmap', 'average heatmap', 'infection histograms', 'two point likelihoods', 'trait histograms']
        assert plot_type in possible_plot_types, "No plot of type {} is known to exist".format(plot_type)
        
        self.ax = ax
        self.interactive = interactive
        self.plot_type = plot_type

        self.mirror_patches = False
        if "heatmap" in self.plot_type:
            self.mirror_patches = True

    def draw(self):
        if self.plot_type == 'logl heatmap':
            self.draw_logl_heatmap()
            
        elif self.plot_type == 'average heatmap':
            self.draw_average_heatmap()

        elif self.plot_type == 'infection histograms':
            self.draw_infection_histograms()

        elif self.plot_type == 'two point likelihoods':
            self.draw_two_point_likelihoods()

        elif self.plot_type == 'trait histograms':
            self.draw_trait_histograms()

    def draw_logl_heatmap(self):
        plt.sca(self.ax)
        sns.heatmap(self.interactive.logl_df.unstack(), ax=self.ax)

        plt.title("Log likelihood of observing {0} baseline versus {1} and {2}\n seeding={3} and daily importation={4:.4f}"
                  .format(self.interactive.baseline_color, self.interactive.key1, self.interactive.key2,
                          self.interactive.baseline_model_dict["seeding"]["name"], self.interactive.baseline_model_dict["importation_rate"]))


    def draw_average_heatmap(self):
        plt.sca(self.ax)
        average_df = full_comparison_df.groupby([self.interactive.key1,self.interactive.key2])["infections"].apply(lambda g: g.mean())
        sns.heatmap(average_df.unstack(), ax=self.ax)
        plt.title("Average infections per household")

            
    def draw_trait_histograms(self):
        plt.sca(self.ax)
        plt.cla()

        possible_traits = ["sus_var", "inf_var"]
        if self.interactive.key1 in possible_traits and self.interactive.key2 in possible_traits:
            print("WARNING: both keys are traits. Graphing only key1")

        if self.interactive.key1 in possible_traits:
            graph_key = self.interactive.key1
        elif self.interactive.key2 in possible_traits:
            graph_key = self.interactive.key2
        else:
            return None

        trait=traits.GammaTrait("{0}".format(graph_key), mean=1.0, variance=self.interactive.baseline_point_dict[graph_key])
        print(trait)
        trait.plot(color=self.interactive.baseline_color, alpha=0.5, bins='auto')

        for _,point_dict,color in self.interactive.patches.values():
            trait=traits.GammaTrait("{0}".format(graph_key), mean=1.0, variance=point_dict[graph_key])
            trait.plot(color=color, alpha=0.5, bins='auto')

        self.interactive.fig.canvas.draw()

    def draw_infection_histograms(self):
        plt.sca(self.ax)
        plt.cla()
        
        labels = []
        for level in [0,1]:
            label = self.interactive.logl_df.index.get_level_values(level).name
            labels.append(self.interactive.baseline_df[label][0])
        
        color_dict = {tuple(labels):self.interactive.baseline_color}
        
        restrictions = [self.interactive.baseline_df]
        for _,point_dict,color in self.interactive.patches.values():
            masks = []
            for name, value in point_dict.items():
                masks.append(self.interactive.comparison_df[name] == value)
            mask = functools.reduce(operator.and_, masks)
            _df = self.interactive.comparison_df[mask]

            labels = []
            for level in [0,1]:
                label = self.interactive.logl_df.index.get_level_values(level).name
                labels.append(_df[label][0])
            
            color_dict[tuple(labels)] = color
            
            restrictions.append(_df)

        restricted_df = pd.concat(restrictions)
        print("COLOR DICT")
        print(color_dict)
        utilities.bar_chart_new(restricted_df, key=[self.interactive.key1, self.interactive.key2], color=color_dict, title="Normalized histogram of infections", ax=self.ax, ylabel="fraction observed", xlabel="household size, observed number infected")
        self.interactive.fig.canvas.draw()

class SelectedPoint:
    def __init__(patch, parameter_coordinates, color):
        # patches = {(x,y): patch, point_dict, color}
        # _,point_dict,color in self.patches.values():
        
        # build a dictionary corresponding to the point
        ## of the form:   point_dict[axis label] = parameter value at point
        self.patch = patch
        self.parameter_coordinates = parameter_coordinates
        self.color = color


class InteractiveLikelihood:
    def __init__(self, baseline_df, full_comparison_df, key1, key2, plt_handles, subfigures):
        self.baseline_df = baseline_df
        self.comparison_df = full_comparison_df

        self.fig, ax = plt_handles
        
        # keys are column names in the dataframe and correspond to the two dimensions of our heatmap
        self.key1 = key1
        self.key2 = key2
        
        with open("./baseline_model.json", 'r') as handle:
            self.baseline_model_dict = json.load(handle) # we load the baseline parameters from a json file

        self.patches = {}

        self.available_colors = ["orange", "red", "green"]
        self.baseline_color = "blue"
            
        logl_df = full_comparison_df.groupby([key1, key2]).apply(lambda g: likelihood.log_likelihood(["size", "infections"], baseline_df, g))
        self.logl_df = logl_df
        
        
        self.ax_dictionary = {}
        print(ax.ravel())

        
        for subfigure, _ax in zip(subfigures, ax.ravel()):
            print(subfigure, _ax)
            self.ax_dictionary[subfigure] = Subfigure(_ax, self, subfigure)

        self.baseline_point_dict = self.highlight_baseline(self.ax_dictionary["logl heatmap"].ax)

        self.draw()
        plt.tight_layout()
            
        #self.ax_dict = {"logl heatmap":ax[0][0], "infection histograms":ax[0][1], "average heatmap":ax[1][0], "trait histograms":ax[1][1]}
        
        
        #self.draw_infections()
        #self.draw_trait()
        #print("done")
        #plt.sca(self.ax_dict["average heatmap"])
        
        
        #self.highlight_baseline(self.ax_dict["average heatmap"])

    def draw(self):
        for sf in self.ax_dictionary.values():
            sf.draw() # draws each subfigure
        
    def xydata_to_parameter_coordinates(self, xdata, ydata):
        x = int(np.floor(xdata))
        y = int(np.floor(ydata))
        print(x,y)

        # build a dictionary corresponding to the point
        ## of the form:   point_dict[axis label] = parameter value at point
        parameter_coordinates = {}
        pairs = zip([x,y],[1,0])
        for coordinate,level in pairs:
            level_values = self.logl_df.index.get_level_values(level)
            parameter_coordinates[level_values.name] = sorted(list(set(level_values)))[coordinate]

        return parameter_coordinates

    def highlight_point(self, ax, x, y, parameter_coordinates):
        if not isinstance(ax, list):
            ax = [ax]
        
        color = self.available_colors.pop()
        new_patches = []
        for _ax in ax:
            patch = _ax.ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=False, edgecolor=color, lw=3))
            self.fig.canvas.draw()
            new_patches.append(patch)

        self.patches[(x,y)] = (new_patches), parameter_coordinates, color
        return patch
    
    def toggle_point(self, ax, x, y, parameter_coordinates):
        patch = self.patches.get((x,y))
        if patch:
            _,_,color = self.patches.pop((x,y)) # removes the dictionary entry
            for p in patch[0]:
                print(p)
                p.remove() # patches are stored in the dict: [patches], point_dict, color

            self.available_colors.append(color)
            self.fig.canvas.draw() 
        else:
            self.highlight_point(ax, x, y, parameter_coordinates)

        return patch

    def onclick(self, event):
        print("CLICK")
        if event.inaxes == self.ax_dictionary["logl heatmap"].ax:
            print("HERE")
            parameter_coordinates = self.xydata_to_parameter_coordinates(event.xdata, event.ydata)
            #self.selected_points.append(point_dict)
            self.toggle_point([self.ax_dictionary["logl heatmap"],self.ax_dictionary["average heatmap"]], np.floor(event.xdata), np.floor(event.ydata), parameter_coordinates)
            #self.toggle_point(self.ax_dict["average heatmap"], np.floor(event.xdata), np.floor(event.ydata), point_dict)

            print("THERE")
            self.draw()
            #self.draw_infections()
            #self.draw_trait()
            
    def highlight_baseline(self, ax):
        indices=[]
        baseline_point_dict = {}
        for level in [1,0]:
            level_values = self.logl_df.index.get_level_values(level)
            axis = sorted(list(set(level_values)))
            baseline_value = float(baseline_df[level_values.name][0]) # pulls out the value of the true parameter (index of 0 because it should be constant)
            dist_tuples = [(i, np.abs(axis[i]-baseline_value)) for i in range(len(axis))]# tuples of the form (index, distance from baseline)
            sorted_distances = sorted(dist_tuples, key=lambda tup: tup[1]) # sort by the distances
            index_of_min_distance = sorted_distances[0][0]
            indices.append(index_of_min_distance)

            baseline_point_dict[level_values.name] = baseline_value

        patch = ax.add_patch(patches.Circle((indices[0]+0.5, indices[1]+0.5), 0.5, fill=False, edgecolor=self.baseline_color, lw=3))
        self.fig.canvas.draw()

        return baseline_point_dict
        #self.patches[(x,y)] = patch
            
        #self.highlight_point(indices[0], indices[1], color="blue")

    def onkey(self, event):
        print(event.key)
        if event.key == " ":
            self.draw_infections()
            self.draw_trait()
        elif event.key == "escape":
            print("ESCAPE")
            for patch,_,color in self.patches.values():
                self.available_colors.append(color)
                patch.remove()
            self.patches = {}
            self.fig.canvas.draw()



#os.chdir("./experiments/sus_var-hsar-seed_one-0.0importation-04-29-16_17")
#os.chdir("./experiments/inf_var-hsar-seed_one-0.0importation-04-29-17_46/")
#os.chdir("./experiments/sus_var-hsar-seed_one-0.0importation-04-29-21_23")
os.chdir("./experiments/inf_var-hsar-seed_one-no_importation-04-30-14_06")
#./experiments/inf_var-hsar-seed_one-importation-04-30-13_40/")

#key1 = "inf_var"
#key2 = "hsar"
#keys = [key1, key2]

#with open('./keys.json', 'w') as handle:
#    json.dump(keys, handle)

with open('keys.json', 'r') as handle:
    key1, key2 = json.load(handle)

full_comparison_df = pd.read_hdf('comparison_df.hdf')
baseline_df = pd.read_hdf('baseline_df.hdf')

fig, ax = plt.subplots(2,2)
#figures = [["logl heatmap", "infection histograms"], ["average heatmap", "trait histograms"]]
#figures = ["logl heatmap", "infection histograms", "average heatmap", "trait histograms"]
figures = ["logl heatmap", "infection histograms", "average heatmap", "trait histograms"]
interactive = InteractiveLikelihood(baseline_df, full_comparison_df, key1, key2, (fig,ax), figures)

interactive.fig.canvas.mpl_connect('button_press_event', interactive.onclick)
interactive.fig.canvas.mpl_connect('key_press_event', interactive.onkey)

plt.show()
