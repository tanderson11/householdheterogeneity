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

class SelectedPoint:
    def __init__(self, parameter_coordinates, color, is_baseline=False):
        self.parameter_coordinates = parameter_coordinates
        self.color = color

        self.patches = []

        self.is_baseline = is_baseline

    def __str__(self):
        return "Selected point at {0}".format(self.parameter_coordinates)
        
    def draw(self, ax, x, y, do_offset, x_scale, y_scale):
        # refactor this so it calls a draw_patch method of the subfigure
        if self.is_baseline:
            patch = ax.add_patch(patches.Ellipse((x+0.5*do_offset,y+0.5*do_offset), 1*x_scale, 1*y_scale, fill=False, edgecolor=self.color, lw=2))
        else:
            patch = ax.add_patch(patches.Rectangle((x,y), 1*x_scale, 1*y_scale, fill=False, edgecolor=self.color, lw=2))

        self.patches.append(patch) 
        return patch

    def remove(self):
        for patch in self.patches:
            patch.remove()

        return self.color

class InteractiveFigure:
    def __init__(self, path, subplots_shape, subfigure_types, baseline_key1_value, baseline_key2_value, recompute_logl=False):
        # -- Reading the files --
        os.chdir(path)
        
        self.comparison_df = pd.read_hdf('comparison_df.hdf')
        self.full_baseline_df = pd.read_hdf('baseline_df.hdf')

        with open('keys.json', 'r') as handle:
            self.key1, self.key2 = json.load(handle)

        self.baseline_df = self.baseline_at_point(baseline_key1_value, baseline_key2_value, one_trial=True)

        with open("./default_model.json", 'r') as handle:
            self.baseline_model_dict = json.load(handle) # we load the baseline parameters from a json file
        print(self.baseline_model_dict)

        with open("./baseline_sizes.json", 'r') as handle:
            self.baseline_sizes = json.load(handle) # we load the baseline parameters from a json file

        # --- Generating a logl df for faster everything ---
        try:
            if recompute_logl:
                raise(FileNotFoundError)
            else:
                self.full_logl_df = pd.read_hdf('logl_df.hdf')  
        except FileNotFoundError: # make the logl df if it doesn't exist
            comparison_grouped = self.comparison_df.groupby([self.key1, self.key2]) # groups by the two axes of the plots
            frequencies = comparison_grouped.apply(lambda g: likelihood.compute_frequencies(g, ["size", "infections"])) # precompute the frequencies at each point because this is the expensive step
            print(frequencies)

            #  --- NEW METHOD ---
            counts = self.full_baseline_df.groupby([self.key1, self.key2, "trialnum", "size", "infections"])["model"].count()
            counts = counts.reindex(counts.index.rename(["baseline " + self.key1, "baseline " + self.key2, "trialnum", "size", "infections"]))

            new_freqs = np.log(frequencies).stack(level=[0,1])

            counts.name = "count"
            new_freqs.name = "freq"

            merged = pd.merge(new_freqs.reset_index(), counts.reset_index(), on=["size", "infections"])
            indexed_merge = merged.set_index(["baseline " + self.key1, "baseline " + self.key2, "trialnum", self.key1, self.key2, "size", "infections"])
            indexed_merge["logl"] = indexed_merge["freq"] * indexed_merge["count"]
            self.full_logl_df = indexed_merge.groupby(["baseline " + self.key1, "baseline " + self.key2, "trialnum", self.key1, self.key2])["logl"].sum()
            # --- END NEW METHOD ---

            self.full_logl_df.to_hdf("logl_df.hdf", key='full_logl_df', mode='w')
            print(self.full_logl_df)

        # -- Selecting the location of the baseline point --
        baseline_coordinates = {self.key1:baseline_key1_value, self.key2:baseline_key2_value}
        self.baseline_color = "blue"
        self.baseline_point = SelectedPoint(baseline_coordinates, self.baseline_color, is_baseline=True)
        
        self.selected_points = [self.baseline_point]
        
        # --
        self.available_colors = ["orange", "red", "green", "violet"]

        self.subplots_shape = subplots_shape
        self.subfigure_types = subfigure_types

        self.make_figure()

    def baseline_at_point(self, key1_value, key2_value, one_trial=False):
        baseline_df = self.full_baseline_df[(self.full_baseline_df[self.key1] == key1_value) & (self.full_baseline_df[self.key2] == key2_value)]

        try: # for legacy dfs that don't always include a trial column
            print(baseline_df["trialnum"])
        except KeyError:
            baseline_df["trialnum"] = 0

        if one_trial:
            baseline_df = baseline_df[baseline_df["trialnum"] == 0] # for concreteness, use only trial 1

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

            self.available_colors.append(removed.remove())
            self.draw_after_toggle()

        self.fig.canvas.draw()        
    
    def draw_after_toggle(self):
        for ax in self.ax.ravel():
            sf = ax.associated_subfigure
            if isinstance(sf, Heatmap):
                sf.draw_patches()
            elif isinstance(sf, ContourPlot):
                print("DRAWING PATCHES!")
                sf.draw_patches()
            elif not(isinstance(sf, ContourPlot)):
                sf.draw()

    def select(self, parameter_coordinates, **kwargs):
        try:
            color = self.available_colors.pop()
        except IndexError:
            print("Error. Tried to select point but no colors are available.")
            return False
        
        point = SelectedPoint(parameter_coordinates, color,  **kwargs)
        self.selected_points.append(point)

        self.draw_after_toggle()

        
        return point
    
    def make_figure(self):
        # -- Associating Subfigure objects with axes objects --
        self.fig, self.ax = plt.subplots(*self.subplots_shape, figsize=(15,7))
        #self.fig, self.ax = plt_handles
        print(self.ax.ravel())
        for subfigure_type, _ax in zip(self.subfigure_types, self.ax.ravel()):
            _ax.associated_subfigure = subfigure_factory(subfigure_type, _ax, self) # adds a subfigure parameter to each plt axes instance

        # -- Drawing --
        self.draw()
        plt.tight_layout()

        self.new_baseline_flag = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

    def reset_baseline(self, parameter_coordinates):
        print("Resetting the baseline point")
        print(parameter_coordinates[self.key1], parameter_coordinates[self.key2])
        self.baseline_df = self.baseline_df = self.baseline_at_point(parameter_coordinates[self.key1], parameter_coordinates[self.key2], one_trial=True)
        #self.full_baseline_df[(self.full_baseline_df[self.key1] == parameter_coordinates[self.key1]) & (self.full_baseline_df[self.key2] == parameter_coordinates[self.key2])]
        print(self.baseline_df)
        self.baseline_point = SelectedPoint(parameter_coordinates, self.baseline_color, is_baseline=True)

        self.selected_points[0] = self.baseline_point

        plt.close()

        self.make_figure()

    def draw(self):
        for ax in self.ax.ravel():
            ax.associated_subfigure.draw() # draws each subfigure

    def on_click(self, event):
        print(event.inaxes.associated_subfigure)
        if self.new_baseline_flag:
            click_type = "reset baseline"
        else:
            click_type = "select"

        event.inaxes.associated_subfigure.click(event.xdata, event.ydata, click_type) # pass clicks to the subfigure objects

    def on_key(self, event):
        if event.key == " ":
            self.new_baseline_flag = True

def subfigure_factory(plot_type, ax, interactive):
    possible_plot_types = ['logl heatmap', 'logl contour plot', 'average heatmap', 'infection histograms', 'two point likelihoods', 'trait histograms', 'average contour plot']
    assert plot_type in possible_plot_types, "No plot of type {} is known to exist".format(plot_type)

    if plot_type == 'logl_heatmap' or plot_type == 'logl contour plot':
        pass
        #logl_df = interactive.comparison_df.groupby([interactive.key1, interactive.key2]).apply(lambda g: likelihood.log_likelihood(["size", "infections"], interactive.baseline_df, g))

    if plot_type == 'logl heatmap':
        # new code with the full logl df
        trialnumber = 0
        #import pdb; pdb.set_trace()
        logl_df = interactive.full_logl_df.loc[interactive.baseline_point.parameter_coordinates[interactive.key1], interactive.baseline_point.parameter_coordinates[interactive.key2], trialnumber] # pulling out the coordinates of the baseline and the trialnumber

        #logl_df_old = interactive.comparison_df.groupby([interactive.key1, interactive.key2]).apply(lambda g: likelihood.log_likelihood(["size", "infections"], interactive.baseline_df, g))
        #logl_df_old = logl_df.unstack() # may or may not want to unstack at this point
        
        #import pdb; pdb.set_trace()
        print("LOGL DF\n", logl_df)

        if interactive.baseline_model_dict["importation_rate"] == 0:
            title = "Log likelihood of observing {0} baseline (size={1}) versus {2} and {3}\n seeding={4}, daily importation=0".format(interactive.baseline_color, sum(interactive.baseline_sizes.values()), interactive.key1, interactive.key2, interactive.baseline_model_dict["seeding"]["name"])
        else:
            title = "Log likelihood of observing {0} baseline (size={1}) versus {2} and {3}\n seeding={4}, daily importation={4:.4f}, and duration={5}".format(interactive.baseline_color, sum(interactive.baseline_sizes.values()), interactive.key1, interactive.key2, interactive.baseline_model_dict["seeding"]["name"], interactive.baseline_model_dict["importation_rate"], interactive.baseline_model_dict["duration"])
                                                                                                
        subfigure = Heatmap(ax, interactive, logl_df, title, scatter_stars=True)
    
    elif plot_type == 'logl contour plot':
        trialnumber=0
        # unstacked for Z
        logl_df = interactive.full_logl_df.loc[interactive.baseline_point.parameter_coordinates[interactive.key1], interactive.baseline_point.parameter_coordinates[interactive.key2], trialnumber].unstack() # pulling out the coordinates of the baseline and the trialnumber
        subfigure = ContourPlot(ax, interactive, logl_df, "Contours of loglikelihood with default levels", color_label="logl")

    elif plot_type == 'average heatmap':
        average_df = interactive.comparison_df.groupby([interactive.key1,interactive.key2])["infections"].apply(lambda g: g.mean()).unstack()
        title = "Average infections per household"
        subfigure = Heatmap(ax, interactive, average_df, title)

    elif plot_type == 'average contour plot':
        average_df = interactive.comparison_df.groupby([interactive.key1,interactive.key2])["infections"].apply(lambda g: g.mean()).unstack()
        subfigure = ContourPlot(ax, interactive, average_df, "Contours of average number of infections with default levels", color_label="average infections")

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
            
            #sns.kdeplot(x=logl_df[key2], y=logl_df[key1], ax=self.ax, annot=True, cbar=False)
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

class OnAxesSubfigure(Subfigure):
    '''A class that exists to serve subclasses OnSeabornAxes and OnMatplotlibAxes, which can manage coordinates, plot patches, etc.'''
    def __init__(self, ax, interactive):
        Subfigure.__init__(self, ax, interactive)
    
    def click(self, event_x, event_y, click_type):
        parameter_coordinates = self.click_to_coordinates(event_x, event_y)

        if click_type == "select":
            self.interactive.toggle(parameter_coordinates)
        elif click_type == "reset baseline":
            self.interactive.reset_baseline(parameter_coordinates)

    def draw_patches(self):
        for point in self.interactive.selected_points:
            x,y = self.parameter_coordinates_to_xy(point.parameter_coordinates)
            print(self)
            print(point,x,y)
            point.draw(self.ax, x, y, self.patches_do_offset, self.patches_x_scale, self.patches_y_scale)

    def scatter_point_estimates(self, **kwargs):
        print("IN SCATTER")
        print(self.logl_df)

        width = self.logl_df.unstack().columns.size

        key1_value = self.interactive.baseline_point.parameter_coordinates[self.interactive.key1]
        key2_value = self.interactive.baseline_point.parameter_coordinates[self.interactive.key2]
        #scatter_df = self.interactive.baseline_at_point(key1_value, key2_value, one_trial=False) # fetch all the trials at the currently selected points

        x_mins = []
        y_mins = []
        for _,_logl_df in self.interactive.full_logl_df.loc[key1_value, key2_value].groupby("trialnum"): # go to the baseline coordinates, then look at all the trials
            #import pdb; pdb.set_trace()
            idx = _logl_df.reset_index()["logl"].argmax() # idiom for finding position of largest value / not 100% sure all the reseting etc. is necessary
            x_min = idx % width + 0.5 + (np.random.rand(1)/2 - 0.25)# the middle of the cell with random fuzzing so there's no overlap
            y_min = idx // width + 0.5 + (np.random.rand(1)/2 - 0.25)

            x_mins.append(x_min)
            y_mins.append(y_min)
        
        #import pdb; pdb.set_trace()

        self.ax.scatter(x_mins, y_mins, s=4, **kwargs) 

class OnMatplotlibAxes(OnAxesSubfigure):
    def __init__(self, ax, interactive):
        OnAxesSubfigure.__init__(self, ax, interactive)
        self.x_grid_values = self.df.columns.to_numpy()
        self.y_grid_values = self.df.index.to_numpy()
        
        print("XGRID:", self.x_grid_values)
        print("YGRID:", self.y_grid_values)

        self.patches_do_offset, self.patches_x_scale, self.patches_y_scale = 0, (self.x_grid_values[-1]-self.x_grid_values[0]) / (len(self.x_grid_values)), (self.y_grid_values[-1]-self.y_grid_values[0]) / (len(self.y_grid_values))
        #import pdb; pdb.set_trace()

    def click_to_coordinates(self, event_x, event_y):
        parameter_coordinates = self.xy_to_parameter_coordinates(event_x, event_y)

        return parameter_coordinates

    def xy_to_parameter_coordinates(self, x, y): # is this reversed in terms of x and y relative to seaborn?
        #x,y = (y,x) THE X AND Y ARE SWITCHED RELATIVE TO SEABORN

        #print("xy to param")
        #print(x,y)
        # we need to snap continuous x and y values onto the grid defined by the dataframe
        x_where_lower = np.ma.MaskedArray(self.x_grid_values, self.x_grid_values > x) # we want to go to the closest point that's up and to the right so things snap to grid in a consistent way across figures
        #print(self.x_grid_values < x, x_where_lower, x-x_where_lower)
        x_idx = (x-x_where_lower).argmin() # nearest value where lower

        y_where_lower = np.ma.MaskedArray(self.y_grid_values, self.y_grid_values > y)
        #print(self.y_grid_values > y, y_where_lower, y-y_where_lower)
        y_idx = (y-y_where_lower).argmin()

        
        parameter_coordinates = {self.interactive.key1:self.y_grid_values[y_idx], self.interactive.key2:self.x_grid_values[x_idx]} # these are inverted because key1 is used in the rows
        #print(parameter_coordinates)
        #import pdb; pdb.set_trace()
        return parameter_coordinates

    def parameter_coordinates_to_xy(self, parameter_coordinates):
        # this is easy because the xy on the matplotlib axes just are parameter coordinates
        x,y = parameter_coordinates[self.interactive.key2], parameter_coordinates[self.interactive.key1] # key order inverted as always
        #import pdb; pdb.set_trace()
        return x,y

class OnSeabornAxes(OnAxesSubfigure):
    def __init__(self, ax, interactive):
        OnAxesSubfigure.__init__(self, ax, interactive)
        self.patches_do_offset, self.patches_x_scale, self.patches_y_scale = 1, 1, 1

    def click_to_coordinates(self, event_x, event_y):
        x = np.floor(event_x)
        y = np.floor(event_y)

        parameter_coordinates = self.xy_to_parameter_coordinates(x, y)

        return parameter_coordinates

    def xy_to_parameter_coordinates(self, x, y):
        key1_value = self.df.index.to_list()[int(y)]
        key2_value = self.df.columns.to_list()[int(x)] #columns associated with key2, index with key1

        parameter_coordinates = {self.interactive.key1:key1_value, self.interactive.key2:key2_value}

        return parameter_coordinates

    def parameter_coordinates_to_xy(self, parameter_coordinates):
        # parameter coordinates is like {key1_name: key1_value ...}
        x = float(self.df.columns.to_list().index(parameter_coordinates[self.interactive.key2])) #columns associated with key2, index with key1
        y = float(self.df.index.to_list().index(parameter_coordinates[self.interactive.key1]))

        return x,y

class ContourPlot(OnMatplotlibAxes):
    def __init__(self, ax, interactive, df, title, color_label, scatter_stars=False):
        self.df = df
        OnMatplotlibAxes.__init__(self, ax, interactive)
        
        self.Z = df
        self.title = title
        self.color_label = color_label

        self.scatter_stars = scatter_stars

        
    def draw(self):
        Subfigure.draw(self)

        #X,Y = np.meshgrid(self.Z.columns, self.Z.index[::-1])
        X,Y = np.meshgrid(self.Z.columns, self.Z.index)
        contourf = plt.contourf(X, Y, self.Z)

        self.ax.set_ylim(self.ax.get_ylim()[::-1]) # invert the y-axis



        cbar = plt.colorbar(contourf)
        cbar.ax.set_ylabel(self.color_label)
        # Add the contour line levels to the colorbar
        #cbar.add_lines(contourf)

        if self.scatter_stars:
            key1_value = self.interactive.baseline_point.parameter_coordinates[self.interactive.key1]
            key2_value = self.interactive.baseline_point.parameter_coordinates[self.interactive.key2]
            width = self.logl_df.unstack().columns.size
            x_mins = []
            y_mins = []
            for _,_logl_df in self.interactive.full_logl_df.loc[key1_value, key2_value].groupby("trialnum"): # go to the baseline coordinates, then look at all the trials
                idx = _logl_df.reset_index()["logl"].argmax() # idiom for finding position of largest value / not 100% sure all the reseting etc. is necessary
                x_min = idx % width
                y_min = idx // width

                # need to convert the x and y into parameter coordinates

                x_mins.append(x_min)
                y_mins.append(y_min)

        plt.title(self.title)
        plt.xlabel(self.interactive.key2)
        plt.ylabel(self.interactive.key1)

        plt.sca(self.ax)
        self.draw_patches()

class Heatmap(OnSeabornAxes):
    def __init__(self, ax, interactive, logl_df, title, scatter_stars=True):
        OnSeabornAxes.__init__(self, ax, interactive)
        self.logl_df = logl_df
        self.df = logl_df.unstack()
        self.title = title

        self.scatter_stars = scatter_stars

    def draw(self):
        Subfigure.draw(self)

        sns.heatmap(self.df, ax=self.ax, cbar=True, cmap=sns.cm.rocket_r) # need .unstack() here if we don't do it by default

        if self.scatter_stars:
            self.scatter_point_estimates(color='blue')

        plt.title(self.title)

        self.draw_patches()
            

#fig, ax = plt.subplots(2,2)
#figures = ["logl heatmap", "infection histograms", "average heatmap", "two point likelihoods"]
#figures = ["logl heatmap", "infection histograms", "average heatmap", "trait histograms"]
#figures = ["logl heatmap", "infection histograms", "logl contour plot", "trait histograms"]
figures = ["logl heatmap", "infection histograms", "logl contour plot", "average contour plot"]

#path = "./experiments/inf_var-hsar-seed_one-no_importation-05-05-13_59/" # hsar=0.3
#path = "./experiments/sus_var-hsar-seed_one-no_importation-05-11-16_35"
#path = "./experiments/inf_var-hsar-seed_one-no_importation-05-11-17_58"
#path = "./experiments/inf_var-hsar-seed_one-no_importation-05-12-15:33"

#path = "./experiments/inf_var-hsar-seed_one-no_importation-05-12-18_32"
#path = "./experiments/inf_var-hsar-seed_one-no_importation-05-17-21:54" #size 500
#path = "./experiments/sus_var-hsar-seed_one-no_importation-05-19-00:48/" #size 500
path = "./experiments/sus_var-hsar-seed_one-no_importation-05-19-17:08" #size 2000


interactive = InteractiveFigure(path, (2,2), figures, 0.8, 0.3, recompute_logl=False)

