import fancy_plotting

#fig, ax = plt.subplots(2,2)
#figures = ["logl heatmap", "infection histograms", "average heatmap", "two point likelihoods"]
#figures = ["logl heatmap", "infection histograms", "average heatmap", "trait histograms"]
figures = ["logl heatmap", "infection histograms", "logl contour plot", "trait histograms"]
#figures = ["logl heatmap", "logl contour plot", "trait histograms", "average contour plot"]#, "average contour plot"]
axes_shape = (2,2)

path = "./experiments/synthetic-iguana-inf_var-sus_var-hsar032--seed_one-no_importation-06-22-12_10/"

#path = "./experiments/synthetic-iguana-sus_var-hsar-seed_one-no_importation-06-22-01_22/"
#path = "./experiments/synthetic-iguana-inf_var-hsar-seed_one-no_importation-06-22-01_52/"
#path = "./experiments/inf_var-hsar-seed_one-no_importation-05-26-18_46/"  #hh size 4, size = 300
#path = "./experiments/inf_var-hsar-seed_one-no_importation-05-27-01_15/" #hh size 4, size = 5000
empirical_path = False
#empirical_path = "./empirical/geneva/"
#path = empirical_path + "geneva-sus_var-hsar-seed_one-no_importation-06-03-14_25/"
#path = empirical_path + "geneva-inf_var-hsar-seed_one-no_importation-06-08-22_24/"

#empirical_path = "./empirical/BneiBrak/"
#path = empirical_path + "bneibrak-inf_var-hsar-seed_one-no_importation-06-09-13_33/"
#path = empirical_path + "bneibrak-sus_var-hsar-seed_one-no_importation-06-09-21_54/"
#path = empirical_path + "bneibrak-sus_var-inf_var-hsar032--seed_one-no_importation-06-10-02_07/"
#path = empirical_path + "bneibrak-sus_var-inf_var-hsar032--seed_one-no_importation-06-15-15_08/"

interactive = fancy_plotting.InteractiveFigure(path, axes_shape, figures, 0.2, 0.2, recompute_logl=False, empirical_path=empirical_path)

