import pandas as pd
import numpy as np
import population
import utilities
import constants

class Vaccine:
    def __init__(self, vax_sus=1.0, vax_inf=1.0):
        self.vax_sus = vax_sus
        self.vax_inf = vax_inf
        
class VaccineStudy:
    """
    A class that hosts and runs a single vaccine study experiment.

    ...

    Attributes
    ----------
    All the attributes passed at object creation. Run `print(VaccineStudy.__init__.__doc__)` for more information. Additionally:

    household_beta : float
        the real probability per contact per day of transmission between individuals in a household
    household_sizes : dict
        a dictionary whose keys are the sizes of households in each arm of the study and whose values are the number of households of that size
    r0 : DataFrame
        a table of r0s calculated relative to unvacccinated households at each size
    vax_sus : float
        the real parameter that defines the susceptibility of vaccinated individuals (0.0 = 0% chance to be infected, 0.1=10% chance relative to unvaccinated)
    vax_inf : float
        the real parameter that defines the infectivity of vaccinated individuals (0.0 = 0% chance to cause infection, 0.1=10% chance relative to unvaccinated)
    vax_m : Model
        a Model object that packages the parameters necessary to simulate the vaccinated arm of the study
    control_m : Model
        a Model object that packages the parameters necessary to simulate the control arm of the study

    Additionally, after running VaccineStudy.run_trials(# trials), the following useful attributes are available:

    vax_df : DataFrame
        the tabular data that describes the results of a time-forward simulation in the vaccinated arm
    control_df : DataFrame
        the tabular data that describes the results of a time-forward simulation in the control arm
    
    Methods
    -------
    run_trials(trials, arms="both"):
        Conducts the study a number of times equal to trials. Optional argument arms can be used to restrict simulation to relevant arms ("vax" or "control")

        returns : DataFrame, DataFrame
            the results in two pandas DataFrame, vax_df and control_df

    ves():
        Measures the vaccine effect on susceptibility. By calculating 1 - (fraction_vaccinated_infected / fraction_placebo_infected)

        returns : Series
            a labeled series of effects indexed by the trials (number of times a full study was simulated)

    vetotal():
        Measures the aggregate vaccine effect on infections. By calculating 1 - (fraction_infected_in_vaccinated_pop / fractionn_infected_in_placebo_pop)

        returns : Series
            a labeled series of effects indexed by the trials (number of times a full study was simulated)

    vecontact():
        Measures the vaccine effect on susceptibility. By calculating 1 - (fraction_vaccinated_infected / fraction_placebo_infected)

        returns : Series
            a labeled series of effects indexed by the trials (number of times a full study was simulated)

    """
    def __init__(self, name, n_per_arm, sizes, hsar, es, et, duration, importation_rate, vaccination_method):
        """
        Creates a VaccineStudy, which represents a two-armed study with a group of households receiving the vaccine and a group receiving the placebo.

        Parameters
        ----------
        name : str
            a name for the study
        n_per_arm : int
            the total number of households per arm of the study
        sizes : list of ints
            the household sizes among which to divide evenly the total number of households in the study
        hsar : float
            the desired average (over the household sizes specified) household secondary attack rate
        es : float
            the true parameter reduction in a vaccinated individual's susceptibility
        (et_method, et) : (string, float)
            either a tuple of the form ('hsarv', float) to express the hsar calculated relative to vaccinated individuals
            or of the form ('et', float) to express the relative infectivity reduction among vaccinated individuals
        duration : int
            the duration of the experiment in days
        importation_rate : float
            the baseline (no vax) probability per individual per day of being infected from outside the household
        vaccination_method : func
            a function that takes in a shape of the form (count, size, 1) and returns a binary vector with 1s representing individuals who received the vaccine

        Returns
        ----------
        VaccineStudy object
        """
        # copy fields
        self.name = name 
        self.es = es

        self.et_method = et[1]
        if self.et_method == 'hsarv':
            self.hsarv = et[0]
            self.et = et[0]
        elif self.et_method == 'et':
            self.et = et[0]
        
        self.sizes = sizes
        self.n_per_arm = n_per_arm
        self.duration = duration
        self.importation_rate = importation_rate
        self.vaccination_method = vaccination_method
        
        self.n_households = int(n_per_arm/len(sizes))

        # gamma distributed state lengths with shape k and period length T
        T = constants.mean_vec[constants.INFECTIOUS_STATE]
        k = constants.shape_vec[constants.INFECTIOUS_STATE]
        self.household_beta = (k/T) * ((1/(1-hsar)**(1/k))-1)

        if self.et_method == 'hsarv':
            implied_vbeta = (k/T) * ((1/(1-self.hsarv)**(1/k))-1) # the beta required to obtain hsar=hsarv
            vax_inf = implied_vbeta / self.household_beta # the ratio of the vbeta and beta gives the infectiousness of vaccinated individuals
        elif self.et_method == 'et':
            vax_inf = 1-self.et

        vax_sus = 1-es
        vaccine = Vaccine(vax_sus=vax_sus, vax_inf=vax_inf)
        self.vaccine=vaccine
        placebo = Vaccine(vax_sus=1.0, vax_inf=1.0)

        # this is incorrect for gamma distributed state lengths. 
        # what we actually want: what's the probability you infect one other contact before you recover
        #self.household_beta = hsar / (DurMildInf * ((sum(sizes) - len(sizes))/ len(sizes)))

        self.household_sizes = {x:self.n_households for x in sizes} #{4:n_households, 5:n_households, 6:n_households, 7:n_households, 8:n_households}

        v_name = "{0} model es {1} {2} {3} intra beta = {4}".format(name, es, self.et_method, self.et, self.household_beta)
        c_name = "control model with intra beta = {0}".format(self.household_beta)
        self.vax_m = population.Model(v_name, vaccine=vaccine, vaccination_method=vaccination_method, household_beta=self.household_beta, initial_seeding=utilities.seed_zero, importation_rate=importation_rate, duration=duration)
        self.control_m = population.Model(c_name, vaccine=placebo, vaccination_method=vaccination_method, initial_seeding=utilities.seed_zero, household_beta=self.household_beta, importation_rate=importation_rate, duration=duration)

        dummy_pop = population.Population(self.vax_m, self.household_sizes)
        self.r0 = dummy_pop.r0_from_mean_length_no_traits(self.household_beta)

    def __repr__(self):
        self_str = "Vaccine study named {0} with:\n\tn_per_arm        = {1}\n\thousehold sizes  = {2}\n\tes, {3}        = {4:.3f}, {5:.3f}\n\tvax_sus, vax_inf   = {6:.3f}, {7:.3f}\n\tduration         = {8}\n\timportation rate = {9:.3f}\n\thousehold beta   = {10:.3f}\n\tmin r0, max r0   = {11:.3f}, {12:.3f}".format(self.name, self.n_per_arm, self.household_sizes, self.et_method, self.es, self.et, self.vaccine.vax_sus, self.vaccine.vax_inf, self.duration, self.importation_rate, self.household_beta, self.r0["r0"].min(), self.r0["r0"].max())
        return self_str
    
    def run_trials(self, trials, arms='both'):
        print("Running study ...\n", str(self))

        self.vax_df = []
        self.control_df = []
        if arms=='vax' or arms=='both':
            self.vax_df = self.vax_m.run_trials(trials, self.household_sizes)
        if arms=='control' or arms=='both':
            self.control_df = self.control_m.run_trials(trials, self.household_sizes)
        
        return self.vax_df, self.control_df

    def sample_hsar(self, samples):
        sizes = {x:samples for x in self.sizes}
        v_df = self.vax_m.sample_hsar(sizes, ignore_traits=False)
        c_df = self.control_m.sample_hsar(sizes)
        return v_df, c_df

    def ves(self):
        """Vaccination effect on susceptibility using the placebo RR as baseline (equation 1/2 in Betz)"""
        print("Calculating VEs ...\n", str(self))

        vg = self.vax_df.groupby(["trialnum"])
        vgs = vg.sum()
        f_v = vg["vaccinated infected"].sum() / vg["num vaccinated"].sum()

        cg = self.control_df.groupby(["trialnum"])
        cgs = cg.sum()
        f_c = cg["vaccinated infected"].sum() / cg["num vaccinated"].sum()     
        
        # fisher exact test record actual number of events: columns either vaccinated or in household with vaccination vs other and rows = individual was infected vs not
        
        # fisher exact test : comparing primary participants in households

        ##             placebo | vaccinated
        ##  uninfected
        ##  -----
        ##  infected

        fisher_df = pd.concat([cgs["num vaccinated"] - cgs["vaccinated infected"], vgs["num vaccinated"] - vgs["vaccinated infected"], cgs["vaccinated infected"], vgs["vaccinated infected"]], axis=1)
        fisher_df.columns =["cuinfected", "vuinfected", "cinfected", "vinfected"]
        p = fisher_df.apply(lambda row: (scipy.stats.fisher_exact([[row["cuinfected"], row["vuinfected"]], [row["cinfected"], row["vinfected"]]]))[1], axis=1) # index 1 to get p value
        p.name = "fisher p value"

        ve = 1. - f_v / f_c
        #ve.name = "VEs"
        ve.name = "VE"

        return pd.concat([ve, p], axis=1)

    def vecontact(self):
        print("Calculating VEcontact ...\n", str(self))
        vg = self.vax_df.groupby(["trialnum"])
        vgs = vg.sum()
        f_v = vg["unvaccinated infected"].sum() / vg["num unvaccinated"].sum()

        cg = self.control_df.groupby(["trialnum"])
        cgs = cg.sum()
        f_c = cg["unvaccinated infected"].sum() / cg["num unvaccinated"].sum()

        ve = 1. - f_v / f_c
        #ve.name = "VEcontact"
        ve.name = "VE"

        # fisher exact test : comparing households by type but only unvaccinated
        ##             control hh secondary (no placebo) | vaccinated hh secondary (no vax)
        ##  uninfected
        ##  -----
        ##  infected


        fisher_df = pd.concat([cgs["num unvaccinated"]-cgs["unvaccinated infected"], vgs["num unvaccinated"]-vgs["unvaccinated infected"], cgs["unvaccinated infected"], vgs["unvaccinated infected"]], axis=1)
        fisher_df.columns =["cuinfected", "vuinfected", "cinfected", "vinfected"]
        p = fisher_df.apply(lambda row: (scipy.stats.fisher_exact([[row["cuinfected"], row["vuinfected"]], [row["cinfected"], row["vinfected"]]]))[1], axis=1) # index 1 to get p value
        p.name = "fisher p value"

        return pd.concat([ve, p], axis=1)

    def vetotal(self):
        print("Calculating VEtotal ...\n", str(self))
        vg = self.vax_df.groupby(["trialnum"])
        vgs = vg.sum()
        f_v = vg["infections"].sum() / vg["size"].sum()
        
        cg = self.control_df.groupby(["trialnum"])
        cgs = cg.sum()
        f_c = cg["infections"].sum() / cg["size"].sum()

        ve = 1. - (f_v)/(f_c)
        #ve.name = "VEtotal"
        ve.name = "VE"

        # fisher exact test : comparing households by type

                ##             control hh | vaccinated hh
        ##  uninfected
        ##  -----
        ##  infected


        fisher_df = pd.concat([cgs["size"]-cgs["infections"], vgs["size"]-vgs["infections"], cgs["infections"], vgs["infections"]], axis=1)
        fisher_df.columns =["cuinfected", "vuinfected", "cinfected", "vinfected"]
        p = fisher_df.apply(lambda row: (scipy.stats.fisher_exact([[row["cuinfected"], row["vuinfected"]], [row["cinfected"], row["vinfected"]]]))[1], axis=1) # index 1 to get p value
        p.name = "fisher p value"

        return pd.concat([ve, p], axis=1)

### Vaccination utility methods
def vaccinate_one(shape):
    inoculations = np.zeros(shape)
    inoculations[:, 0, :] = 1
    return inoculations

def vaccinate_number(shape, number_vaccinated):
    inoculations = np.zeros(shape)
    inoculations[:, np.aranage(number_vaccinated), :] = 1
    return inoculations

def vaccinate_fraction(shape, fraction_vaccinated):
    inoculations = np.zeros(shape)
    vaccinated_in_household = np.arange(np.around(shape[1] * fraction_vaccinated)).astype('int32') #axis 1 corresponds to the individuals in a household
    if np.around(shape[1] * fraction_vaccinated) != shape[1] * fraction_vaccinated:
        print("Warning, attempted to vaccinate fraction={0} of household with size={1}. Vaccinated {2}".format(fraction_vaccinated, shape[1], np.around(shape[1] * fraction_vaccinated)))
    inoculations[:, vaccinated_in_household, :] = 1
    return inoculations


### VE Power utility functions

# calculate power by passing two criteria. lambda threshold; fisher random p value 0.05
# ex. 95% of the time higher than 30% (effect)
def power(ve, threshold, fisher_p_cutoff):
    print("Calculating power")# of {0}".format(ve.name))
    return ve[(ve["VE"] > threshold) & (ve["fisher p value"] < fisher_p_cutoff)]["VE"].count() / ve["VE"].count()
