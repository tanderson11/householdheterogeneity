import pandas as pd
import numpy as np
import utilities
import constants
import traits

from settings import GPU
if GPU:
    from torch_forward_simulation import torch_state_length_sampler, torch_forward_time
    state_length_sampler = torch_state_length_sampler
    forward_time = torch_forward_time
else:
    from forward_simulation import state_length_sampler, forward_time
    

class Model:
    def __init__(self, name,
                 state_length_dist=state_length_sampler,
                 inf_dist=traits.ConstantTrait("inf"),
                 sus_dist=traits.ConstantTrait("sus"),
                 initial_seeding=utilities.seed_one_by_susceptibility,
                 household_beta=0,
                 importation_rate=0,
                 duration=0,
                 vaccine=None,
                 vaccination_method=None):
        
        self.name = name
        self.state_length_dist = state_length_dist

        self.seeding = initial_seeding
        self.importation_rate = importation_rate
        self.duration = duration

        self.household_beta = household_beta

        self.sus_dist = sus_dist
        self.inf_dist = inf_dist
        
        assert (vaccine and vaccination_method) or (not vaccine and not vaccination_method)
        self.vaccine=vaccine
        self.vaccination_method = vaccination_method   

    def __repr__(self):
        labels = ["household_beta", "seeding", "duration", "importation rate", "susceptibility", "infectiousness"]
        fields = ["{0:.3f}".format(self.household_beta), self.seeding, self.duration, "{0:.3f}".format(self.importation_rate), self.sus_dist, self.inf_dist]
        self_str = "Model named {0} with:\n".format(self.name)
        for label,field in zip(labels, fields):
            self_str += "\t{0:18} = {1}\n".format(label, field)  

        return self_str
        
    def run_trials(self, trials, sizes, household_beta=None):
        
        if household_beta:
            beta = household_beta
        else:
            beta = self.household_beta
        
        if household_beta and self.household_beta:
            print("Beta was handed to both the model and the run_trials method. Overriding with the function call value")
        
        #size_of_one_trial = sum(sizes.values()) # total # of households
        expanded_sizes = {size:count*trials for size,count in sizes.items()}

        pop = Population(self, expanded_sizes)
        pop.simulate_population(beta, duration=self.duration)
        
        #trialnums = [i for t in range(trials) for i in range(size_of_one_trial)]
        
        dfs = []
        grouped = pop.df.groupby("size")
        for size, group in grouped:
            count = sizes[size]
            trialnums = [t for t in range(trials) for i in range(count)]
            #print(size, count, trialnums)
            group["trialnum"] = trialnums
            #print(group)
            dfs.append(group)
        #print(pop.df)
        return pd.concat(dfs) #pop.df

    def sample_hsar(self, sizes, household_beta=None, ignore_traits=True):
        pop = Population(self, sizes)
        return pop.sample_hsar(self.household_beta, ignore_traits=ignore_traits)

class Population:
    def __init__(self, model, household_sizes):
        self.model = model
        self.subpops = [SubPopulation(self.model, size, count) for size, count in household_sizes.items()]
        
        sizes = [sp.size for sp in self.subpops for i in range(sp.count)]
        self.df = pd.DataFrame({"size":sizes},
                               columns = ["size","model","infections"])
        self.df["model"] = model.name

    def simulate_population(self, household_beta=0, duration=0, trials=1):
        #assert household_beta==0 or self.model.household_beta==0, "Model has a defined household beta, but another household beta was passed to simulate"
        if household_beta > 0:
            beta = household_beta
        elif self.model.household_beta > 0:
            beta = self.model.household_beta

        for p in self.subpops:
            infections = p.simulate_households(beta, duration)
            #print(infections.shape)
            
            num_infections = np.sum(infections, axis=1)
            self.df.loc[self.df["size"]==p.size, 'infections'] = num_infections
            
            if self.model.vaccine:
                self.df.loc[self.df["size"]==p.size, 'num vaccinated'] = np.sum(p.inoculations, axis=1)
                self.df.loc[self.df["size"]==p.size, 'vaccinated infected'] = np.sum(infections * (p.inoculations == 1), axis=1)

        self.df["infections"] = pd.to_numeric(self.df["infections"])
        
        if self.model.vaccine:
            self.df["vaccinated infected"] = pd.to_numeric(self.df["vaccinated infected"])
            self.df["unvaccinated infected"] = self.df["infections"] - self.df["vaccinated infected"]
            self.df["num vaccinated"] = pd.to_numeric(self.df["num vaccinated"])
            self.df["num unvaccinated"] = self.df["size"] - self.df["num vaccinated"]
            
        return self.df["infections"]

    def sample_hsar(self, household_beta, ignore_traits=True):
        hsars = []
        for p in self.subpops:
            hsar = p.sample_hsar(household_beta, ignore_traits)
            hsar = pd.Series(hsar)
            hsar.name = p.size
            hsars.append(hsar)
            
        hsar_df = pd.concat(hsars, axis=1)
        #hsar_df.columns=[p.size for p in seelf.subpops]
        #means.append(np.mean(hsars))
        #stds.append(np.std(hsars))
        hsar_df.index.name = "sample"
        return hsar_df

    def r0_from_mean_length_no_traits(self, household_beta):
        r0s = [household_beta * (p.size - 1) * constants.numpy_mean_vec[constants.INFECTIOUS_STATE] for p in self.subpops] # simplest approximation
        r0 = pd.DataFrame({"size":[p.size for p in self.subpops],"r0":r0s})
        return r0

    def sample_r0(self, household_beta, samples):
        for p in self.subpops:
            r0 = p.sample_r0(household_beta)
            #print("R0", r0, r0.shape)
            self.df.loc[self.df["size"] == p.size, 'r0'] = r0
            
        return self.df["r0"]

    def likelihoods(self, observed):
        probabilities = use_simulated_data_as_likelihoods(self.df)
    
        return logl

class SubPopulation:
    def __init__(self, model, size, count):
        self.model = model
        
        self.size = size
        self.count = count
        self.susceptibility = model.sus_dist((count, size, 1))
        self.infectiousness = model.inf_dist((count, 1, size))
        
        nd_eyes = np.stack([np.eye(size,size) for i in range(count)])
        adjmat = 1 - nd_eyes
        
        if model.vaccine:
            self.inoculations = model.vaccination_method((count, size, 1)) # a binary vector that tracks who received a vaccine
            _inf_inocs = self.inoculations.transpose(0,2,1) # rearranging this vector to match the order of the infectivity vector
            
            vax_sus_vector = self.inoculations * model.vaccine.vax_sus
            vax_sus_vector[self.inoculations == 0] = 1
            vax_inf_vector = _inf_inocs * model.vaccine.vax_inf
            vax_inf_vector[_inf_inocs == 0] = 1

            self.susceptibility *= vax_sus_vector
            self.infectiousness *= vax_inf_vector
            #vaccination_pmat = (vax_sus_vector @ vax_inf_vector) # no need to multiply by adjmat, because it's already multiplied against prob_mat
            #print(vaccination_pmat)

            #self.probability_mat = self.probability_mat * vaccination_pmat # element-wise product to combine the parameters defined by vaccination and those defined otherwise
        
        self.probability_mat = (self.susceptibility @ self.infectiousness) * adjmat
        #print(self.probability_mat)
        
    def simulate_households(self, household_beta, duration, silent=False):
        if not silent:
            #print("RUNNING POP")
            pass
        
        initial_state = self.model.seeding(self.size, self.count, self.susceptibility)
        if self.model.importation_rate > 0 and initial_state.any():
            print("WARNING: importation rate > 0 while initial infections were seeded. Did you intend this?")
        #print("INITIAL",initial_state)
        
        infections = forward_time(initial_state, self.model.state_length_dist, household_beta, self.probability_mat, self.model.importation_rate * self.susceptibility, duration)
        
        return infections
    
    def sample_hsar(self, household_beta, ignore_traits=True, silent=True):
        if ignore_traits:
            print("Sampling HSAR while ignoring traits of members of household")

            basic_sus = np.ones((self.count, self.size, 1))
            basic_inf = np.ones((self.count, 1, self.size))

            nd_eyes = np.stack([np.eye(self.size,self.size) for i in range(self.count)])
            adjmat = 1 - nd_eyes
            p_mat = household_beta * delta_t * (basic_sus @ basic_inf) * adjmat

            print("Seeding uniformly at random (not taking susceptibility into account)")
            one_infection_state = seed_one_by_susceptibility(self.size, self.count, basic_sus)
        else:
            print("Sampling HSAR while including traits of members of household")
            p_mat = household_beta * delta_t * self.probability_mat

            print("Seeding by susceptibility")
            one_infection_state =  seed_one_by_susceptibility(self.size, self.count, self.susceptibility)

        assert constants.SUSCEPTIBLE_STATE==0 # trying to set the state correctly but using an idiom that only makes sense if SUS_STATE==0
        one_infection_state = ((constants.INFECTIOUS_STATE / constants.EXPOSED_STATE) * one_infection_state).astype('int64')
        #print("Random state", one_infection_state)
        state_lengths = self.model.state_length_dist(one_infection_state)

        #print("state lengths", state_lengths)

        state_lengths[one_infection_state == constants.SUSCEPTIBLE_STATE] = np.inf
        state_lengths[one_infection_state == constants.REMOVED_STATE] = np.inf

        #print("infections", one_infection_state == INFECTIOUS_STATE)
        times = state_lengths[one_infection_state == constants.INFECTIOUS_STATE]
        if not silent:
            print("times", times)

        sus_mask = (one_infection_state == constants.SUSCEPTIBLE_STATE)
        inf_mask = (one_infection_state == constants.INFECTIOUS_STATE)
        probabilities = p_mat * sus_mask * inf_mask.transpose(0, 2, 1) # transposing to take what amounts to an outer product in each household
        if not silent:
            print("probability", probabilities)
            print("summed probs", np.sum(probabilities, axis=(1,2)))
            print("probability * time", np.sum(probabilities, axis=(1,2)) * times)
        return (np.sum(probabilities, axis=(1,2)) * times)

    def sample_r0(self, household_beta, length_dist=False): # closed around delta_t
        # assumes infection is introduced uniformly at random among household members
        
        # dividing by size to account for the fact that summing over this matrix effectively introduces size-many infections
        daily_probabilities = (household_beta / self.size) * self.probability_mat
        #print(daily_probabilities, daily_probabilities.shape)
        if not length_dist:
            time = self.model.state_length_dist((np.ones((self.size, self.count)) * constants.INFECTIOUS_STATE).astype('int32')) # sample an infection length for each individual to mitigate variance
                
        else:     
            time = length_dist((np.ones(self.size) * constants.INFECTIOUS_STATE).astype('int32'))
        print(daily_probabilities.shape)
        #print(np.average(time, axis=1), delta_t, np.sum(daily_probabilities, axis=(1,2)))
        return np.average(time, axis=0) * delta_t * np.sum(daily_probabilities, axis=(1,2))
