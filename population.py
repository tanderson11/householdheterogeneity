import pandas as pd
import numpy as np
import utilities
import constants
import traits
import json
import scipy.linalg

#if GPU:
if True: # let's stick to torch_forward_simulation and make it work with CPU backups
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
                 initial_seeding='seed_one_by_susceptibility',
                 household_beta=None,
                 importation_rate=0.,
                 duration=None,
                 vaccine=None,
                 vaccination_method=None,
                 **forward_simulation_kwargs):
        
        self.name = name
        self.state_length_dist = state_length_dist
        self.sus_dist = sus_dist
        self.inf_dist = inf_dist

        self.initial_seeding = initial_seeding
        self.household_beta = household_beta

        assert (importation_rate and duration) or (not importation_rate and not duration)
        self.importation_rate = importation_rate
        self.duration = duration

        assert (vaccine and vaccination_method) or (not vaccine and not vaccination_method)
        self.vaccine = vaccine
        self.vaccination_method = vaccination_method

        self.forward_simulation_kwargs = forward_simulation_kwargs

    #def __str__(self):
    #    return "{0}-importation{1}-{2}-{3}".format(name, self.seeding.name, self.importation_rate)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __repr__(self):
        labels = ["household_beta", "seeding", "duration", "importation rate", "susceptibility", "infectiousness"]
        fields = ["{0:.3f}".format(self.household_beta), self.seeding, self.duration, "{0:.3f}".format(self.importation_rate), self.sus_dist, self.inf_dist]
        self_str = "Model named {0} with:\n".format(self.name)
        for label,field in zip(labels, fields):
            self_str += "\t{0:18} = {1}\n".format(label, field)  

        return self_str
        
    def run_trials(self, trials, sizes, **kwargs):
        expanded_sizes = {size:count*trials for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations

        pop = Population(self, expanded_sizes)
        pop.simulate_population(**kwargs, **self.forward_simulation_kwargs)
        
        dfs = []
        grouped = pop.df.groupby("size")
        for size, group in grouped:
            count = sizes[size]
            trialnums = [t for t in range(trials) for i in range(count)]
            group["trialnum"] = trialnums
            dfs.append(group)
        return pd.concat(dfs)

    def sample_hsar(self, sizes, household_beta=None, ignore_traits=True):
        pop = Population(self, sizes)
        return pop.sample_hsar(self.household_beta, ignore_traits=ignore_traits)

class NewModel(Model):
    def run_trials(self, trials, sizes, **kwargs):
        expanded_sizes = {size:count*trials for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations

        pop = self.__class__.NewPopulation(self, expanded_sizes)
        pop.simulate_population(**kwargs, **self.forward_simulation_kwargs)
        
        dfs = []
        grouped = pop.df.groupby("size")
        for size, group in grouped:
            count = sizes[size]
            trialnums = [t for t in range(trials) for i in range(count)]
            group["trialnum"] = trialnums
            dfs.append(group)
        return pd.concat(dfs)

    class NewPopulation:
        def __init__(self, model, household_sizes):
            self.model = model
            unpacked_sizes = [[size]*number for size, number in household_sizes.items()]
            flattened_unpacked_sizes = [x for l in unpacked_sizes for x in l]

            self.df = pd.DataFrame({"size":flattened_unpacked_sizes, "model":self.model},
                columns = ["size","model","infections"])

            total_households = len(self.df)
            max_size = self.df["size"].max()

            self.is_occupied = np.array([[1. if i < hh_size else 0. for i in range(max_size)] for hh_size in self.df["size"]]) # 1. if individual exists else 0.
            self.susceptibility = np.expand_dims(model.sus_dist(self.is_occupied), axis=2) # ideally we will get rid of expand dims at some point
            self.infectiousness = np.transpose(np.expand_dims(model.inf_dist(self.is_occupied), axis=2), axes=(0,2,1))

            # not adjacent to yourself
            nd_eyes = np.stack([np.eye(max_size,max_size) for i in range(total_households)]) # we don't worry about small households here because it comes out in a wash later
            adjmat = 1 - nd_eyes # this is used so that an individual doesn't 'infect' themself
            
            # reintroduce vaccines TK
            #import pdb; pdb.set_trace()
            self.probability_mat = (self.susceptibility @ self.infectiousness) * adjmat
            #print(self.probability_mat)

        def seed_one_by_susceptibility(self):
            n_hh = len(self.df["size"])
            initial_state = self.is_occupied * constants.SUSCEPTIBLE_STATE
            
            sus_p = [np.squeeze(self.susceptibility[i,:,:]) for i in range(n_hh)]
            choices = [np.random.choice(range(len(sus)), 1, p=sus/np.sum(sus)) for sus in sus_p] # susceptibility/total_sus chance of getting seed --> means this works with small households
            
            #import pdb; pdb.set_trace()
            choices = np.array(choices).reshape(n_hh)

            initial_state[np.arange(n_hh), choices] = constants.EXPOSED_STATE
            return initial_state

        def seed_zero(self):
            initial_state = self.is_occupied * constants.SUSCEPTIBLE_STATE
            return initial_state

        def simulate_population(self, **kwargs):
            if self.model.initial_seeding == 'seed_one_by_susceptibility':
                initial_state = self.seed_one_by_susceptibility()
            elif self.model.initial_seeding == 'seed_zero':
                initial_state = self.seed_zero()
            else:
                raise(ValueError('model had unknown seeding {self.model.seeding}'))

            initial_state = np.expand_dims(initial_state, axis=2)

            if self.model.importation_rate > 0 and initial_state.any():
                print("WARNING: importation rate is defined while initial infections were seeded. Did you intend this?")
            
            importation_probability = self.model.importation_rate * self.susceptibility

            #import pdb; pdb.set_trace()
            # calling our simulator
            infections = forward_time(initial_state, self.model.state_length_dist, self.model.household_beta, self.probability_mat, importation_probability, self.model.duration, **kwargs)
                            
            num_infections = np.sum(infections, axis=1)
            self.df["infections"] = num_infections
            assert (self.df["infections"] <= self.df["size"]).all(), "Saw more infections than the size of the household"

            if self.model.vaccine:
                self.df["num vaccinated"] = np.sum(self.inoculations, axis=1)
                self.df["vaccinated infected"] = np.sum(infections * (self.inoculations == 1 & self.is_occupied), axis=1)     

                self.df["unvaccinated infected"] = self.df["infections"] - self.df["vaccinated infected"]
                self.df["num unvaccinated"] = self.df["size"] - self.df["num vaccinated"]
                
            return self.df["infections"]

    def __repr__(self):
        labels = ["household_beta", "seeding", "duration", "importation rate", "susceptibility", "infectiousness"]
        fields = ["{0:.3f}".format(self.household_beta), self.initial_seeding, self.duration, "{0:.3f}".format(self.importation_rate), self.sus_dist, self.inf_dist]
        self_str = "Model named {0} with:\n".format(self.name)
        for label,field in zip(labels, fields):
            self_str += "\t{0:18} = {1}\n".format(label, field)  

        return self_str


class Population:
    def __init__(self, model, household_sizes):
        self.model = model
        self.subpops = [SubPopulation(self.model, size, count) for size, count in household_sizes.items()]
        
        sizes = [sp.size for sp in self.subpops for i in range(sp.count)]
        self.df = pd.DataFrame({"size":sizes},
                               columns = ["size","model","infections"])
        self.df["model"] = model.name

    def simulate_population(self, household_beta=0, duration=0, **kwargs):
        if not (household_beta==0 or self.model.household_beta==0):
            print("WARNING: Model has a defined household beta, but another household beta was passed to simulate")
            print(household_beta, self.model.household_beta)

        if not (duration==0 or self.model.duration==0):
            print("WARNING: Model has a defined duration, but another duration was passed to simulate")
            print(duration, self.model.duration)
        
        if duration > 0:
            duration = duration
        elif self.model.duration > 0:
            duration = self.model.duration

        if household_beta > 0:
            beta = household_beta
        elif self.model.household_beta > 0:
            beta = self.model.household_beta

        for p in self.subpops:
            infections = p.simulate_households(beta, duration, **kwargs)
            
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

    def r0_from_mean_length_no_traits(self, household_beta):
        r0s = [household_beta * (p.size - 1) * constants.numpy_mean_vec[constants.INFECTIOUS_STATE] for p in self.subpops] # simplest approximation
        r0 = pd.DataFrame({"size":[p.size for p in self.subpops],"r0":r0s})
        return r0

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
        
        #self.probability_mat = (self.susceptibility @ self.infectiousness) * adjmat
        #print(self.probability_mat)
        
    def simulate_households(self, household_beta, duration, silent=False, **kwargs):
        if not silent:
            #print("RUNNING POP")
            pass
        
        initial_state = self.model.seeding(self.size, self.count, self.susceptibility)
        if self.model.importation_rate > 0 and initial_state.any():
            print("WARNING: importation rate > 0 while initial infections were seeded. Did you intend this?")
        
        infections = forward_time(initial_state, self.model.state_length_dist, household_beta, self.probability_mat, self.model.importation_rate * self.susceptibility, duration, **kwargs)
        
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
            
            use_accurate_pmat = True
            if use_accurate_pmat:
                print("Using no approximation for the probability matrix")
                p_mat = 1-(1-household_beta)**delta_t * self.probability_mat
            else:
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
