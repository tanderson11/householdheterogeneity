import pandas as pd
import numpy as np
import utilities
import constants
import traits
import json
import scipy.linalg


# let's stick to torch_forward_simulation and make it work with CPU backups
from torch_forward_simulation import torch_state_length_sampler, torch_forward_time
state_length_sampler = torch_state_length_sampler
forward_time = torch_forward_time

class Model:
    def __init__(self,
                name,
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
        fields = ["{0:.3f}".format(self.household_beta), self.initial_seeding, self.duration, "{0:.3f}".format(self.importation_rate), self.sus_dist, self.inf_dist]
        self_str = "Model named {0} with:\n".format(self.name)
        for label,field in zip(labels, fields):
            self_str += "\t{0:18} = {1}\n".format(label, field)  

        return self_str
        
    def run_trials(self, trials, sizes, **kwargs):
        expanded_sizes = {size:count*trials for size,count in sizes.items()} # Trials are implemented in a 'flat' way for more efficient numpy calculations

        pop = self.__class__.Population(self, expanded_sizes)
        pop.simulate_population(**kwargs, **self.forward_simulation_kwargs)
        
        dfs = []
        grouped = pop.df.groupby("size")
        for size, group in grouped:
            count = sizes[size]
            trialnums = [t for t in range(trials) for i in range(count)]
            group["trialnum"] = trialnums
            dfs.append(group)
        return pd.concat(dfs)

    class Population:
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
            assert self.model.vaccine==None
            #import pdb; pdb.set_trace()
            self.probability_mat = (self.susceptibility @ self.infectiousness) * adjmat
            #print(self.probability_mat)

        def seed_one_by_susceptibility(self):
            n_hh = len(self.df["size"])
            initial_state = self.is_occupied * STATE.susceptible
            
            sus_p = [np.squeeze(self.susceptibility[i,:,:]) for i in range(n_hh)]
            choices = [np.random.choice(range(len(sus)), 1, p=sus/np.sum(sus)) for sus in sus_p] # susceptibility/total_sus chance of getting seed --> means this works with small households
            
            #import pdb; pdb.set_trace()
            choices = np.array(choices).reshape(n_hh)

            initial_state[np.arange(n_hh), choices] = STATE.exposed
            return initial_state

        def seed_zero(self):
            initial_state = self.is_occupied * STATE.susceptible
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
