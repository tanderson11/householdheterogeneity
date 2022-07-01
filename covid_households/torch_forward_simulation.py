import torch
import numpy as np
from settings import device
from settings import model_constants
from settings import STATE

# numpy_initial_state, beta, state_length_sampler, numpy_sus, numpy_inf, numpy_connectivity_matrix
def torch_forward_time(np_state, beta_household, state_length_sampler, numpy_sus, numpy_inf, numpy_connectivity_matrix, np_importation_probability=None, duration=None, secondary_infections=True):
    debug = False  

    #start = time.time()

    ##  --- Move all numpy data structures onto the device as pytorch tensors ---
    # a matrix of probabilities with ith row jth column corresponding to the probability that ith individual is infected by the jth individual
    connectivity_matrix = torch.from_numpy(numpy_connectivity_matrix).to(device)
    sus = torch.from_numpy(numpy_sus).to(device)
    inf = torch.from_numpy(numpy_inf).to(device)

    state = torch.from_numpy(np_state).to(device)     ## move to device
    #print(state.type())
    use_torch_state_lengths = True
    if use_torch_state_lengths:
        state_lengths = torch.zeros_like(state, dtype=torch.double)
        for s in STATE:
            if state_lengths[state==s].nelement() > 0:
                state_lengths[state==s] = state_length_sampler(s, state[state == s].shape) ## how long spent in each state; already on device
    else:
        np_state_lengths = state_length_sampler(np_state) ## how long spent in each state
        state_lengths = torch.from_numpy(np_state_lengths).to(device)

    if np_importation_probability is not None:
        importation_probability = torch.from_numpy(np_importation_probability).to(device)
        import_flag = importation_probability.any()
    else:
        import_flag = False

    #print("device overhead: ", str(time.time() - start))

    ## --- Everything from here on out should be in the device and should be fast ---
    p_mat = (sus @ inf) * connectivity_matrix
    p_mat = beta_household * model_constants.delta_t * p_mat

    state_lengths[state == STATE.susceptible] = np.inf ## inf b/c doesn't change w/o infection
    state_lengths[state == STATE.removed]     = np.inf ## inf b/c doesn't change from removed
    t = 0

    if import_flag:
        assert(duration>0), "A steady importation rate is defined, but no duration was given."
    total_introductions = torch.sum((state == STATE.exposed), axis=1) # counting the total number of introductions, maintained throughout the run
  
    assert duration != 0, "Duration is 0. when it should be None to represent an untimed run."

    run_flag = True
    while run_flag:
        
        inf_mask = (state == STATE.infectious)
        sus_mask = (state == STATE.susceptible)
        
        ## importing from outside the households
        if import_flag and sus_mask.any(): # if importation is defined and at least one person is in a susceptible state, see if imports happen
            mask = (importation_probability * model_constants.delta_t * sus_mask > 0) # element wise selection of susceptible individuals
            roll = torch.zeros_like(importation_probability, dtype = torch.float)

            ## torch.rand can't work on device without existing cuda tensor
            roll_shape = torch.Size((len(roll[mask]),))

            if torch.cuda.is_available():
                random_tensor = torch.cuda.FloatTensor(roll_shape)
            else:
                random_tensor = torch.FloatTensor(roll_shape)
            torch.rand(roll_shape, out=random_tensor) ## I'm 99% sure torch.rand is going to be the same as numpy TK

            ## For sus people, the roll is used to see if they get infected
            roll[mask] = random_tensor

            ## if random value is less than susceptability per delta_t, in susceptible individuals, importation occurs
            importations = torch.where(roll < importation_probability * model_constants.delta_t * sus_mask, 1, 0)
            total_introductions += torch.sum(importations, axis=1) # I think this is per household
            
            if debug and (importations > 0).any():
                print("time", t)
                print("state", state)
                print("roll", roll)
                print('imps', importations)
        else:
            importations = torch.zeros_like(state, dtype = torch.float)
        
        ## infections within the households
        if inf_mask.any() and sus_mask.any(): # if someone is infectious and someone is susceptible, see if infections happen
            ## permute here works as np.transpose
            #print(p_mat)
            probabilities = p_mat * sus_mask * inf_mask.permute(0, 2, 1) # transposing to take what amounts to an outer product in each household
            #print(probabilities)

            ## do the same mask and random approach for infections as for importation, but within each
            ## household it's a size-by-size matrix for odds each person infects each other person
            mask = probabilities > 0 
            roll = torch.zeros_like(probabilities, dtype = torch.float)

            ## Again, this is just the faster version of torch.rand
            roll_shape = torch.Size((len(roll[mask]),))

            if torch.cuda.is_available():
                random_tensor = torch.cuda.FloatTensor(roll_shape)
            else:
                random_tensor = torch.FloatTensor(roll_shape)

            torch.rand(roll_shape, out=random_tensor)

            roll[mask] = random_tensor

            ## Hits are where infections should be introduced
            hits = torch.where(roll < probabilities, 1, 0)

            ## dstate being change in state, so 1 if someone progresses
            dstate = (torch.sum(hits, axis=2, keepdims=True) >= 1)

            if secondary_infections == False:
                #import pdb; pdb.set_trace()
                # send people directly to removed if we're trying to view the model without secondary infections
                dstate = dstate.int()
                dstate[dstate != 0] = STATE.removed - STATE.susceptible
                #print("modifying dstate with no secondary_infections")
                #print(dstate.dtype)
                #print("new dstate.max()", dstate.max())
        else:
            dstate = torch.zeros_like(state)

        ## importations change 1 state
        dstate[importations == 1] = 1
      
        ## decay the state_lengths, so people progress toward next state
        state_lengths -= 1

        # anyone who is done waiting advances; cut out negative times before update, just in case
        state_lengths[state_lengths <= 0] = 0
        dstate[state_lengths == 0] = 1          

        # remap state lengths if there were any changes
        if (dstate != 0).any():
            #import pdb; pdb.set_trace()
            new_state = (state+dstate).int()
            
            ## When patients move to a new state, we generate the length that they'll be
            ## in that state.
            if use_torch_state_lengths:
                for s in STATE:
                    entrants = new_state[torch.logical_and(new_state != state, new_state==s)]
                    #print("entrants", entrants)
                    #print("state", state)
                    #print("dstate", dstate)
                    #print("new state", new_state)
                    if entrants.nelement() > 0:
                        #print("s", s)
                        assert s>STATE.susceptible # no one should be entering the susceptible state (they start there)
                        entrant_lengths = state_length_sampler(s, entrants.shape)
                        state_lengths[torch.logical_and(new_state != state, new_state==s)] = entrant_lengths
            else:
                states_that_changed = new_state[new_state != state]
                new_state_lengths = get_state_lengths(states_that_changed)

                state_lengths[new_state != state] = new_state_lengths
            state_lengths[new_state == STATE.susceptible] = np.inf
            state_lengths[new_state == STATE.removed] = np.inf

            state = new_state

        t += model_constants.delta_t ## update time
        if duration is not None:
            run_flag = (t <= duration)
        else:
            run_flag = state[state == STATE.exposed].any() or state[state == STATE.infectious].any() # keep running if anyone is exposed or infectious
    
    ## send back to the CPU
    return_state = state.cpu().numpy()

    #end = time.time()
    #print("total time: ", str(end - start))
  
    return return_state != STATE.susceptible
    #return np.sum(state != STATE.susceptible, axis=1) #include exposed and infectious states to catch boundary cases when time course has a fixed duration
    #return np.sum(state == STATE.removed, axis=1)
