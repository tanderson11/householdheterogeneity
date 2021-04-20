from constants import *

def state_length_sampler(new_state):
    """Duration in transitional state. Must be at least 1 time unit."""
    alphas = numpy_shape_vec[new_state]
    betas = delta_t/numpy_scale_vec[new_state]
    #key, subkey = random.split(key)
    samples = np.round(np.random.gamma(alphas, size=alphas.shape) / betas)
    
    return 1 + samples.astype('float32') # Time must be at least 1.

def forward_time(state, get_state_lengths, beta_household, probability_matrix, importation_probability, duration=0, secondary_infections=True): # CLOSES AROUND DELTA_T
    debug = False
    # a matrix of probabilities with ith row jth column corresponding to the probability that ith individual is injected by the jth individual
    p_mat = np.array(beta_household * delta_t * probability_matrix)

    state_lengths = get_state_lengths(state)
    
    state_lengths[state == SUSCEPTIBLE_STATE] = np.inf
    state_lengths[state == REMOVED_STATE] = np.inf
    t = 0

    import_flag = importation_probability.any()
    if import_flag:
        assert(duration>0), "A steady importation rate is defined, but no duration was given."
    total_introductions = np.sum((state == EXPOSED_STATE), axis=1)
    
    run_flag = True
    while run_flag:
        inf_mask = (state == INFECTIOUS_STATE)
        sus_mask = (state == SUSCEPTIBLE_STATE)
        
        if import_flag and sus_mask.any(): # if importation is defined and at least one person is in a susceptible state, see if imports happen
            mask = (importation_probability * delta_t * sus_mask > 0) # element wise selection of susceptible individuals
            roll = np.zeros_like(importation_probability)
            roll[mask] = np.random.rand(len(roll[mask]))
            
            importations = np.where(roll < importation_probability * delta_t * sus_mask, 1, 0)
            total_introductions += np.sum(importations, axis=1)
            
            if debug and (importations > 0).any():
                print("time", t)
                print("state", state)
                print("roll", roll)
                print('imps', importations)
        else:
            importations = np.zeros_like(state)
        
        if inf_mask.any() and sus_mask.any():
            probabilities = p_mat * sus_mask * inf_mask.transpose(0, 2, 1) # transposing to take what amounts to an outer product in each household
            #print("probs", probabilities)
            mask = probabilities > 0

            roll = np.zeros_like(probabilities)
            
            roll[mask] = np.random.rand(len(roll[mask]))
            hits = np.where(roll < probabilities, 1, 0)
            #print(roll)
            dstate = (np.sum(hits, axis=2, keepdims=True) >= 1)
            #print(state, dstate)
        else:
            dstate = np.zeros_like(state)


        dstate[importations == 1] = 1
        
        
        state_lengths -= 1

        # anyone who is done waiting advances; cut out negative times before update
        #state_lengths = np.where(state_lengths <= 0, 0, state_lengths)
        state_lengths[state_lengths <= 0] = 0

        #dstate = np.where(state_lengths == 0, 1, dstate)
        dstate[state_lengths == 0] = 1          

        # remap state lengths if there were any changes
        if (dstate != 0).any():
            new_state = (state+dstate).astype('int32')
            #print("dstate", dstate)
            #print("newstate", new_state)
            state_lengths[new_state != state] = get_state_lengths(new_state[new_state != state])
            state_lengths[new_state == SUSCEPTIBLE_STATE] = np.inf
            state_lengths[new_state == REMOVED_STATE] = np.inf

            state = new_state

        t += delta_t
        if duration > 0:
            run_flag = t <= duration
        else:
            run_flag = state[state == EXPOSED_STATE].any() or state[state == INFECTIOUS_STATE].any()
    
    return state != SUSCEPTIBLE_STATE
    #return np.sum(state != SUSCEPTIBLE_STATE, axis=1) #include exposed and infectious states to catch boundary cases when time course has a fixed duration
    #return np.sum(state == REMOVED_STATE, axis=1)
