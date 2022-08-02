from constants import STATE
from settings import model_constants
import torch
from settings import device
import numpy as np

def find_propensities(state, beta, probability_matrix):
    """
    Determines the propensities towards each event (read: different person being infected) possible in each household.
    
    Parameters
    ----------
    state : ndarray / tensor
        Array of current state for each individual in the population
    beta : float
        Constant rate of infection
    connectivity_matrix : ndarray / tensor
        Matrix where A_ij is the relative probability that i would be infected by j if i is susceptible and j infectious
    Returns
    -------
    propensities : ndarray
        Propensity (~instantaneous probability) for each individual in each household
        to be infected given current state
    time : float
        Time it took for the reaction to occur.
    """
    
    # who is infectious and who is susceptible
    inf_mask = (state == STATE.infectious)
    sus_mask = (state == STATE.susceptible)
    
    # the propensities are those probabilities ... if the states of the two individuals are correct
    propensities = probability_matrix * sus_mask * inf_mask.permute(0, 2, 1)
    propensities = propensities.sum(axis=2)
    return propensities

def vector_gillespie_step(propensity_func, state, t, state_lengths, *propensity_args):
    """
    Draws the next event for each household in the population. Assesses whether the drawn event takes place or time advances so far that someone ages out of their state.
    Returns the change in state and the time advancement for each household.
    
    Parameters
    ----------
    propensity_func : function
        Function used for computing propensities. Signature: propensity_func(state, propensity_args)
        Returns an array of propensities.
    state : ndarray
        Array of current state for each individual in the population
    t : ndarray
        The current time in each household
    state_lengths: ndarray
        The time left for each individual in their current state (if transitory) or ~infinity if stationary
    propensity_args : additional arguments
        Arguments to be passed to the propensity-finding-function
        
    Returns
    -------
    dstate : ndarray
        An array such that state + dstate properly represents the new state after one event in each household.
    dtime : ndarray
        Time that passed in each household before its event.
    """
    # time until next aging event (someone leaving one compartment for the next due to time passing) in each household
    dtime_aging, dstate_aging_indices = state_lengths.min(axis=1)
    # Necessary because state length sampler was built for discrete times with a timestep, so it spits out integers
    dtime_aging *= model_constants.delta_t
    
    # find propensities (for each individual in each household to be infected)
    propensities = propensity_func(state, *propensity_args)

    # sum of propensity per household
    household_total_propensity = propensities.sum(axis=1)

    # restrict to households with non-zero propensity to infection to avoid dividing by 0
    valid_propensity_mask = (household_total_propensity != 0.0)
    # time until the drawn event for each household
    dist = torch.distributions.Exponential(household_total_propensity[valid_propensity_mask])
    # sample from the exponential distribution for time until events
    # (and then transposing futzing to get it pointed the right way)
    dtime_gillespie = torch.full_like(dtime_aging, np.inf)
    dtime_gillespie[valid_propensity_mask] = dist.sample().unsqueeze(0).transpose(0,1)

    # relative probability of each event (read: of each person being infected)
    relative_probabilities = propensities[valid_propensity_mask] / household_total_propensity[valid_propensity_mask].unsqueeze(1)

    # randomly choose an event to happen in proportion to the relative probability
    dstate_gillespie_indices = torch.zeros_like(dtime_gillespie, dtype=torch.long)
    dstate_gillespie_indices[valid_propensity_mask] = relative_probabilities.multinomial(num_samples=1, replacement=True)

    # we choose the type of event (infection from gillespie or aging out of a state at fixed time)
    # in each household based on which happens first
    dstate_indices = torch.where((dtime_aging < dtime_gillespie), dstate_aging_indices, dstate_gillespie_indices)
    # the time we should advance is similarly determined by which type of event happened in each household
    dtime = torch.where(dtime_aging < dtime_gillespie, dtime_aging, dtime_gillespie)
    # create the vector such that state + dstate = new_state
    dstate = torch.zeros_like(state)
    # take advantage of the fact that compartments are sequential
    # we would need to more fastidiously track which events took place if backtracking (ex. SIRS) were possible
    dstate[torch.arange(dstate.shape[0]), dstate_indices.transpose(0,1)] = 1
    
    # if we're trying to age someone out of a stationary state
    # (because both Gillespie and aging want to move infinity time ahead)
    # intercept that change in state and stop it
    force_stationary_idx = torch.where(torch.logical_and(
        (dtime_aging == np.inf).unsqueeze(2),
        torch.logical_or(
            state == STATE.susceptible,
            state == STATE.removed))
    )
    dstate[force_stationary_idx] = 0

    return dstate, dtime

def gillespie_simulation(numpy_initial_state, beta, state_length_sampler, numpy_sus, numpy_inf, numpy_connectivity_matrix, **kwargs):
    """
    Takes an initial state of infections in households forward in time to fixation.
    Returns a matrix of households where True represents an infected individual and False an uninfected individaul.
    
    Parameters
    ----------
    numpy_initial_state : ndarray
        Array of initial state for each individual in the population.
        Integers refer to an enumeration of states in constants.STATE.
    beta : float
        Constant rate of infection
    state_length_sampler: func
        A function with signature
            state_length_sampler(state_integer, new_entrants_shape)
            --> waiting times for each entrant into that state.
    sus : ndarray
        Array of the individuals' relative susceptibilities
    inf : ndarray
        Array of the individuals' relative infectivities
    numpy_connectivity_matrix : ndarray
        Matrix with A_ij = 1 if individuals i and j are connected
        (in the same house but not identical) and 0 otherwise
    **kwargs:
        Old versions supported some additional keyword arguments. These aren't used by us but are allowed for to promote compatibility.
        
    Returns
    -------
    is_infected : ndarray
        An array of individuals valued True if that individual is infected and False if that individual is uninfected.
    """
    # move everything onto the torch device
    state = torch.from_numpy(numpy_initial_state).to(device)
    connectivity_matrix = torch.from_numpy(numpy_connectivity_matrix).to(device)
    sus = torch.from_numpy(numpy_sus).to(device)
    inf = torch.from_numpy(numpy_inf).to(device)
    t = torch.zeros((state.shape[0], 1), dtype=torch.float, device=device)
    
    # a matrix whose ij-th entry is the relative probability that i would be infected by j if they are in the right states
    population_matrix = (sus @ inf) * connectivity_matrix
    population_matrix = beta * population_matrix

    # find the duration of the exposed state for everyone who starts exposed
    state_lengths = torch.zeros_like(state, dtype=torch.double)
    for s in STATE:
        if state_lengths[state==s].nelement() > 0:
            state_lengths[state==s] = state_length_sampler(s, state[state == s].shape)

    # while anyone is infected or exposed, we continue simulating
    while (state == STATE.exposed).any() or (state == STATE.infectious).any():
        # perform an update step by finding the next event (via Gillespie simulation) in each household
        dstate, dtime = vector_gillespie_step(find_propensities, state, t, state_lengths, beta, population_matrix)
        state = state + dstate
        t     = t     + dtime
        # the time of waiting in each state /decreases/ by dtime
        state_lengths = state_lengths - dtime.unsqueeze(1)
        
        changed_states = (dstate != 0)
        # when persons move to a new state, we generate the time that they'll spend in that state
        for s in STATE:
            # find all the people who entered state s
            entrants = state[torch.logical_and(changed_states, state==s)]
            if entrants.nelement() > 0:
                # no one should be entering the susceptible state (they start there)
                assert s>STATE.susceptible
                # find the duration of the state for all the entrants
                entrant_lengths = state_length_sampler(s, entrants.shape)
                state_lengths[torch.logical_and(changed_states, state==s)] = entrant_lengths
            # ensure that stationary states don't age out
            state_lengths[state == STATE.susceptible] = np.inf
            state_lengths[state == STATE.removed] = np.inf

    return_state = state.cpu().numpy()
    return return_state != STATE.susceptible

