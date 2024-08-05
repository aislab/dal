from jax import numpy as jp
import numpy as np
import optax
import argparse

# check for the -e flag to determine if we are running in evaluation mode.
parser = argparse.ArgumentParser()
parser.add_argument('-e','--eval',dest='evaluation_mode',default=False,action='store_true',
                    help='Run in evaluation mode (default: False).')
evaluation_mode = parser.parse_known_args()[0].evaluation_mode

# the list of source files to copy into each run's local source directory.
# if you add files (e.g. new environments), you may want to add them to this list.
source_copy_list = ['config.py','utils.py','evolve.py','nmnn.py','task_environment.py','task_visualiser.py']

# toggles generation of images of agent behaviour during evolution.
visualisation = True

# interval (in generations) at which the full population is saved to disk.
save_interval = 10

# interval at which we rewrap the population in new subprocesses to clear caches.
# this is necessary to avoid running out of memory.
housekeeping_interval = 10

# toggles to enable/disable RL and NM.
nm_enabled = True
rl_enabled = True

# environment settings.
# spatial dimensionality of the navigation task (2 and 3 are supported).
# settings below are configured for the 2D task.
# (we got the 3D version to work with more task instances and trials per individual.)
space_dims = 2
# dimensionality of the action space.
act_dims = space_dims
# dimensionality of the observation space (i.e. network input space).
# observations contain the current observation of the goal position, the previous observation of the goal position, and the previous action.
obs_dims = 2*space_dims+act_dims

# random seed.
# the model is fully deterministic, so repeat runs with the same seed & settings produce the same result.
rng_seed = 2


# -------------------------
# GA settings
n_generations = 1500
n_population_size = 100
n_parent_pool = 25
n_elite = 5
# number of task instances to evaluate each individual on.
n_task_instances = 64
# number of trials to run per task instance.
n_trials_per_individual = 50
# number of steps (observe-act-update) cycles per trial.
n_trial_time_steps = 10


# -------------------------
# nn specification

# number of neurons in hidden layers (excluding bias neuron).
hidden_dims = 8

# number of input and output columns.
n_input_columns = 1 # values other than 1 not supported yet.
n_output_columns = 2 # values other than 2 not supported yet.

"""
initial_connectivity defines initial and allowed connectivity patterns, in matrix form. must be a square binary 2D matrix.
connectivity[i,j] specifies whether or not a neural projection going from neural column i to neural column j should be created when initialising new random individuals.
first neural column receives state input. second-to-last neural column is state-value (V) estimate output. last neural column is action output.
"""
initial_connectivity = \
    [[0,0,1,0,1,0,0],
     [0,0,0,0,0,0,0],
     [0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0],
     [0,0,0,0,0,0,1],
     [0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0]]

# use random initial connectivity instead of the connectivity specified above.
random_connectivity_initialisation = False

# disallows direct connection between input and output columns.
disallow_direct = True

# prevents mutations from producing networks that have no path from the input column to the output columns.
enforce_connectedness = True

# size of hidden column in NM subnets.
nm_fm_hidden_column_size = 16
nm_fg_hidden_column_size = 8

# activation function for input layer (applied to observations)
activation_function_obs = lambda a: a
# activation function for hidden layers
activation_function_hidden = jp.tanh
# activation function for value output
activation_function_value = lambda a: a
# activation function for mu part of action output
activation_function_mu = lambda a: a
# activation function for sigma part of action output
activation_function_sigma = lambda a: jp.exp(a)

# compound activation function for action output
def activation_function_action(a):
    mu = activation_function_mu(a[...,:act_dims])
    sigma = activation_function_sigma(a[...,act_dims:])
    return jp.concatenate((mu,sigma),axis=-1)


# -------------------------
# neuromodulation mechanism

# makes target value in the NM update relative to the genotypic weight.
nm_target_relative_to_geno_w = False
# toggles use of beta in NM update.
nm_beta_enabled = True
# toggles use of eta in NM update.
nm_eta_enabled = True
# sets whether eta operates at projection level or synapse level.
projection_level_eta = True
# sets the inputs that go into calculation of eta.
eta_sees_k = False
eta_sees_i = True
eta_sees_j = True
# limits the number of modulatory projections per activatory projection to one.
allow_only_one_nm_projection_per_a_projection = False
# forces all neuromodulatory connections to use the value output neuron as modulating neuron.
# this imposes an RL-like 1D bottleneck on NM-based learning ("bottlenecked-NM" in the paper).
# generally you'll want to leave this False (setting exists for purpose of comparison experiments only).
restrict_modulating_neurons_to_value_output = False

# activation function for hidden layers of the modulatory MLPs
activation_function_hidden_nm = jp.tanh
# activation function for target values
nm_update_activation_function_t = lambda a: a
# activation function for rate values
nm_update_activation_function_r = lambda a: jp.clip(a+0.5,0,1)
# activation function for gating values
nm_update_activation_function_g = lambda a: jp.clip(a+0.5,0,1)


# -------------------------
# reinforcement learning settings

# interval for running RL updates.
rl_weight_update_interval = [1,5,n_trial_time_steps][2]

# initialisation value for projection-level RL update rates
initial_rl_susceptibility = 1.0

# init value for RL update rate. use None to initialise randomly from range.
rl_learning_rate_init = 0.0
# RL update rates are constrained to this range
rl_learning_rate_range = [0.0,1.0]

# init value for sigma bias. must be larger than zero. use None to initialise randomly from range.
action_sigma_bias_init = 1e-7
# sigma bias values are constrained to this range.
action_sigma_bias_range = [1e-7,1e-7] # must be larger than 0 to avoid NaNs

# external optimiser to use for RL.
# if None, regular SGD or SignSGD is used (set use_grad_signs_only to True for SignSGD).
# default optimiser in Stable Baselines 3 is rmsprop.
rl_optimiser = [None,optax.sgd,optax.adam,optax.rmsprop][3]

# sets whether RL uses regular SGD or signSGD (True for signSGD).
# this setting has no effect when using an external optmiser
rl_signSGD = False

# sets whether to evolve the weight coefficients of the RL loss function or use fixed values
evolve_loss_weights = False

# maximum value for each RL loss weight coefficient (if evolvable)
max_actor_loss_weight = 2
max_critic_loss_weight = 2
max_entropy_loss_weight = 0.1

# fixed coefficients for the terms of the RL loss function.
# we use the defaults from the stable-baselines3 implementation of A2C.
# if coefficients are set to be evolvable, these settings are ignored and coefficients are initialised randomly.
actor_loss_weight = 1.0
critic_loss_weight = 0.5
entropy_loss_weight = 0.0
penalty_loss_weight = 0.0

# misc. RL settings
clip_gradients = True
grad_clip_norm = 0.5
rl_gamma = 0.99


# -------------------------
# evaluation mode settings

# override with special settings for evaluation mode.
if evaluation_mode:
    # evaluation uses a single generation of size one.
    n_generations = 1
    n_population_size = 1
    # evaluation uses 32 task instances of length 5000 each
    n_task_instances = 32
    n_trials_per_individual = 5000
    # RL initialisation settings for networks newly initialised in evaluation mode.
    # these settings are for the "pure RL" evaluation.
    rl_learning_rate_init = 0.0007
    rl_learning_rate_range = [0,0.01]
    action_sigma_bias_init = 1e-7
    action_sigma_bias_range = [0,1.0]


# -------------------------
# mutation rates
# mutation rate for all singleton genes (RL learning rate etc.).
singleton_genes_mutation_rate = 0.25
# strength of RL learning rate mutations.
rl_learning_rate_mutation_strength = 0.01
# strength of action sigma bias mutations.
# (note that the range for action sigma bias is [1e-7,1e-7] by default, this setting has no effect unless the range is modified.)
action_sigma_bias_mutation_strength = 0.1
# probabilities of activatory projection insertions, deletions, and weight mutations.
a_projection_insertion_mutation_rate = 0.05
a_projection_deletion_mutation_rate = 0.05
a_projection_weight_mutation_rate = 0.05
# probabilities of modulatory projection insertions, deletions, and weight mutations.
m_projection_insertion_mutation_rate = 0.2 # note that some proportion of (guided) insertions fail.
m_projection_deletion_mutation_rate = 0.2
m_projection_weight_mutation_rate = 0.2
# maximum weight mutation strength, relative to weight initialisation range.
# mutation strength is picked randomly from [0,max_weight_mutation_strength] for each weight mutation
max_weight_mutation_strength = 0.25
# probability of RL susceptibility mutation.
projection_rl_susceptibility_mutation_rate = 0.1
# maximum strength of RL susceptibility mutation.
rl_suscptibility_mutation_strength = 0.1
# probability of mutating the priority value that determines the application order of NM updates when an activatory projection is modulated by multiple modulatory projections.
# only meaningful when allowing multiple modulating projections per activatory projection.
priority_mutation_rate = 0 if allow_only_one_nm_projection_per_a_projection else 0.1


# -------------------------
# guided mutation settings
# whether to apply guided mutation on newly inserted modulatory projections.
use_guided_weight_init = True
# probability of applying guided mutation when mutating weights of existing activatory and modulatory projections.
guided_weight_mutation_rate = 0.5
# number of attempts for guided mutation.
# values over 1 allows up to n-1 retries when optimisation fails.
guided_m_insert_attempts = 1
# range of the number of optimisation iterations for newly inserted modulatory projections.
n_guided_weight_init_iteration_range = [100,500]
# range of the number of optimisation iterations for mutations of existing modulatory projections.
n_guided_weight_mutation_iteration_range = [10,100]
# when loss does not improve for this number of iterations, reduce the update rate.
guided_drop_learning_rate_at_stale_count = 10
# when loss does not improve for this number of iterations, cut optimisation short.
n_guided_weight_stop_at_stale_count = 50
# initial value for the update rate for initialisation of new modulatory projections.
lr_guided_weight_init = 0.1
# initial value for the update rate for mutation of existing modulatory projections.
lr_guided_weight_mutation = 0.1
# whether to use signSGD optimisation for guided mutation.
# if false, regular gradient descent is used instead.
guided_weight_init_signSGD = True
# the number of trials to use for calculating fitness weights in guided mutation.
# mean reward over the last n trials from each task instance are used to calculate the fitness weight for that task instance.
n_trials_to_use_in_guided_init_loss_weights = 5
# number of trials to use for determining convergence during guided mutation.
# should be smaller than the number of trials per individual.
# if this is zero, all trials are used for both optimisation and determining convergence (this risks overfitting).
n_validation = 16
# whether to ignore trials with fitness below a randomly picked threshold.
fitness_weight_clipping = True
# whether to optimise genotypic activatory weights together with modulatory weights during guided mutation.
optimise_geno_weights_in_guided_modification = False
# distance measure for guided mutation loss calculation.
# 'mse' = mean squared error. 'mae' = mean absolute error.
guided_weight_modification_diff_function = ['mse','mae'][1]
# whether to calculate loss directly over activatory weights or over post-synaptic activation patterns.
guided_weight_modification_loss_locus = ['weight','activation'][1]
# whether to apply the post-synaptic activation function when calculating the guided mutation loss.
# only in effect if loss is calculated over post-synaptic activation patterns.
# if the activation function is a squashing function, a large difference in raw activation may reduce to a small different after application of the activation function.
actvation_locus_loss_applies_activation_functions = True
# whether to consider input from other columns for calculating loss over post-synaptic activation patterns.
# this probably only makes a difference if loss is calculated over post-synaptic activation with the post-synaptic activation function applied.
actvation_locus_loss_considers_activation_from_elsewhere = True
# costs imposed on positive beta and eta values during guided mutation.
nm_beta_loss_weight = 0
nm_eta_loss_weight = 1e-5


# -------------------------
# verbosity control.
suppress_nn_prints = True


# -------------------------
# sanity checks

# sanity check on connectivity grid.
initial_connectivity = np.array(initial_connectivity)
assert initial_connectivity.ndim==2 and \
        initial_connectivity.shape[0]==initial_connectivity.shape[1] and \
        set(np.unique(initial_connectivity)) <= set((0,1)), \
        'config.connectivity must be a 2D square binary matrix'

# warn if n_trial_time_steps setting is problematic.
if n_trial_time_steps%rl_weight_update_interval != 0:
    print('[WARNING] Trial length (n_trial_time_steps) does not divide RL update interval (rl_weight_update_interval)')
    print('Such configurations are untested. Additional RL updates will be inserted at trial termination.')
    input('Press enter to continue...')
