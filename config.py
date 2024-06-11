import jax
from jax import numpy as jp
from jax import custom_jvp
import optax
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-e','--eval',dest='evaluation_mode',default=False,action='store_true',
                    help='Run in evaluation mode (default: False).')
args = parser.parse_known_args()[0]


source_copy_list = ['config.py','utils.py','evolve.py','nmnn.py','task_environment.py','task_visualiser.py']

evaluation_mode = args.evaluation_mode

visualisation = True
save_interval = 10
housekeeping_interval = 10

nm_enabled = True
rl_enabled = True

# environment settings
space_dims = 2
act_dims = space_dims
obs_dims = 2*space_dims+act_dims

# nn specification
hidden_dims = 8
n_input_columns = 1
n_output_columns = 2

# size of single hidden column in direct-to-synapse NM subnet.
nm_hidden_column_size = 8

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

# disallows direct connection between input and output columns.
disallow_direct = True

# initialisation value for projection-level RL update rates
initial_rl_susceptibility = 1.0

# limits the number of modulatory projections per activatory projection to one.
allow_only_one_nm_projection_per_a_projection = False

# prevents mutations from producing networks that have no path from the input column to the output columns.
enforce_connectedness = True

# forces all neuromodulatory connections to use the value output neuron as modulating neuron.
# this imposes an RL-like 1D bottleneck on NM-based learning ("bottlenecked-NM" in the paper).
# generally you'll want to leave this False (setting exists for purpose of comparison experiments only).
restrict_modulating_neurons_to_value_output = False

# random seed.
# the model is fully deterministic, so repeat runs with the same seed & settings produce the same result.
rng_seed = 2

# GA settings
n_generations = 1500
n_population_size = 100
n_task_instances = 64
n_trials_per_individual = 50
n_parent_pool = 25
n_elite = 5

# number of steps per trial
n_trial_time_steps = 10
    
# interval for running RL updates.
rl_grad_application_interval = [1,5,n_trial_time_steps][2]

# init value for RL update rate. use None to initialise randomly from range.
rl_learning_rate_init = 0.0
# RL update rates are constrained to this range
rl_learning_rate_range = [0.0,1.0]
# init value for sigma bias. must be larger than zero. use None to initialise randomly from range.
action_sigma_bias_init = 1e-7
# sigma bias values are constrained to this range.
action_sigma_bias_range = [1e-7,1e-7] # must be larger than 0 to avoid NaNs

# override with special settings for evaluation mode.
if evaluation_mode:
    # evaluation uses a single generation of size one.
    n_generations = 1
    n_population_size = 1
    # evaluation uses 10 task instances of length 5000 each
    n_task_instances = 10
    n_trials_per_individual = 5000
    # RL initialisation settings for networks newly initialised in evaluation mode.
    # these settings are for the "pure RL" evaluation.
    rl_learning_rate_init = 0.0007
    rl_learning_rate_range = [0,0.01]
    action_sigma_bias_init = 1e-7
    action_sigma_bias_range = [0,1.0]

# mutation rates
singleton_genes_mutation_rate = 0.25
rl_learning_rate_mutation_strength = 0.01
action_sigma_mutation_strength = 0.1
a_projection_insertion_mutation_rate = 0.05
a_projection_deletion_mutation_rate = 0.05
a_projection_weight_mutation_rate = 0.05
m_projection_insertion_mutation_rate = 0.2 # some proportion of (guided) insertions fail.
m_projection_deletion_mutation_rate = 0.2
m_projection_weight_mutation_rate = 0.2
max_weight_mutation_strength = 0.25 # relative to weight initialisation range.
projection_rl_susceptibility_mutation_rate = 0.1
rl_suscptibility_mutation_strength = 0.1
priority_mutation_rate = 0 if allow_only_one_nm_projection_per_a_projection else 0.1

# guided initialisation settings
use_guided_weight_init = 1.0
guided_m_insert_attempts = 1
n_guided_weight_init_iteration_range = [100,500]
n_guided_weight_mutation_iteration_range = [10,100]
guided_drop_learning_rate_at_stale_count = 10
n_guided_weight_stop_at_stale_count = 50
guided_weight_mutation_rate = 0.5
lr_guided_weight_init = 0.1
lr_guided_weight_mutation = 0.1
guided_weight_init_signSGD = 1
max_target_noise_range = 0.1
n_trials_to_use_in_guided_init_loss_weights = 5
n_validation = 16
fitness_weight_clipping = True
optimise_geno_weights_in_guided_modification = False
guided_weight_modification_diff_function = ['mse','mae'][1]
guided_weight_modification_loss_locus = ['weight','activation'][1]
actvation_locus_loss_considers_activation_from_elsewhere = True
actvation_locus_loss_applies_activation_functions = True
nm_rate_loss_weight = 0
nm_gate_loss_weight = 1e-5

"""
initial_connectivity defines initial and allowed connectivity patterns, in matrix form. must be a square binary 2D matrix.
connectivity[i,j] specifies whether or not a neural projection going from neural column i to neural column j should be created when initialising new random individuals.
first neural column receives state input.
second-to-last neural column is state-value (V) estimate output.
last neural column is action output.
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

# activation functions
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

# activation function for computing nm updates
# activation function for hidden layers of the modulatory MLPs
activation_function_hidden_nm = jp.tanh
# activation function for target values
nm_update_activation_function_t = lambda a: a
# activation function for rate values
nm_update_activation_function_r = lambda a: jp.clip(a+0.5,0,1)
# activation function for gating values
nm_update_activation_function_g = lambda a: jp.clip(a+0.5,0,1)

# external optimiser to use for RL.
# if None, regular SGD or SignSGD is used (set use_grad_signs_only to True for SignSGD).
# default optimiser in Stable Baselines 3 is rmsprop.
rl_optimiser = [None,optax.sgd,optax.adam,optax.rmsprop][3]

# sets whether RL uses regular SGD or signSGD (True for signSGD).
# this setting has no effect when using an external optmiser
rl_signSGD = False

# misc. RL settings
clip_gradients = True
grad_clip_norm = 0.5
rl_gamma = 0.99

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

# verbosity control.
suppress_nn_prints = True

# warn if n_trial_time_steps setting is problematic.
if n_trial_time_steps%rl_grad_application_interval != 0:
    print('[WARNING] Trial length (n_trial_time_steps) does not divide RL update interval (rl_grad_application_interval)')
    print('Such configurations are untested. Additional RL updates will be inserted at trial termination.')
    input('...')
