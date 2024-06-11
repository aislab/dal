import os
import jax

# This version of the system is designed to run on CPU.
jax.config.update('jax_platforms', 'cpu')

import multiprocessing as mp
from multiprocessing.connection import Connection
import sys
import numpy as np
from jax import numpy as jp
import imageio
import time
from pathlib import Path
import copy
import argparse
import pickle
import shutil
from dataclasses import dataclass
import functools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import concurrent.futures
import signal
from queue import Empty

import config as cfg
import utils
import nmnn

from task_environment import Environment
if cfg.visualisation:
    from task_visualiser import Visualiser


# Global for holding on to the process pool.
process_pool_executor = None


# Colours used for plotting of the evolution process.
pink = np.array([138,43,226])/255
purple = np.array([255,192,203])/255


# Initialiser method for pool processes.
# Signal handler for ignoring SIGINT is added to allow clean exit on Control-C.
# Pool processes typically die on Control-C, which causes the process pool executor to hang when
# trying to exit. Letting the processes survive Control-C allows the executor to clean the pool up
# gracefully when exiting.
def process_initialiser():
    print('Initialising pool process with PID:', os.getpid())
    signal.signal(signal.SIGINT,signal.SIG_IGN)


def set_up_process_pool(n_processes):
    global process_pool_executor
    print('Main process (PID: ', os.getpid(), ') sets up process pool.',sep='')
    process_pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_processes,initializer=process_initialiser)
    #TODO: do we need this call?
    # Call pool with a dummy task to force the initialisation (jax import must happen before the main thread calls jax).
    x = process_pool_executor.map(lambda x:x,[None]*n_processes,chunksize=1)
    print('process pool initialised with', n_processes, 'processes.')


# Class Neural net wrapped in sub-process
class SPNN:
    # Identifier
    individual_id: int
    # Subprocess handle
    process: mp.Process
    # Pipe for communicate with the main process
    pipe: Connection
    # Pipe for direct information transfer between NNs
    peer_pipe: Connection
    # Indicates whether this NN is a mutant or not (largely coincides with being elite)
    is_unmutated_copy: bool
    # Handle for holding on to the fitness score when we retrieve it from the wrapped NN
    normalised_fitness: float = 0
    # Handle for holding on to a dict of statistics when we retrieve it from the wrapped NN
    stats: dict = None
    # Bool for marking broken nets during generation updates
    valid: bool = True
    
    
    def __init__(self,individual_id,process,pipe,peer_pipe,is_unmutated_copy):
        self.individual_id = individual_id
        self.process = process
        self.pipe = pipe
        self.peer_pipe = peer_pipe
        self.is_unmutated_copy = is_unmutated_copy
    
    
    # Convenience function to simplify calls to wrapped NNs
    def send(self,*args):
        self.pipe.send(args)
    
    
    # Convenience function to simplify receiving responses from wrapped NNs
    def recv(self):
        return self.pipe.recv()
    

# Sets up a subprocess-wrapped NN.
# If the nn is given, we wrap it in a process.
# If nn is None, we initialise a new one.
# main_pipe is used to communicate with the main process.
# peer_pipe is used for direct information transfer between NNs.
# Specifically, for sending genotype information for genetic operations.
def nn_subprocess(rng_key,main_pipe,peer_pipe,i,nn):
    
    # NNs run as subprocesses, on CPU.
    # Why CPU? NN architecture is evolvable, so we cannot assume all NNs in the population to have the computational graph.
    # Consequently we cannot simply vmap or pmap over the population to parallelise on GPU.
    # Processing a relatively large number of relatively small but heterogeneous computational graphs turned out to be 
    # more efficient to parallelise NNs over CPU cores than on GPU on our hardware (a machine with 128 CPU cores).
    # Note: It should in principle be possible to pad all NNs with dummy connection to a unified architecture and then 
    # parallelise on GPU, but we have not implemented that here.
    with jax.default_device(jax.devices("cpu")[0]):
        
        if nn is None:
            nn = nmnn.nn(rng_key,i,peer_pipe)
        else:
            nn.reset()
            
        nn.set_id_global()
        nn.peer_pipe = peer_pipe
        # redo setup of activation functions because functions cannot be pushed through pipes
        nn.set_activation_functions()
        
        while True:
            # NN waits for commands from main process
            command, *args = main_pipe.recv()
            try:
                # if the is a method of the NN, call it.
                f = eval('nn.'+command)
            except:
                # Special command for pulling the NN into the main process.
                if command == 'get_over_here':
                    # Optimiser and function attributes don't fit through pipes so we blank those out for a moment.
                    optimiser = nn.optimiser
                    opt_state = nn.opt_state
                    activation_functions = nn.activation_functions
                    nn.optimiser = None
                    nn.opt_state = None
                    nn.activation_functions = None
                    # Send the NN through the main pipe to the main process.
                    main_pipe.send(nn)
                    # Restore optimiser and function attributes.
                    nn.optimiser = optimiser
                    nn.opt_state = opt_state
                    nn.activation_functions = activation_functions
                # command for setting attribute values in the NN.
                elif command == 'set':
                    setattr(nn,args[0],args[1])
                # Termination commmand. This breaks the command loop, implicitly ending the subprocess.
                elif command == 'terminate':
                    break
                # If we end up here something went wrong...
                else:
                    print(' NN subprocess received unknown command type:', command)
                # If we hit the except clause, we are done with the current command and jump back to waiting for the next.
                continue
            
            # Command was a method of the NN class --> apply the method on the given arguments.
            ret = f(*args)
            # If we received a return value, send it to the main process.
            # Methods without a return statement return None by default.
            # Note: Be careful when adding methods to the NN class that conditionally return a single None value,
            # because that None will not be forwarded to the main process, potentially leaving the main process waiting forever.
            if ret is not None:
                main_pipe.send(ret)
        
        # If we popped out of the command loop the subprocess terminates here.
        if not cfg.suppress_nn_prints:
            print('NN#'+str(nn.individual_id), 'terminates')


# Initialise a population of NNs wrapped into subprocesses.
# If a list of NNs is given, we wrap those NNs (used when loading an existing population).
# Otherwise, we initialise a new population of randomly initialised NNs, of population size n.
# If n is not given, the default population size is used.
def initialise_spnn_population(rng_key=None,nn_pop=None,n=None):
    if nn_pop is None:
        print('initialising subprocess-wrapped NN population')
        if n is None:
            n = cfg.n_population_size
        rng_key, *pop_keys = jax.random.split(rng_key,1+cfg.n_population_size)
    else:
        print('wrapping NN population in subprocesses')
        n = len(nn_pop)
    spnn_pop = []
    for i in range(n):
        utils.print_progress(i,n)
        nn = None if nn_pop is None else nn_pop[i]
        k = None if rng_key is None else pop_keys[i]
        main_pipe_end, subprocess_main_pipe_end = mp.Pipe(True)
        peer_pipe_end, subprocess_peer_pipe_end = mp.Pipe(True)
        if nn_pop is None:
            is_unmutated_copy = True 
        else:
            try:
                is_unmutated_copy = nn.is_unmutated_copy
                del nn.is_unmutated_copy # managed in main process
            except:
                is_unmutated_copy = True
        sp = mp.Process(target=nn_subprocess,args=(k,subprocess_main_pipe_end,subprocess_peer_pipe_end,i,nn),daemon=True)
        sp.start()
        spnn_pop.append(SPNN(i,sp,main_pipe_end,peer_pipe_end,is_unmutated_copy))
    utils.print_progress()
    return spnn_pop
        

# Rewraps a population of subprocess-wrapped NNs.
# Subprocess memory usage tends to increase gradually (probably due to JAX compilation cache growth).
# To keep memory usage in check, we rewrap the population every now and then, letting the old subprocesses terminate and starting fresh ones.
def rewrap_population(pop):
    new_pop = []
    for i, spnn in enumerate(pop):
        utils.print_progress(i,len(pop))
        while True:
            try:
                spnn.send('get_over_here')
                break
            except BrokenPipeError:
                print('main failed to pull', i, 'through pipe (BrokenPipeError) --> wait and retry')
                time.sleep(1)
        nn = spnn.recv()
        nn.peer_pipe = None
        main_pipe_end, subprocess_main_pipe_end = mp.Pipe(True)
        peer_pipe_end, subprocess_peer_pipe_end = mp.Pipe(True)
        sp = mp.Process(target=nn_subprocess,args=(None,subprocess_main_pipe_end,subprocess_peer_pipe_end,spnn.individual_id,nn),daemon=True)
        sp.start()
        new_pop.append(SPNN(spnn.individual_id,sp,main_pipe_end,peer_pipe_end,spnn.is_unmutated_copy))
    # Terminate old processes in separate loop to avoid broken pipe errors.
    for i, spnn in enumerate(pop):
        spnn.send('terminate')
    utils.print_progress()
    return new_pop


# Runs a single trial with the given population.
def run_trial(env_reset_key,action_keys,i_generation,i_trial,pop,t_prev_trial_start):
    
    n_pop = len(pop)
    t_trial_start = time.time()
    trial_time_cost = 'time: '+str(np.round(t_trial_start-t_prev_trial_start,3)) if i_trial else 'time: ---'
    t_prev_trial_start = t_trial_start
    utils.print_progress(i_trial,cfg.n_trials_per_individual,message=trial_time_cost)
    draw = cfg.visualisation and (not cfg.evaluation_mode) and (i_generation%10==0) and (i_trial==0 or (i_trial+1)%10==0)
    
    trial_reward = 0
    state = env.reset(env_reset_key,n_pop)
    if draw:
        vis.reset(state.goal_pos,state.agent_pos)

    prev_obs = None
    prev_action = np.zeros((n_pop,cfg.n_task_instances,cfg.act_dims))
    for i_step in range(cfg.n_trial_time_steps):
        
        obs = env.get_observation(state)
        
        if prev_obs is None: prev_obs = obs
        
        # extended observation consists current and previous observation and previous action
        ext_obs = np.concatenate((obs,prev_obs,prev_action),axis=-1)
        
        action_key = action_keys[i_trial,i_step]
        action = []
        for i_individual in range(n_pop):
            pop[i_individual].send('choose_action',action_key,ext_obs[i_individual])
        for i_individual in range(n_pop):
            action.append(pop[i_individual].recv())
        action = np.array(action)
            
        state = env.step(state,action)
        new_obs = env.get_observation(state)
        
        ext_new_obs = np.concatenate((new_obs,obs,action),axis=-1)
            
        reward = env.get_reward(state)
        
        trial_reward += reward
        
        done = (i_step == cfg.n_trial_time_steps-1)

        if cfg.rl_enabled:
            for i_individual in range(n_pop):
                pop[i_individual].send('reinforcement_learning_update',action_key,ext_obs[i_individual],ext_new_obs[i_individual],reward[i_individual],done)
        
        if cfg.nm_enabled and i_step:
            for i_individual in range(n_pop):
                pop[i_individual].send('neuromodulation_update')
        
        for i_individual in range(n_pop):
            pop[i_individual].send('record_weights')
            
        prev_obs = obs
        prev_action = action
        
    if draw:
        agent_pos = state.agent_pos
        vis.draw_reward(agent_pos[0],reward[0])
        image = vis.prepare_image()
    else:
        image = None
    
    trial_fitness = env.get_trial_fitness(state)
    return trial_fitness, trial_reward, state, t_prev_trial_start, image


def eval_generation(rng_key_for_env,rng_key_for_actions,i_generation,pop,aux_pop=None):
    
    if aux_pop is not None:
        aux_pop_size = len(aux_pop)
        if type(pop) is not list: pop = [pop]
        pop += aux_pop
    
    n_pop = len(pop)
        
    rng_key, k = jax.random.split(rng_key_for_env)
    env.randomise_task(k,n_pop)
    rng_key, env_reset_key = jax.random.split(rng_key_for_env)
    original_action_key = rng_key_for_actions
    rng_key_for_actions, *action_keys = jax.random.split(rng_key_for_actions,1+cfg.n_trials_per_individual*cfg.n_trial_time_steps)
    action_keys = np.array(action_keys).reshape((cfg.n_trials_per_individual,cfg.n_trial_time_steps,2))
    
    if cfg.evaluation_mode:
        trial_score_record = []
        smooth_trial_reward = 0
        smooth_trial_reward_length = 0
        best_smooth = -np.inf
        evaluation_log = open('evaluation_log.csv','a')
    
    for i_individual in range(n_pop):
        pop[i_individual].env_reset_key = rng_key_for_env
        pop[i_individual].action_key = original_action_key
    
    t = 0
    t_last = cfg.n_trials_per_individual*cfg.n_trial_time_steps
    images = []
    t_prev_trial_start = None
    for i_trial in range(cfg.n_trials_per_individual):
        env_reset_key, k = jax.random.split(env_reset_key)
        trial_fitness, trial_reward, state, t_prev_trial_start, image = run_trial(k,action_keys,i_generation,i_trial,pop,t_prev_trial_start)
        if image is not None: images.append(image)
        
        for i_individual in range(n_pop):
            pop[i_individual].send('add_fitness',trial_fitness[i_individual])
        
        trial_score = trial_reward.mean()#/cfg.n_trial_time_steps
        if cfg.evaluation_mode:
            trial_score_record.append(trial_reward)
            print('trial', i_trial, 'mean reward:', trial_score, '            ')
            
            for i, r in enumerate(trial_reward):
                print(i, r.mean(), r)

            print()
            if cfg.evaluation_mode:
                evaluation_log.write(str(i_trial)+' '+str(trial_score)+'\n')
                evaluation_log.flush()
        
    utils.print_progress()
    if len(images):
        im = np.concatenate(images,axis=1).swapaxes(0,1)
        imageio.imwrite('behaviour_'+str(i_generation).zfill(5)+'.png',im)
    
    if cfg.evaluation_mode:
        trial_score_record = np.array(trial_score_record)[:,0]
        np.save('evaluation_log',trial_score_record)
    
    if aux_pop is None:
        return pop, aux_pop
    return pop[:-aux_pop_size], pop[-aux_pop_size:]
    
    
    
# time series
ts_focal_fitness = [] 
ts_focal_fitness_rl_only = []
ts_focal_fitness_nm_only = []
ts_unmutated_fitness = []
ts_mean_fitness = []
ts_learning_progress = []
ts_rl_learning_rate = []
ts_action_sigma_bias = []
ts_learning_type_ratio = []
ts_weight_change_nm = []
ts_weight_change_rl = []
       
def next_generation(rng_key,evo_log,pop,i_generation,aux_pop=None):
    print('generate next generation with rng_key:', rng_key)
    
    mean_fitness = 0
    n_unmutated_copies = 0
    unmutated_fitness = 0
    
    for nn in (pop if aux_pop is None else pop+aux_pop):
        nn.send('report_stats')
        nn.stats = nn.recv()
        nn.normalised_fitness = nn.stats['normalised_fitness']
        if np.isnan(nn.normalised_fitness):
            nn.normalised_fitness = -np.inf
            nn.valid = False
        else:
            nn.valid = True
        
    pop[0].send('get_over_here')
    focal = pop[0].recv()
    
    if aux_pop is not None:
        dummy_nm_only = aux_pop[0]
        dummy_rl_only = aux_pop[1]
    pop = sorted(pop,key=lambda p:p.normalised_fitness,reverse=True)
    
    # set up parent retrieval for later
    for i_individual in range(cfg.n_parent_pool,cfg.n_population_size):
        i_parent = i_individual%cfg.n_parent_pool
        pop[i_parent].send('send_genotype_to_child',pop[i_individual].peer_pipe)
        
    if aux_pop is not None:
        # set up test dummy population for next generation
        pop[0].send('send_genotype_to_child',dummy_nm_only.peer_pipe)
        pop[0].send('send_genotype_to_child',dummy_rl_only.peer_pipe)
        dummy_nm_only.send('clone_from_parent')
        dummy_rl_only.send('clone_from_parent')
        aux_pop = [dummy_nm_only,dummy_rl_only]
    
    print('\nGeneration', i_generation, 'fitness ranking:')
    n_valid = 0
    for i_individual in range(cfg.n_population_size):
        marker = ''
        if i_individual < cfg.n_elite: marker += 'elite '
        if i_individual < cfg.n_parent_pool: marker += 'parent'
        mu_list = pop[i_individual].stats['mutations_applied']
        print('  rank', i_individual, marker)
        print('    id:', pop[i_individual].individual_id,' fitness:', pop[i_individual].normalised_fitness)
        print('    mutations:', *mu_list if mu_list else '-')
        if pop[i_individual].valid:
            mean_fitness += pop[i_individual].normalised_fitness
            n_valid += 1
        if pop[i_individual].is_unmutated_copy:
            n_unmutated_copies += 1
            unmutated_fitness += pop[i_individual].normalised_fitness
    
    unmutated_fitness /= n_unmutated_copies
    learning_progress = np.array(focal.trial_fitness_log).mean(1)
        
    mean_fitness /= n_valid
    print('mean fitness:', mean_fitness)
    if n_valid<cfg.n_population_size:
        print('[WARNING] population contains', cfg.n_population_size-n_valid, 'invalid individual(s)')
    
    print('\nGeneration', i_generation, 'focal individual properties:')
    print('  id:', focal.individual_id)
    print('  fitness:', focal.normalised_fitness)
    if aux_pop is not None:
        print('  NM-only fitness:', dummy_nm_only.normalised_fitness)
        print('  RL-only fitness:', dummy_rl_only.normalised_fitness)
    print('  best trial:', np.max(focal.trial_fitness_log))
    print('  action sigma bias:', focal.action_sigma_bias)
    print('  RL learning rate:', focal.mean_active_rl_learning_rate)
    print('  actor_loss_weight:', focal.actor_loss_weight)
    print('  critic_loss_weight:', focal.critic_loss_weight)
    print('  entropy_loss_weight:', focal.entropy_loss_weight)
    print('  weight change:')
    print('    nm:', focal.total_weight_change_nm)
    print('    rl:', focal.total_weight_change_rl)
    print('  Weight stats:', focal.report_weight_stats())
    print('  most recent mutation in lineage occurred in generation:', focal.generation_of_most_recent_mutation)
    print()
    print('activation connectivity grid:')
    print(focal.connectivity_grid_a)
    print()
    for (i,o,sus) in focal.io_list_a:
        i_nm = list(np.where(focal.connectivity_grid_m[:,i,o])[0])
        print(i, '-->', o, '/ rl-susceptibility:', sus, '/', 'modulated by:', i_nm)
        
    for (i,o,sus) in focal.io_list_a:
        w = focal.geno_projections_a[i,o].w
        print(i, '-->', o, '/ geno w min,mean,max:', w.min(), w.mean(), w.max())
    print()
        
    # update time series
    log_str = str(i_generation)+', '
    ts_focal_fitness.append(focal.normalised_fitness)
    log_str += str(ts_focal_fitness[-1])+', '
    ts_unmutated_fitness.append(unmutated_fitness)
    log_str += str(ts_unmutated_fitness[-1])+', '
    ts_mean_fitness.append(mean_fitness)
    log_str += str(ts_mean_fitness[-1])+', '
    ts_weight_change_nm.append(focal.total_weight_change_nm)
    log_str += str(ts_weight_change_nm[-1])+', '
    ts_weight_change_rl.append(focal.total_weight_change_rl)
    log_str += str(ts_weight_change_rl[-1])+', '
    ts_rl_learning_rate.append(focal.mean_active_rl_learning_rate)
    log_str += str(ts_rl_learning_rate[-1])+', '
    ts_action_sigma_bias.append(focal.action_sigma_bias)
    log_str += str(ts_action_sigma_bias[-1])+', '
    total_weight_change = (focal.total_weight_change_nm+focal.total_weight_change_rl)
    if total_weight_change>0:
        ts_learning_type_ratio.append(focal.total_weight_change_nm/total_weight_change)
    else:
        ts_learning_type_ratio.append(0)
    log_str += str(ts_learning_type_ratio[-1])+', '
    
    if aux_pop is not None:
        ts_focal_fitness_nm_only.append(dummy_nm_only.normalised_fitness)
        ts_focal_fitness_rl_only.append(dummy_rl_only.normalised_fitness)
    else:
        ts_focal_fitness_nm_only.append(None)
        ts_focal_fitness_rl_only.append(None)
    log_str += str(ts_focal_fitness_nm_only[-1])+', '
    log_str += str(ts_focal_fitness_rl_only[-1])+', '
    
    ts_learning_progress.append(learning_progress)
    
    for i in range(cfg.n_trials_per_individual):
        log_str += str(ts_learning_progress[-1][i])+', '
    
    evo_log.write(log_str[:-2]+'\n')
    evo_log.flush()
    
    # reorder non-parents to avoid unintentional sorting effects
    rng_key, k = jax.random.split(rng_key)
    clone_order = jax.random.permutation(k,np.arange(cfg.n_parent_pool,cfg.n_population_size))
    pop = pop[:cfg.n_parent_pool]+[pop[i] for i in clone_order]
    
    for i_individual in range(cfg.n_elite):
        pop[i_individual].is_unmutated_copy = True
    
    print('plotting...')
    plt.clf()
    # plot generation data
    t = np.arange(i_generation+1-len(ts_focal_fitness),i_generation+1)
    ar_learning_progress = np.array(ts_learning_progress)
    emphasis_interval = 5
    for i in range(cfg.n_trials_per_individual):
        r = i/(cfg.n_trials_per_individual-1)
        c = tuple(r*pink+(1-r)*purple)
        emphasis = i%emphasis_interval==emphasis_interval-1
        width = 1.0 if emphasis else 0.5
        plt.plot(t,ar_learning_progress[:,i],color=c,lw=width)
    plt.plot(t,ts_mean_fitness,'black')
    plt.plot(t,ts_focal_fitness_rl_only,'darkred')
    plt.plot(t,ts_focal_fitness_nm_only,'magenta')
    plt.plot(t,ts_focal_fitness,'red')
    plt.plot(t,ts_unmutated_fitness,'orange')
    div = max(np.max(ts_weight_change_nm),1)
    plt.plot(t,list(np.array(ts_weight_change_nm)/div),'lightblue')
    div = max(np.max(ts_weight_change_rl),1)
    plt.plot(t,list(np.array(ts_weight_change_rl)/div),'lightgreen')
    norm_rl = np.array(ts_rl_learning_rate)
    max_norm_rl = norm_rl.max()
    if max_norm_rl>0: norm_rl /= max_norm_rl
    plt.plot(t,norm_rl,'limegreen')
    plt.plot(t,ts_action_sigma_bias,'darkgreen')
    plt.plot(t,ts_learning_type_ratio,'grey')
    ymin, ymax = plt.gca().get_ylim()
    plt.yticks(np.arange(np.round(ymin,1),ymax,0.1))
    plt.grid()
    plt.draw()
    plt.pause(0.001)
    
    # apply cloning and mutation to obtain next generation
    print('cloning...')
    for i_individual in range(cfg.n_parent_pool,cfg.n_population_size):
        i_parent = i_individual%cfg.n_parent_pool
        pop[i_individual].send('clone_from_parent')
    
    print('mutating...')
    n_mutations = 0
    n_mutated_individuals = 0
    for i_individual in range(cfg.n_elite,cfg.n_population_size):
        pop[i_individual].send('mutate',i_generation+1)
        pop[i_individual].is_unmutated_copy = False
        
    print('**** done instructing mutations ****')
        
    for i_individual in range(cfg.n_elite,cfg.n_population_size):
        n_mutations += len(pop[i_individual].recv())
        n_mutated_individuals += 1

    mean_mutations_per_mutated_individual = n_mutations/n_mutated_individuals
    print('\nmean mutations per mutated individual:', mean_mutations_per_mutated_individual)
    
    # reset everyone to newborn state
    for i_individual in range(cfg.n_population_size):
        pop[i_individual].send('reset')
    if aux_pop is not None:
        for individual in aux_pop:
            individual.send('reset')
        
    return pop, aux_pop


def save_population(rng_key,pop,i_generation):
    fname = 'generation'+str(i_generation).zfill(5)
    print('saving population state to:\n',fname)
    
    nn_pop = []
    for spnn in pop:
        spnn.send('get_over_here')
        nn = spnn.recv()
        nn.is_unmutated_copy = spnn.is_unmutated_copy
        nn.peer_pipe = None
        nn_pop.append(nn)
    
    state = {'rng_key': rng_key,
             'population': nn_pop}
    with open(fname,'wb') as f:
        pickle.dump(state,f)
    
    print('save completed')


def load_population(i_generation,n_truncate=None):
    fname = 'generation'+str(i_generation).zfill(5)
    print('loading population state from:\n', fname)
    with open(fname,'rb') as f:
        state = pickle.load(f)
        
    pop = state['population'][:n_truncate]
    for nn in pop:
        nn.rl_enabled = cfg.rl_enabled
        nn.nm_enabled = cfg.nm_enabled
        nn.rl_learning_rate = np.clip(nn.rl_learning_rate,cfg.rl_learning_rate_range[0],cfg.rl_learning_rate_range[1])
        nn.action_sigma_bias = np.clip(nn.action_sigma_bias,cfg.action_sigma_bias_range[0],cfg.action_sigma_bias_range[1])
    
    if len(pop)>cfg.n_population_size:
        print('population in state file larger than population size set in configuration file --> truncating population')
        pop = pop[:cfg.n_population_size]
        
    if n_truncate is None and len(pop)<cfg.n_population_size:
        print('population in state file smaller than population size set in configuration file --> padding population with duplicates')
        n_loaded = len(pop)
        n_add = cfg.n_population_size-len(pop)
        for i in range(n_add):
            pop.append(copy.deepcopy(pop[i%n_loaded]))
    
    spnn_pop = initialise_spnn_population(nn_pop=pop)
    return state['rng_key'], spnn_pop
    

# sets up the environment and its visualiser
def set_up_environment():
    global env, vis
    env = Environment()
    if cfg.visualisation:
        vis = Visualiser()
    # return value is used by an analysis script
    return env


if __name__ == '__main__':
    
    # set forking method for multiprocessing
    mp.set_start_method('forkserver')
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('from_generation',type=int,default=0,nargs='?')
    parser.add_argument('-u','--update',dest='update_source_files',default=False,action='store_true',
                        help='When resuming an existing run, update the source files (default: false).')
    args = parser.parse_known_args()[0]
    
    print('launching run')
    
    cwd = os.getcwd()
    
    # set up initial population
    if args.from_generation == 0:
        # initialise RNG
        rng_key = utils.init_rng()
        rng_key, k = jax.random.split(rng_key)
        pop = initialise_spnn_population(k);
    else:
        rng_key, pop = load_population(args.from_generation)
    
    # open a file for logging the evolution process.
    # this file can be plotted later to produce figures.
    if not cfg.evaluation_mode: 
        evo_log = open('evolution_log.csv','a')
    
    # if we run evolution with both RL and NM enabled, we create an auxiliary population consisting of one
    # individual with RL disabled (NM-only in the paper) and one individual with NM disabled (RL-only in the paper)
    if cfg.rl_enabled and cfg.nm_enabled and (not cfg.evaluation_mode):
        print('setting up test dummies for tracking...')
        pop[0].send('get_over_here')
        nn = pop[0].recv()
        print('dummy source NN received...')
        aux_pop = initialise_spnn_population(nn_pop=[nn,nn])
        print('sending set commands...')
        aux_pop[0].send('set','rl_enabled',False)
        aux_pop[1].send('set','nm_enabled',False)
        print('test dummies ok')
    else:
        aux_pop = None
    
    # set up the environment and its visualiser
    set_up_environment()
    
    # matplotlib setup for progress plotting during evolution
    plt.ion()
    plt.show()
    
    # loop over generations
    generation_range = [0] if cfg.evaluation_mode else range(args.from_generation,cfg.n_generations+1)
    for i_generation in generation_range:
        
        print('---------')
        print('RUN:', cwd)
        print('GENERATION:', i_generation, '[RNG state:', rng_key,']')
        
        # RNG keys for the generation
        rng_key, k_env, k_act = jax.random.split(rng_key,3)
        
        # evaluate the current population
        pop, aux_pop = eval_generation(k_env,k_act,i_generation,pop,aux_pop=aux_pop)
        
        # in evaluation mode we only run one singleton generation so we bail out here
        if cfg.evaluation_mode: break
        
        # make the next generation
        rng_key, k = jax.random.split(rng_key)
        pop, aux_pop = next_generation(k,evo_log,pop,i_generation,aux_pop=aux_pop)
        plt.savefig('plot_'+str(args.from_generation).zfill(4)+'.png')
        
        # if we hit the save interval, save the population to disk
        if i_generation%cfg.save_interval==0:
            save_population(rng_key,pop,i_generation)
        
        # if we hit the housekeeping interval, apply housekeeping routines
        if i_generation>args.from_generation and i_generation%cfg.housekeeping_interval==0:
            print('housekeeping time!')
            print('clearing JAX caches...')
            jax.clear_caches()
            print('caches cleared')
            print('rewrapping population...')
            pop = rewrap_population(pop)
            if aux_pop is not None:
                aux_pop = rewrap_population(aux_pop)
            print('rewrap finished')
            
    # save final population
    if not cfg.evaluation_mode:
        save_population(rng_key,pop,i_generation)
    
    # shut down subprocesses
    for i, spnn in enumerate(pop):
        spnn.send('terminate')
    for i, spnn in enumerate(pop):
        spnn.process.join()
    
    print(f'\nRUN COMPLETED ({cfg.n_generations} generations)')
    
