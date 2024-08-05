import os
import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import jax
from jax import numpy as jp
import config as cfg
import utils
import evolve
import task_environment as env_module

run_name = sys.argv[1]
generations_to_analyse = [0,10,100,500,1000,1500]

prop_cycle = plt.rcParams['axes.prop_cycle']
colours = prop_cycle.by_key()['color']

def analyse():
    
    fixed_rng_key = utils.init_rng()
    env = evolve.set_up_environment()
    
    for i_generation in generations_to_analyse:
        
        print('\ngeneration:', i_generation)
        _rng_key, pop = evolve.load_population(i_generation,1)
        pop[0].send('reset')
        n_pop = len(pop)

        rng_key, k = jax.random.split(fixed_rng_key)
        env.randomise_task(k,n_pop)
        
        rng_key, k_env = jax.random.split(rng_key)
        rng_key, k_act = jax.random.split(rng_key)

        rng_key, env_reset_key = jax.random.split(k_env)
        rng_key_for_actions, *action_keys = jax.random.split(k_act,1+cfg.n_trials_per_individual*cfg.n_trial_time_steps)
        action_keys = jp.array(action_keys).reshape((cfg.n_trials_per_individual,cfg.n_trial_time_steps,2))
        i_trial = 0
        t_prev_trial_start = None
        
        print('running first trial')
        trial_fitness, trial_reward, state, t_prev_trial_start, ims = evolve.run_trial(env_reset_key,action_keys,i_generation,i_trial,pop,t_prev_trial_start)
        utils.print_progress()
        
        pop[0].send('report_action_history')
        action_history = pop[0].recv()
        
        plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal',adjustable='box')
        ax.axis('off')
        plt.xlim([-1.05,1.05])
        plt.ylim([-1.05,1.05])
        plt.xticks(jp.arange(-1.0,1.1,0.5))
        plt.yticks(jp.arange(-1.0,1.1,0.5))
        
        for i_instance in range(8):
            actions = env_module.action_shaping(action_history[:,0,i_instance])/cfg.n_trial_time_steps
            action_lengths = jp.linalg.norm(actions,axis=-1)
            positions = jp.cumsum(actions,axis=0)
            
            positions = jp.concatenate((jp.zeros((1,2)),positions),axis=0)
            loc = env_module.action_shaping(action_history[:,1,i_instance])/cfg.n_trial_time_steps
            scale = action_history[:,2,i_instance]/cfg.n_trial_time_steps
            ellipse = Ellipse(xy=(0,0),
                              width=2,
                              height=2,
                              edgecolor='lightgrey',
                              fill=False,
                              zorder=1)
            ax.add_patch(ellipse)
            for i_step in range(cfg.n_trial_time_steps):
                ellipse = Ellipse(xy=(positions[i_step,0]+loc[i_step,0],positions[i_step,1]+loc[i_step,1]),
                                  width=scale[i_step,0],
                                  height=scale[i_step,1],
                                  edgecolor='lightgrey',
                                  facecolor='lightgrey',
                                  zorder=2)
                ax.add_patch(ellipse)
            
            plt.plot(positions[:,0],positions[:,1],zorder=3)
            plt.scatter(positions[:,0],positions[:,1],zorder=5,c='black',s=1)
            goal = state.goal_pos[0,i_instance]
            plt.scatter(goal[0],goal[1],c=colours[i_instance],zorder=6)
            
            
        image_path = 'analysis_gen'+str(i_generation)+'.svg'
        plt.savefig(image_path,format='svg',bbox_inches='tight')
        print('image saved to:\n ', os.getcwd()+'/'+image_path)
        plt.clf()
        
        pop[0].send('terminate')
        
    print('\ncompleted\n')

if __name__ == '__main__':
    # set_start_method must be inside main clause
    mp.set_start_method('forkserver')
    os.chdir('runs/'+run_name)
    analyse()
