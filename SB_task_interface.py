import numpy as np
import jax
import torch
import gymnasium as gym
from gymnasium import spaces
from task_environment import Environment
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import sys
import config as cfg
from utils import make_dir

verbose = False
fixed_goal_position = False
task_randomisation_seed = 0

def pprint(args):
    if verbose:
        print(args)


class SB_env_interface(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,rng_key):
        super().__init__()
        self.action_space = spaces.Box(low=-1,high=1,shape=(cfg.act_dims,),dtype=np.float32)
        self.observation_space = spaces.Box(low=-2,high=2,shape=(cfg.obs_dims,),dtype=np.float32)
        self.env = Environment()
        self.env.randomise_task(rng_key,1)


    def step(self, action):
        
        prev_obs = self.env.get_observation(self.state)
        self.state = self.env.step(self.state,action[None,None])
        obs = self.env.get_observation(self.state)
        observation = np.concatenate((obs[0,0],prev_obs[0,0],self.prev_action))
        
        self.prev_action = action
        
        reward = float(self.state.reward[0,0])
        pprint(f'step: {self.step_index} action: {action} reward:{reward}')
        pprint(f'new observation:{observation}')
        
        self.step_index += 1
        terminated = (self.step_index==cfg.n_trial_time_steps)
        
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None, fixed_goal_position=False):
        pprint(f'reset with seed: {seed}')
        if seed is None:
            seed = 0 if fixed_goal_position else np.random.randint(np.iinfo(np.int64).max)
        key = jax.random.PRNGKey(seed)
        self.state = self.env.reset(key,1)
        self.step_index = 0
        agent_pos = self.state.agent_pos
        self.prev_action = np.zeros(cfg.act_dims,dtype=np.float32)
        observation = np.concatenate((agent_pos[0,0],agent_pos[0,0],self.prev_action))
        info = {}
        
        return observation, info
    
    
    def get_reward(self):
        reward = float(self.state.reward[0,0])
        return reward


    def render(self):
        print('SB interface renders')
        return np.zeros((10,10,3))


    def close(self):
        print('SB interface closes')

        
if __name__ == '__main__':
    
    run_name = sys.argv[1]
    rng_key = jax.random.PRNGKey(task_randomisation_seed)
    
    # environment verification
    from stable_baselines3.common.env_checker import check_env
    env = Monitor(SB_env_interface(rng_key))
    check_env(env)
    
    # settings
    n_task_instances = 10
    n_training_trials = 5000
    policy_kwargs = dict(net_arch=dict(pi=[8],vf=[8]))
    
    trial_rewards = []
    for i_body in range(n_task_instances):
        print('running task instance', i_body)
        
        # initialise an environment instance
        rng_key, k = jax.random.split(rng_key)
        env = Monitor(SB_env_interface(k))
        
        # model setup
        model = A2C('MlpPolicy', 
                    env, 
                    n_steps=cfg.n_trial_time_steps,
                    use_rms_prop=True,
                    policy_kwargs=policy_kwargs,
                    device='cpu',
                    verbose=1)
        
        # run learning
        model.learn(n_training_trials*cfg.n_trial_time_steps)
        
        trial_rewards_for_task_instance = env.get_episode_rewards()
        trial_rewards.append(trial_rewards_for_task_instance)
        
        print('\nevaluating performance')
        mean_reward = 0
        n_trials = 1
        for i_trial in range(n_trials):
            obs, _ = env.reset(fixed_goal_position=fixed_goal_position)
            for i_step in range(cfg.n_trial_time_steps):
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
            print(f'trial {i_trial} reward {reward}')
            mean_reward += reward/n_trials
        
        print('mean_reward:', mean_reward)
    
    make_dir('runs')
    make_dir('runs/'+run_name)
    np.save('runs/'+run_name+'/evaluation_log',np.array(trial_rewards).T)
    model.save('runs/'+run_name+'/model.zip')
