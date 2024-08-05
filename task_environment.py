import numpy as np
from dataclasses import dataclass
import jax
from jax import numpy as jp
import config as cfg
from functools import partial

 
action_sampling_shape = ['rectangle','disk','none'][1]
reward_at_final_step_only = True


@dataclass
class State:
    agent_pos: np.ndarray
    goal_pos: np.ndarray
    step: int
    reward: float


@jax.jit
def action_shaping(action):
    if action_sampling_shape == 'rectangle':
        action = jp.clip(action,-1,1)
    elif action_sampling_shape == 'disk':
        action_length = jp.linalg.norm(action,axis=-1,keepdims=True)
        max_magnitude = jp.abs(action).max(-1,keepdims=True)
        max_action_for_direction = jp.where(action_length>0,action/max_magnitude,0)
        max_action_for_direction = jp.linalg.norm(max_action_for_direction,axis=-1,keepdims=True)
        action = action/max_action_for_direction
        action_length = jp.linalg.norm(action,axis=-1,keepdims=True)
        action = jp.where(action_length>1,action/action_length,action)
    return action


@jax.jit
def _step(i_step,agent_pos,goal_pos,action):
    
    action = action_shaping(action)
    agent_pos += action/cfg.n_trial_time_steps
    
    # new distance to goal
    new_dist = jp.linalg.norm(agent_pos-goal_pos,axis=-1)
    is_final_step = (i_step == cfg.n_trial_time_steps-1).astype(int)
    reward = 1-new_dist
    if reward_at_final_step_only:
        reward *= is_final_step
    return agent_pos, reward


@jax.jit
def apply_continuous_task_variation(d_goal,obs_variation_angles):
    
    if cfg.space_dims==2:
        angles = jp.arctan2(d_goal[...,1],d_goal[...,0])+obs_variation_angles[...,0]
        lengths = jp.linalg.norm(d_goal,axis=-1)
        x = lengths*jp.cos(angles)
        y = lengths*jp.sin(angles)
        d_goal = jp.stack((x,y),axis=-1)
    
    if cfg.space_dims==3: # This case could be implemented more efficiently
              
        # rotate around x-axis
        lengths = jp.linalg.norm(d_goal[...,(1,2)],axis=-1)
        x,y,z = d_goal[...,0], d_goal[...,1], d_goal[...,2]
        angles = jp.arctan2(z,y)+obs_variation_angles[...,0]
        y = lengths*jp.cos(angles)
        z = lengths*jp.sin(angles)
        d_goal = jp.stack((x,y,z),axis=-1)

        # rotate around y-axis
        lengths = jp.linalg.norm(d_goal[...,(0,2)],axis=-1)
        x,y,z = d_goal[...,0], d_goal[...,1], d_goal[...,2]
        angles = jp.arctan2(z,x)+obs_variation_angles[...,1]
        x = lengths*jp.cos(angles)
        z = lengths*jp.sin(angles)
        d_goal = jp.stack((x,y,z),axis=-1)
        
        # rotate around z-axis
        lengths = jp.linalg.norm(d_goal[...,(0,1)],axis=-1)
        x,y,z = d_goal[...,0], d_goal[...,1], d_goal[...,2]
        angles = jp.arctan2(y,x)+obs_variation_angles[...,2]
        x = lengths*jp.cos(angles)
        y = lengths*jp.sin(angles)
        d_goal = jp.stack((x,y,z),axis=-1)
        
    return d_goal
    

class Environment:
        
    def set_random_goal(self,rng_key,n_pop):
        rng_key, k = jax.random.split(rng_key)
        # generate coordinates on unit sphere of radius 1
        goal_pos = jax.random.normal(k,[cfg.n_task_instances,cfg.space_dims])
        goal_pos /= jp.linalg.norm(goal_pos,axis=-1,keepdims=True)
        goal_pos = jp.tile(goal_pos,(n_pop,1,1))
        return goal_pos
    
    
    def randomise_task(self,rng_key,n_pop):
        
        rng_key, k = jax.random.split(rng_key)
        
        if cfg.space_dims == 2:
            self.obs_variation_angles = 2*np.pi*jax.random.uniform(k,[cfg.n_task_instances,1])
        
        if cfg.space_dims == 3:
            self.obs_variation_angles = 2*np.pi*jax.random.uniform(k,[cfg.n_task_instances,3])
        
        self.obs_variation_angles = self.obs_variation_angles[None]
    
    
    def reset(self, rng_key: jp.ndarray, n_pop):
        agent_pos = jp.zeros((n_pop,cfg.n_task_instances,cfg.space_dims),dtype=float)
        goal_pos = self.set_random_goal(rng_key,n_pop)
        reward = np.zeros((n_pop,cfg.n_task_instances))
        state = State(agent_pos,goal_pos,0,reward)
        self.trial_fitness = 0
        return state

    
    def step(self, state: State, action: jp.ndarray):
        agent_pos, reward = _step(state.step,state.agent_pos,state.goal_pos,action)
        self.trial_fitness += reward
        state.step = state.step+1
        state = State(agent_pos,state.goal_pos,state.step,reward)
        return state
        
        
    def get_observation(self,state):
        d_goal = np.array(state.goal_pos-state.agent_pos)
        d_goal = apply_continuous_task_variation(d_goal,self.obs_variation_angles)
        return d_goal
    
    
    def get_reward(self,state: State):
        return state.reward
    
    
    def get_goal_position(self,state: State):
        return state.goal_pos
    
    
    # call after trial is completed to get the fitness score for the trial
    def get_trial_fitness(self, state: State):
        return self.trial_fitness

        
