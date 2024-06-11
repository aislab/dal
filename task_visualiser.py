
# Module for rudimentary visualisation of agent behaviour.

import numpy as np
import config as cfg
import imageio

border = 4
canvas_res = (100+2*border,100+2*border)
dotradius = 1
max_tasks_to_show = 4

class Visualiser():

    def __init__(self):
        self.canvas_xy = np.zeros(canvas_res+(3,),dtype=float)
        if cfg.space_dims==3:
            self.canvas_xz = np.zeros(canvas_res+(3,),dtype=float)
        
    def clear_canvas(self):
        self.canvas_xy[:] = 0
        self.canvas_xy[border:-border,border:-border] = 0.5
        if cfg.space_dims==3:
            self.canvas_xz[:] = 0
            self.canvas_xz[border:-border,border:-border] = 0.5
        
    def draw_reward(self,coords,reward):
        for i in range(min(max_tasks_to_show,cfg.n_task_instances)):
            self.draw_at(self.canvas_xy,coords[i,0],coords[i,1],reward[i])
            if cfg.space_dims==3:
                self.draw_at(self.canvas_xz,coords[i,0],coords[i,2],reward[i])
    
    def draw_pos(self,coords):
        for i in range(min(max_tasks_to_show,cfg.n_task_instances)):
            self.draw_at(self.canvas_xy,coords[i,0],coords[i,1],colour=1)
            if cfg.space_dims==3:
                self.draw_at(self.canvas_xz,coords[i,0],coords[i,2],colour=1)
    
    def draw_at(self,canvas,x,y,v=0,colour=None):
        try:
            ix = int(border+(canvas_res[0]/2-border)+(canvas_res[0]/2-border)*0.5*x)
            iy = int(border+(canvas_res[1]/2-border)+(canvas_res[1]/2-border)*0.5*y)
            if colour is None: colour = [np.clip(-v,0,1),np.clip(v,0,1),0]
            canvas[ix-dotradius:ix+dotradius+1,iy-dotradius:iy+dotradius+1] = colour
        except:
            pass
        
    def reset(self,goal_pos,agent_pos):
        self.clear_canvas()
        self.draw_pos(agent_pos[0])
        self.draw_pos(goal_pos[0])
        
    def prepare_image(self):
        im = self.canvas_xy[:,::-1]
        if cfg.space_dims==3:
            im = np.concatenate((im,self.canvas_xz[:,::-1]),axis=0)
        im = (255*im).astype(np.uint8)
        return im
        
