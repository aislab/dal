import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import importlib

run_name = sys.argv[1]
b = 'runs.'+run_name+'.source.config'
print('importing config from:', b)
cfg = importlib.import_module(b)
print('RL is', ['OFF','ON'][cfg.rl_enabled])
print('NM is', ['OFF','ON'][cfg.nm_enabled])

gridcolour = 0.9
ticklabelsize = 16

if len(sys.argv)>2:
    limit = int(sys.argv[2])
    marker = '_'+sys.argv[2]
else:
    limit = None
    marker = ''

run_path = 'runs/'+run_name
fname = run_path+'/evo_log.csv'
print('loading:', fname)
data = pd.read_csv(fname,sep=',',header=None)
data = pd.DataFrame(data)

print('found data with shape:', data.shape)

t = data[0]

# drop duplicate generations (duplicates occur when trial was resumed from saved generation)
i = 1
while i<data.shape[0]:
    if i and t[i]<t[i-1]:
        drop = range(int(t[i]),int(t[i-1])+1)
        print('data file contains records for overlapping generations:', t[i], '~', t[i-1], '--> dropping older records')
        data.drop(data.index[drop],inplace=True)
        data.reset_index(drop=True, inplace=True)
        i = int(t[i])
        t = data[0]
    else:
        i += 1

if limit is not None and data.shape[0] > limit:
    print('truncating to', limit, 'generations')
    drop = range(limit,data.shape[0])
    data.drop(data.index[drop],inplace=True)
    t = data[0]

ts_best_fitness = data[1]
ts_unmutated_fitness = data[2]
ts_mean_fitness = data[3]
ts_weight_change_nm = data[4]
ts_weight_change_rl = data[5]
ts_rl_learning_rate = data[6]
ts_action_sigma_bias = data[7]
ts_learning_type_ratio = data[8]
ts_best_fitness_nm_only = data[9]
ts_best_fitness_rl_only = data[10]

n_trials = data.shape[1]-11
ts_unmutated_learning_progress = np.array([data[11+i] for i in range(n_trials)]).T

plt.figure(figsize=(6.4,4.8))
plt.ylim([-0.4,1.05])
plt.yticks(np.arange(-0.4,1.01,0.1))
plt.tick_params(axis='both', which='major', labelsize=ticklabelsize)
[l.set_visible(False) for (i,l) in enumerate(plt.gca().yaxis.get_ticklabels()) if i % 2]
plt.grid(which='both',color=str(gridcolour))
plt.xscale('log')
plt.gca().set_xlabel('Generation',fontsize='x-large',loc='right')

pink = np.array([138,43,226])/255
purple = np.array([255,192,203])/255

t += 1
if cfg.nm_enabled and cfg.rl_enabled:
    plt.plot(t,ts_best_fitness,'orange',label='Fitness',zorder=5)
    plt.plot(t,ts_best_fitness_nm_only,'magenta',label='NM-only',zorder=4)
    plt.plot(t,ts_best_fitness_rl_only,'blue',label='RL-only',zorder=3)
elif cfg.nm_enabled:
    plt.plot(t,ts_best_fitness,'magenta',label='NM-only',zorder=3)
elif cfg.rl_enabled:
    plt.plot(t,ts_best_fitness,'blue',label='RL-only',zorder=3)

for i in range(n_trials):
    plt.plot(t,ts_unmutated_learning_progress[:,i],color=[0.5]*3,lw=0.1,zorder=2)

plt.legend(fontsize='x-large',loc='upper left')

figpath = run_path+'/plot_main'+marker+'.svg'
plt.savefig(figpath,format='svg',bbox_inches='tight')
print('saved main plot to:', figpath)
plt.clf()

plt.figure(figsize=(4.8,4.8))
plt.tick_params(axis='both', which='major', labelsize=ticklabelsize)
plt.grid(which='both',color=str(gridcolour))
plt.xscale('log')
plt.gca().set_xlabel('Generation',fontsize='x-large',loc='right')

if cfg.rl_enabled:
    plt.plot(t,ts_weight_change_rl,'blue',label='RL ΔW')
if cfg.nm_enabled:
    plt.plot(t,ts_weight_change_nm,'fuchsia',label='NM ΔW')

plt.legend(fontsize='x-large')
figpath = run_path+'/plot_weight_change'+marker+'.svg'
plt.savefig(figpath,format='svg',bbox_inches='tight')
print('saved weight change plot to:', figpath)
plt.clf()

plt.ylim([-0.2,1.05])
plt.yticks(np.arange(-0.2,1.1,0.1))
plt.tick_params(axis='both', which='major', labelsize=ticklabelsize)
[l.set_visible(False) for (i,l) in enumerate(plt.gca().yaxis.get_ticklabels()) if i % 2]
plt.grid(which='both',color=str(gridcolour))
plt.xscale('log')
plt.gca().set_xlabel('Generation',fontsize='x-large',loc='right')
    
emphasis_interval = 5
for i in range(n_trials):
    r = i/(n_trials-1)
    c=plt.cm.rainbow(r)
    emphasis = i%emphasis_interval==emphasis_interval-1
    style = '-' if emphasis else '.'
    width = 1.0 if emphasis else 0.1
    plt.plot(t,ts_unmutated_learning_progress[:,i],color=c,lw=width)

figpath = run_path+'/plot_learning_progress'+marker+'.svg'
plt.savefig(figpath,format='svg',bbox_inches='tight')
print('saved learning progress plot to:', figpath)
plt.clf()
