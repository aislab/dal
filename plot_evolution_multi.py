import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys

limit = None
ticklabelsize = 16

run_names = sys.argv[1:]
plt.ylim([-0.4,1.05])
plt.yticks(np.arange(-0.4,1.01,0.1))
plt.tick_params(axis='both', which='major', labelsize=ticklabelsize)
[l.set_visible(False) for (i,l) in enumerate(plt.gca().yaxis.get_ticklabels()) if i % 2]
plt.grid(which='both',color='0.9')
plt.xscale('log')
plt.gca().set_xlabel('Generation',fontsize='x-large',loc='right')
plt.legend(fontsize='x-large')

for i_run, run_name in enumerate(run_names):
    fname = 'runs/'+run_name+'/evo_log.csv'
    print('loading:', fname)
    data = pd.read_csv(fname,sep=',',header=None)
    data = pd.DataFrame(data)

    print('found data of shape:', data.shape)

    t = data[0]

    # drop duplicate generations (occur when trial was resumed from saved generation)
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

    print('data length after overlap fix:', len(t))
    
    if limit is not None and data.shape[0] > limit:
        drop = range(limit,data.shape[0])
        data.drop(data.index[drop],inplace=True)
        t = data[0]
    
    t += 1
    
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
    
    print('fitness peak:', ts_best_fitness.max())
    if i_run==0:
        mean_fitness = ts_best_fitness.copy()
        mean_fitness_nm_only = ts_best_fitness_nm_only.copy()
        mean_fitness_rl_only = ts_best_fitness_rl_only.copy()
        min_fitness = ts_best_fitness.copy()
        max_fitness = ts_best_fitness.copy()
        min_fitness_nm_only = ts_best_fitness_nm_only.copy()
        max_fitness_nm_only = ts_best_fitness_nm_only.copy()
        min_fitness_rl_only = ts_best_fitness_rl_only.copy()
        max_fitness_rl_only = ts_best_fitness_rl_only.copy()
    else:
        mean_fitness += ts_best_fitness
        mean_fitness_nm_only += ts_best_fitness_nm_only
        mean_fitness_rl_only += ts_best_fitness_rl_only
        min_fitness = np.minimum(min_fitness,ts_best_fitness)
        max_fitness = np.maximum(max_fitness,ts_best_fitness)
        min_fitness_nm_only = np.minimum(min_fitness_nm_only,ts_best_fitness_nm_only)
        max_fitness_nm_only = np.maximum(max_fitness_nm_only,ts_best_fitness_nm_only)
        min_fitness_rl_only = np.minimum(min_fitness_rl_only,ts_best_fitness_rl_only)
        max_fitness_rl_only = np.maximum(max_fitness_rl_only,ts_best_fitness_rl_only)
    
mean_fitness /= len(run_names)
mean_fitness_nm_only /= len(run_names)
mean_fitness_rl_only /= len(run_names)
    
plt.fill_between(t,min_fitness,max_fitness,color='orange',zorder=4,alpha=0.2,lw=0)
plt.fill_between(t,min_fitness_nm_only,max_fitness_nm_only,color='magenta',zorder=3,alpha=0.2,lw=0)
plt.fill_between(t,min_fitness_rl_only,max_fitness_rl_only,color='blue',zorder=2,alpha=0.2,lw=0)
plt.plot(t,mean_fitness,'orange',label='Fitness',zorder=7)
plt.plot(t,mean_fitness_nm_only,'magenta',label='NM-only',zorder=6)
plt.plot(t,mean_fitness_rl_only,'blue',label='RL-only',zorder=5)

plt.legend(fontsize='x-large')
plt.savefig('plot_evolution_multi.svg',format='svg',bbox_inches='tight')
