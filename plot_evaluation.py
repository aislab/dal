import numpy as np
import matplotlib.pyplot as plt
import sys

style = [['black',3],
         ['blue',4],
         ['green',2],
         ['orange',2],
         ['fuchsia',5]]
up_to = [None,51,5001][2]
gridcolour = 0.9
ticklabelsize = 16

plt.figure(figsize=(8.75,4.8))
plt.plot([50,50],[-0.25,1.1],c='grey',ls=':',lw=2)

args = sys.argv[1:]
for i_run in range(len(args)//2):
    path = 'runs/'+args[2*i_run]+'/evaluation_log.npy'
    label = args[2*i_run+1]
    print('run #'+str(i_run))
    print('  path:', path)
    print('  label:', label)
    d = np.load(path)
    print('found reward record with shape:', d.shape, 'for path:', path)
    reward_mean = d.mean(1)
    reward_min = d.min(1)
    reward_max = d.max(1)
    print('mean reward over first trial:', reward_mean[0])

    t = np.arange(reward_mean.shape[0])+1

    i_stype = i_run%len(style)
    plt.fill_between(t,reward_min,reward_max,color=style[i_stype][0],alpha=0.05,zorder=style[i_stype][1],lw=0)
    plt.plot(t,reward_mean,color=style[i_stype][0],alpha=0.4,zorder=style[i_stype][1]+5,label=label)


plt.ylim([-0.25,1.05])
plt.yticks(np.arange(-0.2,1.1,0.1))
plt.tick_params(axis='both', which='major', labelsize=ticklabelsize)
[l.set_visible(False) for (i,l) in enumerate(plt.gca().yaxis.get_ticklabels()) if i % 2]
plt.legend(fontsize='large',loc='lower right').set_zorder(10)
ax = plt.gca()
ax.set_xlabel('Trial',fontsize='x-large',loc='right')
plt.grid(which='both',color=str(gridcolour))
plt.xscale('log')
figpath = 'learning_process.svg'
plt.savefig(figpath,format='svg',bbox_inches='tight')
print('figure saved to:', figpath)
plt.show()
