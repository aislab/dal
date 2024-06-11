import numpy as np
import jax
import config as cfg
import os
import errno


# create initial RNG key
def init_rng():
    if cfg.rng_seed is None:
        seed = np.random.randint(np.iinfo(np.int64).min,np.iinfo(np.int64).max)
    else:
        seed = cfg.rng_seed
    rng_key = jax.random.PRNGKey(seed)
    print('Initialised RNG with seed:', seed)
    print('Initial RNG key:', rng_key)
    return rng_key


# makes dir if it does not exist.
# if clear is true and dir exists, its content is cleared.
def make_dir(path,clear=False):
    try:
        os.makedirs(path)
        print('created directory:',path)
        return False
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            if clear: clear_dir(path)
            return True
        else:
            raise
        

# Call with i, n to display that i steps out of n steps are completed.
# Call without arguments or with i==n to finalise.
# Optionally supply a message to display next to the progress bar.
def print_progress(i=1,n=1,w=20,message=None):
    r = float(i/n) # convert to float because original values may be jax types that can cause rounding issues below
    p = str(np.round(100*r,1)).rjust(6)
    i = int(np.round(w*r))
    string = '  ['+('▆'*i)+' '*(w-i)+'] '+p+'%'
    if message is not None:
        string += '  '+str(message)
    string += ' '*5
    print(string+'\r',end='\n' if r==1 else '')
