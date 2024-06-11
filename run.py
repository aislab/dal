from config import source_copy_list, evaluation_mode
from utils import make_dir
import argparse
import shutil
import importlib
import os
import subprocess
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('from_generation',type=int,default=0,nargs='?')
    parser.add_argument('-u','--update',dest='update_source_files',default=False,action='store_true',
                        help='When resuming an existing run, update the source files (default: False).')
    args = parser.parse_known_args()[0]

    # make a directory for the run
    run_dir = 'runs/'+args.run_name+'/'
    dir_existed = make_dir(run_dir)

    if dir_existed:
        print('\n[NOTE]'+'*'*77)
        if args.update_source_files:
            print("This will overwrite the run's copies of the source files with current source files.")
            print('Press ENTER to continue or CONTROL-C to abort.')
        else:
            print("Only changes in the run's own source file copies will be reflected.")
            print("To update the run to the current source files, use the -u flag.")
            print('Press ENTER to continue or CONTROL-C to abort.')
        print('*'*83)
        input('...')

    # copy all files listed as source files to the run directory.
    if not dir_existed or args.update_source_files:
        source_dir = run_dir+'/source'
        make_dir(source_dir)
        for f in source_copy_list: shutil.copy2(f,source_dir)

    os.chdir('runs/'+args.run_name)
    print('changed working directory to:', os.getcwd())
    
    # run from the copy
    process = subprocess.run(['python','source/evolve.py']+sys.argv[1:])
    
