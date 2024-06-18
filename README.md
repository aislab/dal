
## Introduction

This repository contains the source code for the experiments described in the following paper:\
__Breaching the Bottleneck: Evolutionary Transition from Reward-Driven Learning to Reward-Agnostic Domain-Adapted Learning in Neuromodulated Neural Nets__

### Dependencies
```
jax (0.4.25)
optax (0.1.7)
numpy (1.24.3)
imageio (2.31.1)
matplotlib (3.7.1)
pandas (2.1.2)
```
Numbers in brackets are versions we tested with.\
To run comparative evaluation with the Stable Baselines 3 implementation of A2C, you will additionally need Pytorch, gymnasium, and Stable Baselines 3.

### Notes

- The implementation is designed to run on CPUs, to facilitate parallel evaluation of populations of neural networks of heterogeneous architecture. Each individual runs as a separate subprocess, so the system runs best on machines with many CPU cores.
- If you get GPU OOM errors, chances are the subprocesses are all trying to claim GPU memory. An easy solution is to make the GPU(s) invisible using e.g. the CUDA_VISIBLE_DEVICES environment variable.
- When running on a screenless machine you may need a virtual framebuffer. We used xvfb for this as follows: `xvfb-run -a python run.py RUN_NAME [FROM_GENERATION] [-u] [-e]`.

## Evolution

Run:
```
python run.py RUN_NAME [FROM_GENERATION] [-u]
```

Where:\
`RUN_NAME`: A freely chosen name for the run, or the name of an existing run you wish to continue.\
`FROM_GENERATION` (optional): The generation to resume from, when resuming an existing run. For this to work, a file containing the relevant generation must exist in the `runs/RUN_NAME/` directory.\
`-u` (optional): If the `-u` flag is set, the source code in the `runs/RUN_NAME/source/` directory will be updated (explained below).

When starting a new run named `RUN_NAME`, the following happens:
- A directory `runs/RUN_NAME` is created.
- The source files in the base directory are copied to the `runs/RUN_NAME/source/` directory.
- The current working directory is changed to `runs/RUN_NAME`.
- `runs/RUN_NAME/source/evolve.py` is launched.

This means that what is actually running now is not the code in the base directory, but the copy thereof in the run's own source directory.

When resuming an existing run, you should indicate whether or not you want to copy the source files again or not. If the `-u` flag ("update") is set, the source files in the base directory will be copied to the `runs/RUN_NAME/source/` directory again.

A few implications:
-   Modifying source in the base directory does not affect a running run.
-   Modifying source in the `runs/RUN_NAME/source/` directory CAN affect an ongoing run. Due to subprocesses being launched repeatedly during evolution, you can break a running run by modifying its local source files (`config.py` and `nmnn.py` in particular).
-   If you modify source in the base directory and want to resume an existing run using the newly modified code, you must resume WITH the `-u` flag to update the run's copy of the source code.
-   If you have modified source in the base directory and want to resume an existing run using the original unmodified code that the run was launched with, you must resume WITHOUT the `-u` flag.
-   If you want to make changes specifically to the source code of an existing run, stop the run, modify the code in the `runs/RUN_NAME/source/` directory, and then resume the run WITHOUT the `-u` flag.

Evolutionary progress is plotted in `runs/RUN_NAME/plot_FROM_GENERATION.png` (updated every generation).
These plots contain a lot of information and can be hard to read.
More legible plots can be produced after (or during) evolution following the instructions below.

A rudimentary visualisation of agent behaviour is saved to the same directory, at 10-generation intervals.
These images show the focal individual's final position for a few task instances, at 10-trial intervals.
The central white dot is the origin, surrounding white dots are goal positions, and coloured dots are end-of-trial agent positions, with colour indicating reward received.

Numerous system settings can be tweaked in `config.py`.\
To replicate the RL-only or NM-only baseline experiments, set `nm_enabled` or `rl_enabled` to `False`.\
To replicate the "RL with bottlenecked NM" baseline, set `restrict_modulating_neurons_to_value_output` to `True`.

### Notes:
- If terminal output looks garbled, you may need to widen the terminal window.
- To see more detailed information in the terminal during evolution, disable `suppress_nn_prints` in `config.py`. This allows NNs to print their own information to the terminal (asynchronously).

## Plotting evolution

Plot for single run (fig 4b,e,f in the paper):
```
python plot_evolution RUN_NAME [LIMIT]
```

Where:\
`RUN_NAME`: Name of an existing run.\
`LIMIT` (optional): maximum number of generations to plot.

Plots are saved to `runs/RUN_NAME`.

---
Plot over multiple runs (fig 4a in the paper):
```
python plot_evolution_multi.py RUN_NAME1 [RUN_NAME2] [RUN_NAME3] ...
```

Where:\
`RUN_NAMEx`: Name of an existing run.

The plot is saved to the base directory.

Notes:
- If your runs have different lengths, set the limit variable at the top of `plot_evolution_multi.py` to specify the number of generations to plot.
- Plotting over multiple runs assumes that all runs have RL and NM both enabled.

## Evaluation

You can evaluate the focal individual from a specific generation in an existing run using the `-e` flag.\
Run:
```
python run.py RUN_NAME [FROM_GENERATION] -e [-u]
```

Where:\
`RUN_NAME`: Name of the run to evaluate.\
`FROM_GENERATION` (optional): The generation to load. The focal individual from this generation will be evaluated.\
`-e`: Activates evaluation mode.\
`-u` (optional): Updates the run's copy of the source code (see above).

To evaluate an unevolved RL agent, use a new `RUN_NAME` and omit `FROM_GENERATION`. This will produce a freshly initialised RL agent and evaluate it. The config file contains different initialisation settings for NNs that are initialised in evaluation mode. These settings correspond to the A2C defaults from Stable Baselines 3.

Evaluation mode produces the files `evaluation_log.csv` and `evaluation_log.npy` in the `runs/RUN_NAME/` directory. See __plotting evaluations__ below to plot their content.

---
We compared our A2C implementation to the Stable Baselines 3 implementation to ensure that ours is on par.\
To replicate this evaluation, run:
```
python SB_task_interface RUN_NAME
```

Where:\
`RUN_NAME`: an unused run name

This produces an `evaluation_log.npy` file as above.

## Plotting evaluations
Evaluation results can be plotted as follows (fig 4g in the paper).\
Run:
```
python plot_evaluation.py RUN_NAME1 LABEL1 [RUN_NAME2] [LABEL2] [RUN_NAME3] [LABEL3] ...
```

Where:\
    `RUN_NAMEx`: Name of a run for which evaluation_log.npy exists.\
    `LABELx`: Plotting label for run RUN_NAMEx.

---
First-trial agent trajectories can be generated as follows (fig 5 in the paper).\
Run:
```
python analyse_learning_process.py RUN_NAME
```

Where:\
`RUN_NAME`: Name of an existing run.

Figures of the agent trajectories are stored in `runs/RUN_NAME/`.\
The list of generations to analyse can be found near the top of `analyse_learning_process.py`.
The test set of target positions is fixed across the generations, and determined by the seed in `config.py`.

---

## Citation
Breaching the Bottleneck: Evolutionary Transition from Reward-Driven Learning to Reward-Agnostic Domain-Adapted Learning in Neuromodulated Neural Nets\
Solvi Arnold, Reiji Suzuki, Takaya Arita, Kimitoshi Yamazaki\
ALIFE 2024 (The 2024 Conference on Artificial Life)

