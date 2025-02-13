# Learning Mapping between Equilibrium States of Liquid Systems Using Normalizing Flows

Companion repository for the paper "Learning mappings between equilibrium states of liquid systems using Normalizing Flows" aka "Boltzmann 2 Boltzmann"

## Installation
The necessary software for the execution of the notebooks and the instruction for its installation can be found in the folder `conda_envs`.
In particular see the header of the file `spec-file_b2b.txt`.

## Training Data and Flow Parameters
Training Data and parameters for the trained flows can be found on [Zenodo](https://doi.org/10.5281/zenodo.14505665).
Download and extract the two archives into the folder `WCA2LJ`.

## Executing the Notebooks
Without altering the tree of the repository, you can now run the notebooks. Just remember to set the thermodynamic parameters of the system you are interested in to the appropriate ones and the suffix of the variable `run_id` (usually to `_main` for reproducing the runs in the main text of the paper and to `_big_network_SM` or `_long_training_SM` for the ones in the supplementary materials).

Do not hesitate to open issues for bugs or questions.
