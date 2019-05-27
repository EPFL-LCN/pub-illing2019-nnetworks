###########################################################
This is a (julia) code collection for the publication:

Biologically plausible deep learning - but how far can we go with shallow networks?

by

Bernd Illing, Wulfram Gerstner, Johanni Brea

Contact:
bernd.illing@epfl.ch

###########################################################
Usage:

The main simulations of the publication can be replicated with the scripts collected in “scripts”.
The naming convention is the same as in the paper, e.g. to run the Sparse Coding script cd to the script directory and run:

julia SC.jl

To access or change parameters of the simulations please have a look at the respective script.
Especially for the spiking LIF simulations runtime is a serious problem.
Even though some time was spent on optimising for speed, certain scripts run up to a week (single thread on an Intel Xeon E5-2680 v3 2.5 GHz) for the parameters given in the paper.

The source code for these scripts is in “src”. This code is collected from three different frameworks,

- autoencoders
- lifintegrator
- ratenets

that were developed for different purposes. For that reason different scripts have different syntax.
The julia project environment (BioPlausibleShallowDeepLearning/) should take care of all dependencies.
