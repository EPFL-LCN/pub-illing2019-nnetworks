###########################################################
This is a (julia) code collection for the publication:

Biologically plausible deep learning - but how far can we go with shallow networks?

by

Bernd Illing, Wulfram Gerstner, Johanni Brea

Contact:
bernd.illing@epfl.ch

###########################################################
Usage:

The main simulations of the publication can be replicated with the scripts collected in "scripts".
The naming convention is the same as in the paper, e.g. to run the Sparse Coding script cd to the "scripts" directory and run:

julia SC.jl

(Note: at first execution all dependencies will be installed and the dataset(s) will be downloaded. This could take a few minutes.)
To access or change parameters of the simulations please have a look at the respective script.
Especially for the spiking LIF simulations runtime can be very long.
Even though quite some effort was spent on optimising for speed, certain scripts run up to a week (single thread on an Intel Xeon E5-2680 v3 2.5 GHz) for the parameters given in the paper.

You can find the core source code used by the above mentioned scripts in "src". This code is collected from three different frameworks,

- autoencoders
- lifintegrator
- ratenets

that were developed for different purposes. For that reason different scripts have different syntax.
The julia project environment (BioPlausibleShallowDeepLearning/) should take care of all dependencies.
