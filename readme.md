This is a (julia) code collection for the publication:

B. Illing, W. Gerstner & J. Brea,
[*Biologically plausible deep learning - but how far can we go with shallow networks?*](https://arxiv.org/abs/1905.04101), arXiv:1905.04101, Feb. 2019

(to appear in [Neural Networks](https://www.journals.elsevier.com/neural-networks))

Contact:
[bernd.illing@epfl.ch](mailto:bernd.illing@epfl.ch)

## Installation

Dependencies:

* Mac or Linux
* [Julia](https://julialang.org) (1.1 or higher)

All other (julia) dependencies and packages will be installed or updated automatically.

## Usage

The main simulations of the publication can be replicated with the scripts collected in "scripts".
The naming convention is the same as in the paper. E.g. to run the Sparse Coding simulation, cd to the "scripts" directory and run:

julia SC.jl

(Note: at first execution all dependencies will be installed and the dataset(s) will be downloaded which could take a few minutes)

To access or change parameters of the simulations please have a look at the respective script.
Parameters in the scripts are *not* the same as in the publication (because simulations would take very long), however, the actual parameters are given in the appendix of the paper.
Especially for the spiking LIF simulations runtime can be very long.
Even though quite some effort was spent on optimising for speed, certain scripts run up to a week (single thread on an Intel Xeon E5-2680 v3 2.5 GHz) for the parameters given in the paper.

## Core source code

You can find the core source code used by the above mentioned scripts in "core". This code is collected from three different frameworks,

* autoencoders
* lifintegrator
* ratenets

that were developed for different purposes. For that reason different the scripts have different syntax.
The julia project environment (core/, activated automatically in every script) should take care of all dependencies.
