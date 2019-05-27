####################################################
# Script to train a patchy SC network with 1 hidden layer (SGD)
# patch connectivity is controled via "n_patches" & "n_hidden_per_patch" variables
# Algorithm for sparse coding as proposed in:
#   Brito CSN, GerstnerW(2016) PLoS Comput Biol 12(9).
# OR (depending on "Földiak_model" boolian)
#   Földiak 1990 Biol. Cybernetics

###################################################
# Include libraries and core

using Pkg; Pkg.activate("./../BioPlausibleShallowDeepLearning/"); Pkg.instantiate()
using ProgressMeter, LinearAlgebra, Statistics, Distributions
include("./../src/autoencoders/autoencoders.jl")

###################################################
# Define task and network

dataset = "MNIST" # dataset to be used: "MNIST" or "CIFAR10"
colorimage = dataset == "CIFAR10"
Földiak_model = false

patch_size = 10 # (linear) patch size of receptive field in first layer p ∈ [1,28] or [1,32] for MNIST/CIFAR10 resp.
n_patches = 100 # number of feature extraction patches
n_hidden_per_patch = 10 # hidden neurons/features per patch. Overall number of hidden neurons is: n_patches * n_hidden_per_patch

iterations_SC = 10^5 # iterations for SC learning
n_inits_c = 3 # number of initializations for classifier
iterations_c = 10^6 # iterations for classifier learning
learningrate_c = 1e-3*[1] # learningrate for classifier

###################################################
# Load and preprocess data

smallimgs, labels, smallimgstest, labelstest = import_data(dataset)
subtractmean!(smallimgs) # substract pixel-wise mean
subtractmean!(smallimgstest)

############################################################################
# Set up network for sparse coding layer

layers = [[size(smallimgs)[1],n_hidden_per_patch]]
nAE = size(layers, 1)[1]
learningrates = [1e-3*[1] for i in 1:nAE] # learningrates for SC learning
lambdas = [[1]*1e-2 for i in 1:nAE] # sparsity parameters for SC learning

nonlinearity, nonlinearity_diff = init_nonlin([lin!],[lin_diff!]) # nonlinearities
nonlinearity = [nonlinearity for i in 1:nAE]
nonlinearity_diff = [nonlinearity_diff for i in 1:nAE]
confSAE = configSAE_sparse("_",[size(smallimgs, 2),size(smallimgstest, 2)],
            nAE, layers, 1, convert(Array{Int64, 1}, [iterations_SC for i in 1:nAE]), learningrates, lambdas, 0. ,
            "none", [0. for i in 1:nAE], nonlinearity, nonlinearity_diff, true)

############################################################################
# Set up classifier

hidden_sizes_c = [0] # shallow classifier (no hidden layer)
confClass = configClass("_", hidden_sizes_c,n_inits_c,iterations_c,
        learningrate_c, 0., [relu!], [relu_diff!], true)
confSAEc = configSAEc_sparse(confSAE,confClass)

############################################################################
# Train network

if colorimage
    im_size = Int(sqrt(size(smallimgs)[1]/3))
else
    im_size = Int(sqrt(size(smallimgs)[1]))
end
sae, errors = trainSAEc_full_patchy!(smallimgs, smallimgstest, labels, labelstest, n_hidden_per_patch, confSAEc;
                                        patchsize = patch_size, n_of_populations = n_patches,
                                        stride = 0, im_size = im_size,
                                        Földiak_model = Földiak_model,
                                        colorimage = colorimage)
