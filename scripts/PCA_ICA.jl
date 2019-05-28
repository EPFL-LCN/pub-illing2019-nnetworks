####################################################
# Script to train a patchy PCA/ICA network with 1 hidden layer (SGD)
# patch connectivity is controled via "n_patches" & "n_hidden_per_patch" variables

###################################################
# Include libraries and core

using Pkg; Pkg.activate("./../core/"); Pkg.instantiate()
include("./../core/src/ratenets/core.jl")
include("./../core/src/ratenets/helpers_patchy.jl")

###################################################
# Define task and network

# This keyword decides whether PCA or ICA is used for feature extraction:
featurefunction = getPCAscores
# OR: getICAscores

rectifyfeatures = true # Otherwise patchy PCA/ICA is not more powerful than global PCA/ICA (since vanilla PCA/ICA is linear)
dataset = "MNIST" # dataset to be used: "MNIST" or "CIFAR10"
colorimage = dataset == "CIFAR10"
patch_size = 10 # (linear) patch size of receptive fields in first layer p ∈ [1,28] or [1,32] for MNIST/CIFAR10 resp.
n_patches = 500 # number of feature extraction patches
n_hidden_per_patch = 10 # hidden neurons/features per patch. Overall number of hidden neurons is: n_patches * n_hidden_per_patch
n_inits = 3 # number of initializations per net (for averaging)
iterations = 10^6 # iterations per initialization
learningrates = 1e-3*[1; 1] # learningrates per layer
nonlinearity = [relu!] # nonlinearities
nonlinearity_diff = [relu_diff!]

###################################################
# Load and preprocess data

smallimgs, labels, smallimgstest, labelstest = import_data(dataset)
subtractmean!(smallimgs) # substract pixel-wise mean
subtractmean!(smallimgstest)

# Extract PCA/ICA features per patch
smallimgs, smallimgstest = getpatchyhiddenreps(smallimgs, smallimgstest, n_patches, n_hidden_per_patch;
    featurefunction = featurefunction, patchsize = patch_size, colorimage = colorimage)

# Rectify features if applicable
rectifyfeatures && for imgs in [smallimgs,smallimgstest]
     clamp!(imgs,0,Inf64)
end

##############################################################################
# Train network

print(string("train classifier on (", n_patches * n_hidden_per_patch,") unsupervised features (without hidden layer) ...\n"))
for j in 1:n_inits
  net = Network([size(smallimgs)[1], 10])
  learn!(net, getsmallimg, getlabel, iterations, learningrates, 0.,
                     nonlinearity = nonlinearity, nonlinearity_diff = nonlinearity_diff)
  error_train = geterrors(net, smallimgs, labels, nonlinearity = nonlinearity)
  error_test = geterrors(net, smallimgstest, labelstest, nonlinearity = nonlinearity)
  print(string(100 * error_train, " % on training set\n"))
  print(string(100 * error_test, " % on test set\n"))
end
