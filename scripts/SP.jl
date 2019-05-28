###################################################
# Script to train a simple perceptron

###################################################
# Include libraries and core

using Pkg; Pkg.activate("./../core/"); Pkg.instantiate()
include("./../core/src/ratenets/core.jl")

###################################################
# Define task and network

dataset = "MNIST" # dataset to be used: "MNIST" or "CIFAR10"
hidden_sizes = [0] # number of neurons in hidden layer (0 for simple perceptron)
n_inits = 3 # number of initializations per net (for averaging)
iterations = 10^6 # iterations per initialization
learningrates = [1e-4] # learningrates per layer
nonlinearity = [relu!] #nonlinearities
nonlinearity_diff = [relu_diff!]

###################################################
# Load and preprocess data

smallimgs, labels, smallimgstest, labelstest = import_data(dataset)
subtractmean!(smallimgs) # substract pixel-wise mean
subtractmean!(smallimgstest)

###################################################
# Train network

print("train simple perceptron (without hidden layer)...\n")
for j in 1:n_inits
  net = Network([size(smallimgs)[1], 10])
  learn!(net, getsmallimg, getlabel, iterations, learningrates, 0.,
                     nonlinearity = nonlinearity, nonlinearity_diff = nonlinearity_diff)
  error_train = geterrors(net, smallimgs, labels, nonlinearity = nonlinearity)
  error_test = geterrors(net, smallimgstest, labelstest, nonlinearity = nonlinearity)
  print(string(100 * error_train, " % on training set\n"))
  print(string(100 * error_test, " % on test set\n"))
end
