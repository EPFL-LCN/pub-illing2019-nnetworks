####################################################
# Script to train a spiking (LIF, STDP), patchy RP/RG network with 1 hidden layer on MNIST.
# Event-based and euler-forward integration are available, the latter being faster
# depending on the network size, firing rates and of course accepted error made by
# the time discretization (Δt keyword in EulerNet function).
# Simulations might take several days (!) for achieving good accuracy (test acc. > 98%).

###################################################
# Include libraries and core

push!(LOAD_PATH, string(pwd(),"./../src/lifintegrator/core/"))
using EventBasedIntegrator, LinearAlgebra, ProgressMeter, Knet

###################################################
# Define task and network

# This keyword decides whether Random Projections ("RP")
# or Random Gabor filters ("RG") are used for feature extraction:
mode = "RG"
# OR "RP"

n_hidden = 50 # number of hidden neurons
patchsize = 10 # (linear) patch size of receptive fields in first layer p ∈ [1,28]
n_iterations = 10^4 # number of iterations (images) for training
learningrate = 2e-4 # learning rate for STDP in second layer
n_eval_trn = n_eval_tst = 10^3 # number of patterns used for evaluation (train & test).
# This is done because evaluating the whole train and test sets might take a long time.
# For the paper the whole data sets were evaluated:
# (n_eval_trn = size(smallimgs)[2], n_eval_tst = size(smallimgstest)[2])
doEuler = true # decides whether Euler forward integration (true) or event based integration is used (false)

###################################################
# Load and preprocess data

include(Knet.dir("data", "mnist.jl"))
function import_mnist()
    smallimgs, labels, smallimgstest, labelstest = mnist()
    smallimgs = reshape(smallimgs, 28^2, 60000)
    smallimgstest = reshape(smallimgstest, 28^2, 10000)
	return smallimgs, labels .- 1, smallimgstest, labelstest .- 1
end

smallimgs, labels, smallimgstest, labelstest = import_mnist()
subtractmean!(smallimgs) # substract pixel-wise mean
subtractmean!(smallimgstest)

###################################################
# Setup network and plasticity

# set additional parameters
n_in = size(smallimgs)[1] # number of input neurons
n_out = 10 # number of output neurons
n_total = n_in + n_hidden + n_out # total number of neurons
threshold = 20. # mV
in_amplitude = 500 # mA
teacher_amplitude = 0.5 # kHz
pat_duration = 5e1 # learning rate should be scaled inversely to this!
transient_duration = 1e2 # ms
pat_duration_test = 2e2 # ms

## Set plasticity rule
plasticityrule = TeacherSignalPlasticity(falses(n_hidden * (n_in + n_out)), zeros(n_total), learningrate)
plasticityrule.isweightplastic[end - n_hidden * n_out + 1:end] .= true

## Setup network
layer_sizes = [n_in,n_hidden,n_out]
network = FeedForwardNet(layer_sizes, LIFParams(threshold = threshold .+ 1. .* rand(n_total));
                        recorder = NoRecorder, plasticityrule = plasticityrule)
if mode == "RP"
    set_connectivity!(network, n_in, n_hidden, patchsize)
elseif mode == "RG"
    set_connectivity_gabor!(network, n_in, n_hidden, patchsize)
else
    error("mode must be either 'RP' or 'RG'")
end
## Set forward euler integration if applicable
doEuler && (network = EulerNet(network; Δt = 5e-2))

###################################################
# Train network

print("Train network... \n")
let endtime = 0
elapsed = @elapsed @showprogress for i in 1:n_iterations
  input, label = getMNISTinput(smallimgs, labels, threshold, n_hidden, n_out;
    in_amplitude = in_amplitude)
  plasticityrule.islearning = false # learning off during transient
  endtime += transient_duration
  integratenet!(network, input, endtime) # integrate transient

  plasticityrule.islearning = true # enable learning and define target
  plasticityrule.targets[end-n_out+1:end] = [Int(i == label+1) for i in 1:n_out]*teacher_amplitude
  endtime += pat_duration
  integratenet!(network, input, endtime) # integrate learning period
end
end

###################################################
# Get errors

print("Test phase... \n")
error_train, endtime = geterror!(network, smallimgs, labels, threshold,
                    n_hidden, n_out, in_amplitude, n_eval_trn, pat_duration_test)
error_test, endtime = geterror!(network, smallimgstest, labelstest, threshold,
                    n_hidden,n_out, in_amplitude, n_eval_tst, pat_duration_test)
print("\n Train error:")
print(error_train)
print("\n Test error:")
print(error_test)
print("\n")
