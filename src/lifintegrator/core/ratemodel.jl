"""
  Implements a rate model corresponding to LIF-integrator
"""

struct RateParameters
    tau::Float64
    reset::Float64
    thresholdmean::Float64
    thresholds::Array{Array{Float64, 1}, 1}
    refractory_period::Float64
    spiketrace_timeconstant::Float64
    activation_function::Function
    plasticityrule
    in_amplitude::Float64
    target_amplitude::Float64
    learningrate::Float64
end
export RateParameters
RateParameters(; tau = 10., reset = 0., thresholdmean = 20., thresholds = Array{Array{Float64, 1}, 1}(),
             refractory_period = 2., spiketrace_timeconstant = 20., #refractory_period = 2.
             activation_function = LIFactivationfunction!, plasticityrule = deltarule,
               in_amplitude = 100., target_amplitude = 0.2, learningrate = 1e1) =
RateParameters(tau, reset, thresholdmean, thresholds, refractory_period,
           spiketrace_timeconstant, activation_function, plasticityrule,
           in_amplitude, target_amplitude, learningrate)

mutable struct RateNet
  layer_sizes::Array{Int64, 1} #numnber of neurons per layer
  v::Array{Array{Float64, 1}, 1} #membrane potential/current
  input::Array{Array{Float64, 1}, 1} #external input/bias
  rates::Array{Array{Float64, 1}, 1} #firing rate (nonlinear function of v)
  targets::Array{Float64, 1} #only for last layer
  weights::Array{Array{Float64, 2}, 1}
  parameters::RateParameters
end
export RateNet
function RateNet(layer_sizes, parameters = RateParametes())
  nl = length(layer_sizes)
  RateNet(layer_sizes,
  [zeros(layer_sizes[i]) for i in 1:nl],
  [zeros(layer_sizes[i]) for i in 1:nl],
  [zeros(layer_sizes[i]) for i in 1:nl],
  zeros(layer_sizes[end]),
  [randn(layer_sizes[i+1], layer_sizes[i]) ./ (sqrt(layer_sizes[i])) for i in 1:nl - 1],
  parameters)
end

function Linear!(input, output, parameters, layerindex)
  for i in 1:length(input)
    output[i] = input[i]
  end
end
export Linear!
function ReLU!(input, output, parameters, layerindex)
  for i in 1:length(input)
    output[i] = max(0, input[i] - parameters.thresholds[layerindex][i])
  end
end
export ReLU!
function LIFactivationfunction!(input, output, parameters, layerindex)
  for i in 1:length(input)
    if input[i] <= parameters.thresholds[layerindex][i]
      output[i] = 0.
    else #factor 1000 for unit Hz
      output[i] = 1000. / (parameters.refractory_period - parameters.tau *
                  log(1 - parameters.thresholds[layerindex][i] / input[i]))
    end
  end
end
export LIFactivationfunction!

function forwardprop!(net::RateNet)
  net.v[1] = net.input[1]
  net.parameters.activation_function(net.v[1], net.rates[1], net.parameters, 1)
	for i in 2:length(net.layer_sizes)
		BLAS.gemv!('N', 1., net.weights[i-1], net.rates[i-1], 0., net.v[i])
		BLAS.axpy!(1., net.input[i], net.v[i])
		net.parameters.activation_function(net.v[i], net.rates[i], net.parameters, i)
	end
end

struct SGD
    learningrate::Float64
end
export SGD
struct ADAM
    learningrate::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    βs::Array{Float64, 1}
    m::Array{Float64, 2}
    v::Array{Float64, 2}
end
export ADAM
function ADAM(lr, nin, nout; β1 = .9, β2 = .999, ϵ = 1e-8)
    ADAM(lr, β1, β2, ϵ, [β1; β2], zeros(nout, nin), zeros(nout, nin))
end

function update!(net::RateNet, opt::SGD)
  BLAS.ger!(opt.learningrate, net.targets-net.rates[end], net.rates[end-1], net.weights[end])
end
function update!(net::RateNet, opt::ADAM)
    Δw = (net.targets - net.rates[end]) * net.rates[end-1]'
    @. opt.m = opt.β1 * opt.m + (1 - opt.β1) * Δw
    @. opt.v = opt.β2 * opt.v + (1 - opt.β2) * Δw^2
    @. net.weights[end] += opt.learningrate * opt.m/(1 - opt.βs[1]) /
                            √(opt.v / (1 - opt.βs[2]) + opt.ϵ)
    opt.βs .*= [opt.β1; opt.β2]
end
using ProgressMeter

function learnMNIST!(net::RateNet, opt,	images::Array{Float64, 2}, labels, iterations::Int64)
    for i in 2:length(net.input) net.input[i] = ones(length(net.input[i])) .* net.parameters.thresholdmean end
    	@showprogress for i in 1:iterations
        patternindex = rand(1:size(images)[2])
        net.input[1] = net.parameters.in_amplitude .* images[:, patternindex] .+ net.parameters.thresholdmean
        net.targets = [Int(j == labels[patternindex] + 1) for j in 1:10] .* net.parameters.target_amplitude
    	forwardprop!(net)
    	update!(net, opt)
    end
end
export learnMNIST!

function geterrors(net::RateNet, images::Array{Float64, 2}, labels)
  for i in 2:length(net.input) net.input[i] = ones(length(net.input[i])) .* net.parameters.thresholdmean end
	error = 0
	noftest = size(images)[2]
	for i in 1:noftest
		net.input[1] = net.parameters.in_amplitude .* images[:, i] .+ net.parameters.thresholdmean
		forwardprop!(net)
		error += findmax(net.rates[end])[2] != labels[i] + 1
	end
	error/noftest
end
export geterrors
