mutable struct Network
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	b::Array{Array{Float64, 1}, 1}
end

function Network(ns::Array{Int64, 1})
	nl = length(ns)
	Network(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[rand(ns[i])/10 for i in 2:nl])
end

#network with sparse (hacked) connectivy
mutable struct Network_sparseconnect
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	w_connect::Array{Array{Float64, 2}, 1} #connectivity matrix with only ones or zeros
	b::Array{Array{Float64, 1}, 1}
end
function Network_sparseconnect(ns::Array{Int64, 1})
	nl = length(ns)
	Network_sparseconnect(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[ones(ns[i+1], ns[i]) for i in 1:nl - 1],
			[rand(ns[i])/10 for i in 2:nl])
end

#init for sigmoid nonlinearity for reproduction of Lillicrap paper
function Network_LC(ns::Array{Int64, 1})
	nl = length(ns)
	Network(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[(rand(ns[i+1], ns[i])-0.5)/2. for i in 1:nl - 1],#2
			[(rand(ns[i])-0.5)/10. for i in 2:nl]) #-0.5
end

#init for Oja's rule (no biases)
function Network_Oja(ns::Array{Int64, 1})
	nl = length(ns)
	Network(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[zeros(ns[i]) for i in 2:nl])
end

#  intialize network (wi, bi) with random projections with:
#- learned weights (w) and offsets (b) by backprop for the first weights
#- random weights and offsets in output layer
function Network(ns::Array{Int64, 1},w::Array{Array{Float64, 2}, 1},b::Array{Array{Float64, 1}, 1})
	nl = length(ns)
	wi = [w[i] for i in 1:nl - 2]
	push!(wi,randn(ns[nl], ns[nl-1])/(10*sqrt(ns[nl-1])))
	bi = [b[i] for i in 1:nl - 2]
	push!(bi,rand(ns[nl])/10)
	Network(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			wi,
			bi)
end

#Network initialized with principal components (stored in w and b)
#ONLY ONE HIDDEN LAYER!!!
function Network(ns::Array{Int64, 1},w::Array{Float64, 2},b::Array{Float64, 1})
	nl = length(ns)
	wi = [w]
	push!(wi,randn(ns[nl], ns[nl-1])/(10*sqrt(ns[nl-1])))
	bi = [b]
	#push!(bi,zeros(ns[nl]))
	push!(bi,rand(ns[nl])/10)
	Network(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			wi,
			bi)
end

#Network with random feedback weights and offsets for "backprop"
mutable struct Network_RF
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	w_rf::Array{Array{Float64, 2}, 1}
	b::Array{Array{Float64, 1}, 1}
end

function Network_RF(ns::Array{Int64, 1})
	nl = length(ns)
	Network_RF(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1], #feedback
			[rand(ns[i])/10 for i in 2:nl])
end

mutable struct Network_RF_sparseconnect
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	w_rf::Array{Array{Float64, 2}, 1}
	w_connect::Array{Array{Float64, 2}, 1} #connectivity matrix with only ones or zeros
	b::Array{Array{Float64, 1}, 1}
end
function Network_RF_sparseconnect(ns::Array{Int64, 1})
	nl = length(ns)
	Network_RF_sparseconnect(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1], #feedback
			[ones(ns[i+1], ns[i]) for i in 1:nl - 1], #connectivity
			[rand(ns[i])/10 for i in 2:nl])
end

#network with random feedback for reproduction of Lillicrap paper
function Network_RF_LC(ns::Array{Int64, 1})
	nl = length(ns)
	Network_RF(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[(rand(ns[i+1], ns[i])-0.5)/2 for i in 1:nl - 1],
			[(rand(ns[i+1], ns[i])-0.5)/1 for i in 1:nl - 1], #feedback
			[(rand(ns[i])-0.5)/2 for i in 2:nl])
end

#Network with weights respecting Dale's law
mutable struct Network_Dale
	nl::Int64
	ntypes::Array{Array{Bool, 1}, 1} #determines if neuron is excit. (true) or inhib. (false)
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	b::Array{Array{Float64, 1}, 1}
end

#function for initializating a Network respecting Dale's law
#type_ratio: n_inhibit/n_all for all layers
function Network_Dale(ns::Array{Int64, 1}, inhib_ratio::Array{Float64, 1})
	nl = length(ns)
	#draw inhibitory neurons for each layer
	ntypes = Array{Array{Bool, 1}, 1}()
	wi = Array{Array{Float64, 2}, 1}()
	for i in 1:nl-1
		inhibitory_neurons = [inhib_ratio[i] < rand(1)[1] for j in 1:ns[i]]
		wi_temp = randn(ns[i+1], ns[i])/(10*sqrt(ns[i]))
		for j in 1:ns[i]
			if inhibitory_neurons[j]
				wi_temp[:,j] = abs(wi_temp[:,j]) #excitatory -> all weights positive
			else
				wi_temp[:,j] = - 1. * abs(wi_temp[:,j]) #inhibitory -> all weights negative
			end
		end
		push!(ntypes,inhibitory_neurons)
		push!(wi,wi_temp)
	end
	Network_Dale(nl - 1,
			ntypes,
		  [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			wi,
			[rand(ns[i])/10 for i in 2:nl])
end

#Network with random feedback weights and offsets for "backprop" AND DALE'S LAW!
mutable struct Network_Dale_RF
	nl::Int64
	ntypes::Array{Array{Bool, 1}, 1} #determines if neuron is excit. (true) or inhib. (false)
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	w_rf::Array{Array{Float64, 2}, 1} #feedback
	b::Array{Array{Float64, 1}, 1}
end

function Network_Dale_RF(ns::Array{Int64, 1}, inhib_ratio::Array{Float64, 1})
	nl = length(ns)
	#draw inhibitory neurons for each layer
	ntypes = Array{Array{Bool, 1}, 1}()
	wi = Array{Array{Float64, 2}, 1}()
	for i in 1:nl-1
		inhibitory_neurons = [inhib_ratio[i] < rand(1)[1] for j in 1:ns[i]]
		wi_temp = randn(ns[i+1], ns[i])/(10*sqrt(ns[i]))
		for j in 1:ns[i]
			if inhibitory_neurons[j]
				wi_temp[:,j] = abs(wi_temp[:,j]) #excitatory -> all weights positive
			else
				wi_temp[:,j] = - 1. *abs(wi_temp[:,j]) #inhibitory -> all weights negative
			end
		end
		push!(ntypes,inhibitory_neurons)
		push!(wi,wi_temp)
	end
	Network_Dale_RF(nl - 1,
			ntypes,
		  [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			wi,
			wi, #feedback
			[rand(ns[i])/10 for i in 2:nl])
end


#Network for sparse coding with "activity memory" A,B
mutable struct Network_sparse
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	b::Array{Array{Float64, 1}, 1}
	A::Array{Array{Float64, 2}, 1}
	B::Array{Array{Float64, 2}, 1}
end

#init for sparse coding network (no biases)
function Network_sparse(ns::Array{Int64, 1})
	nl = length(ns)
	Network_sparse(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i], ns[i]) for i in 2:nl],
			[zeros(ns[i], ns[i+1]) for i in 1:nl - 1])
end

mutable struct NetworkWTA
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	ax_tr::Array{Array{Float64, 1}, 1} # low pass filtered internal variable
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	b::Array{Array{Float64, 1}, 1}
end

function NetworkWTA(ns::Array{Int64, 1})
	nl = length(ns)
	NetworkWTA(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[rand(ns[i])/10 for i in 2:nl])
end


mutable struct config
	task::String
	hidden_sizes::Array{Int64, 1}
  n_inits::Int64
	iterations::Int64
	learningrates::Array{Float64, 1}
	weight_decay::Float64
	nonlinearity::String
end

mutable struct config_iter
	task::String
	hidden_sizes::Array{Int64, 1}
  n_inits::Int64
	iterations::Array{Int64, 1}
	learningrates::Array{Float64, 1}
	weight_decay::Float64
	nonlinearity::String
end

mutable struct config_iterations
	task::String
	hidden_sizes::Array{Int64, 1}
  n_inits::Int64
	iterations::Array{Int64, 1}
	n_sampling::Int64
	learningrates::Array{Float64, 1}
	weight_decay::Float64
	nonlinearity::String
end

mutable struct config_sparse
	task::String
	hidden_sizes::Array{Int64, 1}
	n_inits::Int64
	iterations::Int64
	learningrates::Array{Float64, 1}
	lambda::Array{Float64, 1}
	weight_decay::Float64
	nonlinearity::String
end


#############################################################
#Data Import

using Knet
include(Knet.dir("data", "cifar.jl"))
include(Knet.dir("data", "mnist.jl"))
function import_data(dataset::String)
	if dataset == "CIFAR10"
		smallimgs, labels, smallimgstest, labelstest = cifar10()
	    smallimgs = reshape(smallimgs, 32^2 * 3, 50000)
	    smallimgstest = reshape(smallimgstest, 32^2 * 3, 10000)
	elseif dataset == "MNIST"
		smallimgs, labels, smallimgstest, labelstest = mnist()
	    smallimgs = reshape(smallimgs, 28^2, 60000)
	    smallimgstest = reshape(smallimgstest, 28^2, 10000)
	end
	smallimgs, labels .- 1, smallimgstest, labelstest .- 1
end

#############################################################
# data is dxn array of n d-dimensional random vectors
using LinearAlgebra, Distributions
function subtractmean!(data)
        m = mean(data, dims=2)
        d, n = size(data)
        for i in 0:n-1
                BLAS.axpy!(-1., m, 1:d, data, i*d + 1: (i+1) * d)
        end
end
function subtractmean(data)
        m = deepcopy(data)
        subtractmean!(m)
        m
end

# data is dxn array of n d-dimensional random vectors with mean 0
function whiten(data; method = :ZCA)
        f = svd(data)
        eval(Symbol(method, "Whitening"))(f) * sqrt(size(data, 2) - 1)
end

ZCAWhitening(f::LinearAlgebra.SVD) = f.U * f.Vt
PCAWhitening(f::LinearAlgebra.SVD) = f.U


#####################################################
#Helpers

function getsmallimg()
    global patternindex = rand(1:size(smallimgs)[2])
    smallimgs[:, patternindex]
end

function getsample()
    global patternindex = rand(1:n_samples)
    smallimgs[:, patternindex]
end

function getlabel(x)
    [labels[patternindex] == i for i in 0:9]
end

function gethiddenrep()
    global patternindex = rand(1:60000)
    hidden_rep_train[hidden_size_index][:, patternindex]
end

function gethiddensample()
    global patternindex = rand(1:n_samples)
    hidden_rep_train[hidden_size_index][:, patternindex]
end

function geterrors(net, imgs, labels; nonlinearity = nonlinearity, getwrongindices = false)
	error = 0
	noftest = size(imgs)[2]
	if getwrongindices
		wrongindices = []
		for i in 1:noftest
			net.x[1] = imgs[:, i]
			forwardprop!(net, nonlinearity = nonlinearity)
			if findmax(net.x[end])[2] != labels[i] + 1
				error += 1
				push!(wrongindices,i)
			end
		end
		return error/noftest, wrongindices
	else
		for i in 1:noftest
			net.x[1] = imgs[:, i]
			forwardprop!(net, nonlinearity = nonlinearity)
			error += findmax(net.x[end])[2] != labels[i] + 1
		end
		error/noftest
	end
end

# reduce full random connectivity to patches with random connectivity
# by multiplying with mask of linear dimension "patch_size"
# mode:
# - spherical: random vectors for patches lie on the surface of a hypersphere
# - randn: conventional variance conserving initialisation with gaussian random number
function _rescale_weights!(net, i, input_dim, patch_size, mode; colorimage = false)
	if mode == "randn"
		net.w[1][i,:] .*= input_dim./(patch_size) #rescale weights for same input strength as full connectivity
		!colorimage && (net.w[1][i,:] .* 3)
	elseif mode == "spherical"
		net.w[1][i,:] .*= 1. / norm(net.w[1][i,:]) # normalize random patches: since sampled from randn they lie on the surface of a hypersphere now! norm grows as sqrt(eff. in-dim)
		net.w[1][i,:] .*= 3.#sqrt(patch_size) # heuristic rescaling for best performance
	end
end
function set_connectivity!(net,patch_size; mode = "randn", colorimage = false)
	if colorimage
		  input_size = Int(length(net.x[1])/3)
	else
		  input_size = Int(length(net.x[1]))
	end
	input_dim = Int(sqrt(input_size))
	nr_hidden_neurons = length(net.x[2])
	mask = zeros(input_dim,input_dim) #initialize mask
	mask[1:patch_size,1:patch_size] .= 1.
	if colorimage
		for i in 1:nr_hidden_neurons
			shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
			for j in 1:3
			  net.w[1][i,(j-1)*input_size+1:j*input_size] = reshape(reshape(net.w[1][i,(j-1)*input_size+1:j*input_size],input_dim,input_dim).*circshift(mask,shifts),input_size)
			  _rescale_weights!(net, i, input_dim, patch_size, mode; colorimage = colorimage)
			end
		end
	else
		for i in 1:nr_hidden_neurons
			shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
			net.w[1][i,:] = reshape(reshape(net.w[1][i,:],input_dim,input_dim).*circshift(mask,shifts),input_size)
			_rescale_weights!(net, i, input_dim, patch_size, mode; colorimage = colorimage)
		end
	end
end
function set_connectivity_gabor!(net, patchsize; colorimage = false,
      λrange = [patchsize/4,2*patchsize], ψupperbound = 2π,
      σrange = [patchsize/8,patchsize], γrange = 0, amplitude = 1)
    if colorimage
          input_size = Int(length(net.x[1])/3)
    else
          input_size = Int(length(net.x[1]))
    end
	input_dim = Int(sqrt(input_size))
    hiddensize = length(net.x[2])
    if colorimage
        for i in 1:hiddensize
            mask = zeros(input_dim,input_dim)
            mask[1:patchsize,1:patchsize] =
                randMNISTgaborfilterBayesOpt(patchsize, λrange, ψupperbound, σrange, γrange, amplitude)
            shifts = rand(0:input_dim-patchsize,2)
            for j in 1:3
              net.w[1][i,(j-1)*input_size+1:j*input_size] = circshift(mask,shifts)[:]
            end
        end
    else
        for i in 1:hiddensize
            mask = zeros(input_dim,input_dim)
            mask[1:patchsize,1:patchsize] =
                randMNISTgaborfilterBayesOpt(patchsize, λrange, ψupperbound, σrange, γrange, amplitude)#randMNISTgaborfilter(patchsize)#randgaborfilter(patchsize)
            shifts = rand(0:input_dim-patchsize,2)
            net.w[1][i,:] = circshift(mask,shifts)[:]
        end
    end
end
function set_connectivity_matrix!(net,patch_size; mode = "randn", colorimage = false) # for Network_sparse & Network_RF_sparseconnect
	if colorimage
		input_size = Int(length(net.x[1])/3)
	else
		input_size = length(net.x[1])
	end
	input_dim = Int(sqrt(input_size))
	nr_hidden_neurons = length(net.x[2])
	mask = zeros(input_dim,input_dim) #initialize mask
	mask[1:patch_size,1:patch_size] .= 1.
	if colorimage
		for i in 1:nr_hidden_neurons
			shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
			for j in 1:3
				net.w[1][i,(j-1)*input_size+1:j*input_size] = reshape(reshape(net.w[1][i,(j-1)*input_size+1:j*input_size],input_dim,input_dim).*circshift(mask,shifts),input_size)
				net.w_connect[1][i,(j-1)*input_size+1:j*input_size] = clamp.(ceil.(abs.(net.w[1][i,(j-1)*input_size+1:j*input_size])),0,1)
			end
			_rescale_weights!(net, i, input_dim, patch_size, mode; colorimage = colorimage)
		end
	else
		for i in 1:nr_hidden_neurons
			shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
			net.w[1][i,:] = reshape(reshape(net.w[1][i,:],input_dim,input_dim).*circshift(mask,shifts),input_size)
			net.w_connect[1][i,:] = clamp.(ceil.(abs.(net.w[1][i,:])),0,1)
			_rescale_weights!(net, i, input_dim, patch_size, mode; colorimage = colorimage)
		end
	end
end

"""
    gaborfilter
"""
function gaborfilter(λ, θ, ψ, σ, γ, N)
    w = zeros(N, N)
    for x in 1:N
        for y in 1:N
            xᴵ, yᴵ = [cos(θ) sin(θ); -sin(θ) cos(θ)] * ([x; y] .- (1 + (N-1)/2))
            w[x, y] = exp(-(xᴵ^2 + γ^2 * yᴵ^2)/(2σ^2)) * cos(2π * xᴵ/λ + ψ)
        end
    end
    w
end
"""
    randgaborfilter
"""
function randgaborfilter(N)
    w = gaborfilter(2N*rand(), 2π*rand(), 2π*rand(), N*rand(), N*rand(), N)
    w ./ sum(abs.(w))
end
function randMNISTgaborfilter(N)
    w = gaborfilter((2N-N/4)*rand()+N/4, 2π*rand(), 2π*rand(), (N-N/8)*rand()+N/8, 1, N)
    w ./ sum(abs.(w))
end
function randMNISTgaborfilterBayesOpt(N, λrange, ψupperbound, σrange, γrange, amplitude)
    w = gaborfilter(λrange[1] + (λrange[2] - λrange[1]) * rand(),
                    2π*rand(),
                    ψupperbound * rand(),
                    σrange[1] + (σrange[2] - σrange[1]) * rand(),
                    1 - γrange * rand(), N)
    w .* amplitude / sum(abs.(w))
end

# reduce full random connectivity to sparse random connectivity
# by connecting with Bernoulli-sampling
function set_connectivity_sparse!(net,in_degree)
	input_size = length(net.x[1])
	if in_degree != 1.
		nr_hidden_neurons = length(net.x[2])
		for i in 1:nr_hidden_neurons
			no_connection = rand(Bernoulli(in_degree),input_size)
			net.w[1][i,:] .*= no_connection*3/sqrt(in_degree) #rescale weights for same input strength as full connectivity
		end
	end
end

#re_initialize random weights differently for RP
function re_initialize!(net; w_bool = false,b_bool = true)
	if w_bool
		net.w[1] = (rand(size(net.w[1])[1], size(net.w[1])[2])-0.5)/sqrt(size(net.w[1])[2]) #(10*sqrt(size(net.w[1])[2]))
	elseif b_bool
		net.b[1] = zeros(size(net.w[1])[1]) #reference = rand(size(net.w[1])[1])/10
	end
end

#####################################################

#forwardprop for different nonlinearities
function forwardprop!(net; nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		BLAS.gemv!('N', 1., net.w[i], net.x[i], 0., net.ax[i])
		BLAS.axpy!(1., net.b[i], net.ax[i])
		nonlinearity[i](net.ax[i], net.x[i+1])
	end
end

#forwardprop for random feedback weights (functionality = same)
function forwardprop!(net::Network_RF; nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		BLAS.gemv!('N', 1., net.w[i], net.x[i], 0., net.ax[i])
		BLAS.axpy!(1., net.b[i], net.ax[i])
		nonlinearity[i](net.ax[i], net.x[i+1])
	end
end

Z(x) = map(f -> (f(pi*x[1]) + f(pi*(x[1]+x[2])) + 2)/4, [cos; sin])

function heaviside!(x::Array{Float64, 1})
	for i in 1:length(x)
		x[i] = x[i] > 0
	end
end
function heaviside!(inp, outp, threshold = zeros(inp))
	for i in 1:length(inp)
		outp[i] = Int(inp[i] >= threshold[i])
	end
end
function heaviside!(inp, outp; threshold = 0.)
	for i in 1:length(inp)
		outp[i] = Int(inp[i] >= threshold)
	end
end

function relu!(inp, outp)
	for i in 1:length(inp)
		outp[i] = max(0, inp[i])
	end
end

function relu_diff!(ax, error)
	for i in 1:length(ax)
		error[i] *= ax[i] > 0
	end
end

function lin!(inp, outp)
	for i in 1:length(inp)
		outp[i] = inp[i]
	end
end

function lin_diff!(ax, error)
end

function tanh!(x::Array{Float64, 1})
	for i in 1:length(x)
		x[i] = tanh(x[i])
	end
end

function tanh_diff!(ax, error)
	for i in 1:length(ax)
		error[i] *= sech(ax[i])^2
	end
end

function sigm!(inp,outp)
	for i in 1:length(inp)
		outp[i] = 1. / (1. + exp(-inp[i]))
	end
end

function sigm!(x::Array{Float64, 1})
	for i in 1:length(x)
		x[i] = 1. / (1. + exp(-x[i]))
	end
end

function sigm_diff!(ax, error)
	for i in 1:length(ax)
		error[i] *= 1. / (1 + exp(-ax[i])) * (1. - 1. / (1 + exp(-ax[i])))
	end
end

#for different nonlinearities
function backprop!(net::Network, learningrate, weight_decay;
				   nonlinearity_diff = Array{Function, 1})
	for i in net.nl:-1:2
		nonlinearity_diff[i](net.ax[i], net.e[i])
		BLAS.gemv!('T', 1., net.w[i], net.e[i], 0., net.e[i-1])
	end
	nonlinearity_diff[1](net.ax[1], net.e[1])
	for i in 1:net.nl
		BLAS.ger!(learningrate[i], net.e[i], net.x[i], net.w[i])
		BLAS.axpy!(learningrate[i], net.e[i], net.b[i])
		if weight_decay != 0.
			scale!(net.w[i],1. - weight_decay)
			scale!(net.b[i],1. - weight_decay)
		end
	end
end

#for different nonlinearities
function backprop!(net::Network_sparseconnect, learningrate, weight_decay;
				   nonlinearity_diff = Array{Function, 1})
	for i in net.nl:-1:2
		nonlinearity_diff[i](net.ax[i], net.e[i])
		BLAS.gemv!('T', 1., net.w[i], net.e[i], 0., net.e[i-1])
	end
	nonlinearity_diff[1](net.ax[1], net.e[1])
	for i in 1:net.nl
		BLAS.ger!(learningrate[i], net.e[i], net.x[i], net.w[i])
		net.w[i] .*= net.w_connect[i]
		BLAS.axpy!(learningrate[i], net.e[i], net.b[i])
		if weight_decay != 0.
			scale!(net.w[i],1. - weight_decay)
			scale!(net.b[i],1. - weight_decay)
		end
	end
end

#backprop with random feedback WITH derivative of activation function
#nonlin_diff: Use or don't use diff for calculating errors
function backprop!(net::Network_RF, learningrate, weight_decay;
				   nonlinearity_diff = Array{Function, 1},
					 nonlin_diff = Bool(1))
	if nonlin_diff
		for i in net.nl:-1:2
			nonlinearity_diff[i](net.ax[i], net.e[i])
			BLAS.gemv!('T', 1., net.w_rf[i], net.e[i], 0., net.e[i-1])
		end
		nonlinearity_diff[1](net.ax[1], net.e[1])
	else
		for i in net.nl:-1:2
			#nonlinearity_diff(net.ax[i], net.e[i])
			BLAS.gemv!('T', 1., net.w_rf[i], net.e[i], 0., net.e[i-1])
		end
		#nonlinearity_diff(net.ax[1], net.e[1])
	end
	for i in 1:net.nl
		BLAS.ger!(learningrate[i], net.e[i], net.x[i], net.w[i])
		BLAS.axpy!(learningrate[i], net.e[i], net.b[i])
		if weight_decay != 0.
			scale!(net.w[i],1. - weight_decay)
			scale!(net.b[i],1. - weight_decay)
		end
	end
end
function backprop!(net::Network_RF_sparseconnect, learningrate, weight_decay;
				   nonlinearity_diff = Array{Function, 1},
					 nonlin_diff = Bool(1))
	if nonlin_diff
		for i in net.nl:-1:2
			nonlinearity_diff[i](net.ax[i], net.e[i])
			BLAS.gemv!('T', 1., net.w_rf[i], net.e[i], 0., net.e[i-1])
		end
		nonlinearity_diff[1](net.ax[1], net.e[1])
	else
		for i in net.nl:-1:2
			#nonlinearity_diff(net.ax[i], net.e[i])
			BLAS.gemv!('T', 1., net.w_rf[i], net.e[i], 0., net.e[i-1])
		end
		#nonlinearity_diff(net.ax[1], net.e[1])
	end
	for i in 1:net.nl
		BLAS.ger!(learningrate[i], net.e[i], net.x[i], net.w[i])
		net.w[i] .*= net.w_connect[i]
		BLAS.axpy!(learningrate[i], net.e[i], net.b[i])
		if weight_decay != 0.
			scale!(net.w[i],1. - weight_decay)
			scale!(net.b[i],1. - weight_decay)
		end
	end
end

#backprop for random projections and different nonlinearities
function backprop_rp!(net::Network, learningrate;
				   nonlinearity_diff = Array{Function, 1})
	for i in net.nl:-1:net.nl
		nonlinearity_diff[i](net.ax[i], net.e[i])
		BLAS.gemv!('T', 1., net.w[i], net.e[i], 0., net.e[i-1])
	end
	nonlinearity_diff[1](net.ax[1], net.e[1])
	for i in net.nl:net.nl
		BLAS.ger!(learningrate[i], net.e[i], net.x[i], net.w[i])
		BLAS.axpy!(learningrate[i], net.e[i], net.b[i])
	end
end

function getlrates(layerdims)
	1 ./ sqrt(layerdims[1:end-1] + 1)
end

using ProgressMeter

#target must be one-hot
function _seterror_mse!(net, target)
	net.e[end] = target - net.x[end]
end

function _loss_mse(net, target)
	return norm(net.e[end])
end

function _softmax(input)
	input = deepcopy(input)
	exps = exp.(input .- maximum(input))
  return exps / sum(exps)
end

#target must be one-hot
function _seterror_crossentropysoftmax!(net, target)
	probs = _softmax(net.x[end])
	net.e[end] = target - probs
end

function _loss_crossentropy(net, target)
	probs = _softmax(net.x[end])
	return -target'*log.(probs)
end

#learn function for different nonlinearities
function learn!(net::Network,
				inputfunction::Function,
				targetfunction::Function,
				iterations::Int64,
				learningrates::Array{Float64, 1},
				weight_decay::Float64;
				nonlinearity = Array{Function, 1},
				nonlinearity_diff = Array{Function, 1},
				nonlin_diff = true,
				lossfunction = _loss_mse,
				set_error_lossderivative! = _seterror_mse!)
	losses = Float64[]
	loss = 0.
	@showprogress for i in 1:iterations
		net.x[1] = inputfunction()
		forwardprop!(net, nonlinearity = nonlinearity)
		target = targetfunction(net.x[1])
		set_error_lossderivative!(net, target)#net.e[end] = target - net.x[end]
		loss += lossfunction(net, target)#loss += norm(net.e[end])
		if i % div(iterations, 100) == 0
			push!(losses, loss/div(iterations, 100))
			loss = 0.
		end
		backprop!(net, learningrates, weight_decay, nonlinearity_diff = nonlinearity_diff)
	end
	losses
end

function learn!(net::Network_sparseconnect,
				inputfunction::Function,
				targetfunction::Function,
				iterations::Int64,
				learningrates::Array{Float64, 1},
				weight_decay::Float64;
				nonlinearity = Array{Function, 1},
				nonlinearity_diff = Array{Function, 1},
				nonlin_diff = true,
				lossfunction = _loss_mse,
				set_error_lossderivative! = _seterror_mse!)
	losses = Float64[]
	loss = 0.
	@showprogress for i in 1:iterations
		net.x[1] = inputfunction()
		forwardprop!(net, nonlinearity = nonlinearity)
		target = targetfunction(net.x[1])
		set_error_lossderivative!(net, target)#net.e[end] = target - net.x[end]
		loss += lossfunction(net, target)#loss += norm(net.e[end])
		if i % div(iterations, 100) == 0
			push!(losses, loss/div(iterations, 100))
			loss = 0.
		end
		backprop!(net, learningrates, weight_decay, nonlinearity_diff = nonlinearity_diff)
	end
	losses
end

#learn for random projections and different nonlinearities
function learn_rp!(net::Network,
				inputfunction::Function,
				targetfunction::Function,
				iterations::Int64,
				learningrates::Array{Float64, 1},
				weight_decay::Float64;
				nonlinearity = Array{Function, 1},
				nonlinearity_diff = Array{Function, 1},
				nonlin_diff = Bool(1),
				lossfunction = _loss_mse,
				set_error_lossderivative! = _seterror_mse!)
	losses = Float64[]
	loss = 0.
	@showprogress for i in 1:iterations
		net.x[1] = inputfunction()
		forwardprop!(net, nonlinearity = nonlinearity)
		target = targetfunction(net.x[1])
		set_error_lossderivative!(net, target)#net.e[end] = target - net.x[end]
		loss += lossfunction(net, target)#loss += norm(net.e[end]))
		if i % div(iterations, 100) == 0
			push!(losses, loss/div(iterations, 100))
			loss = 0.
		end
		backprop_rp!(net, learningrates, nonlinearity_diff = nonlinearity_diff)
	end
	losses
end

#learn for random feedback weights
function learn!(net::Network_RF,
				inputfunction::Function,
				targetfunction::Function,
				iterations::Int64,
				learningrates::Array{Float64, 1},
				weight_decay::Float64;
				nonlinearity = Array{Function, 1},
				nonlinearity_diff = Array{Function, 1},
				nonlin_diff = Bool(1),
				lossfunction = _loss_mse,
				set_error_lossderivative! = _seterror_mse!)
	losses = Float64[]
	loss = 0.
	@showprogress for i in 1:iterations
		net.x[1] = inputfunction()
		forwardprop!(net, nonlinearity = nonlinearity)
		target = targetfunction(net.x[1])
		set_error_lossderivative!(net, target)#net.e[end] = target - net.x[end]
		loss += lossfunction(net, target)#loss += norm(net.e[end]))
		if i % div(iterations, 10) == 0
			push!(losses, loss/div(iterations, 10))
			loss = 0.
		end
		backprop!(net, learningrates, weight_decay, nonlinearity_diff = nonlinearity_diff,
		 					nonlin_diff = nonlin_diff)
	end
	losses
end

function learn!(net::Network_RF_sparseconnect,
				inputfunction::Function,
				targetfunction::Function,
				iterations::Int64,
				learningrates::Array{Float64, 1},
				weight_decay::Float64;
				nonlinearity = Array{Function, 1},
				nonlinearity_diff = Array{Function, 1},
				nonlin_diff = Bool(1),
				lossfunction = _loss_mse,
				set_error_lossderivative! = _seterror_mse!)
	losses = Float64[]
	loss = 0.
	@showprogress for i in 1:iterations
		net.x[1] = inputfunction()
		forwardprop!(net, nonlinearity = nonlinearity)
		target = targetfunction(net.x[1])
		set_error_lossderivative!(net, target)#net.e[end] = target - net.x[end]
		loss += lossfunction(net, target)#loss += norm(net.e[end]))
		if i % div(iterations, 10) == 0
			push!(losses, loss/div(iterations, 10))
			loss = 0.
		end
		backprop!(net, learningrates, weight_decay, nonlinearity_diff = nonlinearity_diff,
		 					nonlin_diff = nonlin_diff)
	end
	losses
end
