#############################################################
#Basic types and constructors

mutable struct AE # basic autoencoder type (MLP)
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	b::Array{Array{Float64, 1}, 1}
end

function AE(ns::Array{Int64, 1})
	nl = length(ns)
	AE(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[zeros(ns[i])/10 for i in 2:nl])
end

#Autoencoder learnt by sparse coding (and with tied weights)
#Minimizes lasso problem (see, e.g. Mairal and Bach 2009)
mutable struct AE_sparse
	nl::Int64
	x::Array{Array{Float64, 1}, 1}
	ax::Array{Array{Float64, 1}, 1}
	e::Array{Array{Float64, 1}, 1}
	w::Array{Array{Float64, 2}, 1}
	b::Array{Array{Float64, 1}, 1}
	A::Array{Array{Float64, 2}, 1}
	B::Array{Array{Float64, 2}, 1}
end

#init for sparse coding AE (no biases)
function AE_sparse(ns::Array{Int64, 1})
	nl = length(ns)
	AE_sparse(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[5 .* ones(ns[i]) for i in 2:nl],
			[zeros(ns[i], ns[i]) for i in 2:nl],#[rand(ns[i], ns[i])/5 for i in 2:nl],
			[zeros(ns[i], ns[i+1]) for i in 1:nl - 1])
end

mutable struct AE_sparse_patchy
	n_AE_sparse_patches::Int64
	n_out_per_population::Int64
	patchsize::Int64
	stride::Int64
	x::Array{Array{Float64, 1}, 1}
	AE_sparse_patches::Array{AE_sparse, 1}
end

function AE_sparse_patchy(n_in::Int64, n_out_per_population::Int64, n_of_populations::Int64;
					patchsize = 10, stride = 3, im_size = 28, colorimage = true)
	#n_AE_sparse_patches = Int(((im_size - patchsize)/stride + 1)^2)
	if colorimage
		ns = [3 * Int(patchsize^2), n_out_per_population]
	else
		ns = [Int(patchsize^2), n_out_per_population]
	end
	AE_sparse_patchy(n_of_populations,
		n_out_per_population,
		patchsize,
		stride,
		[zeros(n_in),zeros(n_of_populations * n_out_per_population)],
		[AE_sparse(ns) for i in 1:n_of_populations])
end

#same type as AE, different initialization
function Classifier(ns::Array{Int64, 1})
	nl = length(ns)
	AE(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[rand(ns[i])/10 for i in 2:nl])
end

mutable struct SAE_sparse #stacked autoencoder
	nae::Int64
	aes::Array{AE_sparse, 1}
	reps::Array{Array{Array{Float64, 2}, 1}, 1} #AE -> train/test -> 2d array
end
function SAE_sparse(nAE::Int64,layers::Array{Array{Int64, 1}, 1},n_samples::Array{Int64, 1})
	SAE_sparse(nAE,
	[AE_sparse(layers[i]) for i in 1:nAE],
	[[zeros(layers[i][end],n_samples[j])  for j in [1,2]] for i in 1:nAE])
end

mutable struct SAE_sparse_patchy # stacked auto encoder
	nae::Int64
	aes::Array{AE_sparse_patchy, 1}
	reps::Array{Array{Array{Float64, 2}, 1}, 1} #AE -> train/test -> 2d array
end
function SAE_sparse_patchy(n_AE_sparse_patches::Int64,
		n_in::Int64, n_out_per_population::Int64;
		patchsize = 10, stride = 3, im_size = 28,
		n_samples = [60000,10000])
	SAE_sparse_patchy(1,
	[AE_sparse_patchy(n_in, n_out_per_population, n_AE_sparse_patches; patchsize = patchsize, stride = stride, im_size = im_size) for i in 1:nAE],
	[[zeros(n_out_per_population * n_AE_sparse_patches,n_samples[j]) for j in [1,2]]])
end

mutable struct SAEc_sparse #stacked sparse autoencoder with back-end classifier
	sae::SAE_sparse
	classifier::Array{AE, 1}
end


#############################################################
#Configuration Type

mutable struct configClass
	task::String
	hidden_sizes::Array{Int64, 1}
	n_inits::Int64
	iterations::Int64
	learningrates::Array{Float64, 1}
	weight_decay::Float64
	nonlinearity::Array{Function, 1}
	nonlinearity_diff::Array{Function, 1}
	nonlin_diff::Bool
end
mutable struct configAE
	task::String
	hidden_sizes::Array{Int64, 1}
  	n_inits::Int64
	iterations::Int64
	learningrates::Array{Float64, 1}
	weight_decay::Float64
	noise_type::String
	noise::Float64
	nonlinearity::Array{Function, 1}
	nonlinearity_diff::Array{Function, 1}
	nonlin_diff::Bool
end
mutable struct configAE_sparse
	task::String
	hidden_sizes::Array{Int64, 1}
  	n_inits::Int64
	iterations::Int64
	learningrates::Array{Float64, 1}
	lambda::Array{Float64, 1}
	weight_decay::Float64
	noise_type::String
	noise::Float64
	nonlinearity::Array{Function, 1}
	nonlinearity_diff::Array{Function, 1}
	nonlin_diff::Bool
end
mutable struct configSAE_sparse
	task::String
	n_samples::Array{Int64, 1}
	nAE::Int64
	hidden_sizes::Array{Array{Int64, 1}, 1}
  	n_inits::Int64
	iterations::Array{Int64, 1}
	learningrates::Array{Array{Float64, 1}, 1}
	lambdas::Array{Array{Float64, 1}, 1}
	weight_decay::Float64
	noise_type::String
	noise::Array{Float64, 1}
	nonlinearity::Array{Array{Function, 1}, 1}
	nonlinearity_diff::Array{Array{Function, 1}, 1}
	nonlin_diff::Bool
end
mutable struct configSAEc_sparse
	configSAE_sparse::configSAE_sparse
	configClass::configClass
end
function init_nonlin(nonlinarray, nonlindiffarray)
	nl = length(nonlinarray)
	nonlinearity = Array{Function, 1}()
	nonlinearity_diff = Array{Function, 1}()
	for i in 1:nl
		push!(nonlinearity,nonlinarray[i])
		push!(nonlinearity_diff,nonlindiffarray[i])
	end
	return nonlinearity, nonlinearity_diff
end
