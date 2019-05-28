
#####################################################
#Learning functions

using ProgressMeter

#learning classifier
function learn!(net::AE,
				sae,
				inputfunction::Function,
				targetfunction::Function,
				iterations::Int64,
				learningrates::Array{Float64, 1},
				weight_decay::Float64;
				nonlinearity = Array{Function, 1},
				nonlinearity_diff = Array{Function, 1},
				nonlin_diff = true,
				BPuptolayer = net.nl)
	losses = Float64[]
	loss = 0.
	@showprogress for i in 1:iterations
		net.x[1] = inputfunction(sae)
		forwardprop!(net, nonlinearity = nonlinearity)
		target = targetfunction(net.x[1])
		net.e[end] = target - net.x[end]
		loss += norm(net.e[end])
		if i % div(iterations, 10) == 0
			push!(losses, loss/div(iterations, 10))
			loss = 0.
		end
		backprop!(net, learningrates, weight_decay, nonlinearity_diff = nonlinearity_diff, BPuptolayer = BPuptolayer)
	end
	losses
end

# sparse patchy AE
function learnAE_Brito_patchy!(net::AE_sparse_patchy,
					inputfunction::Function,
					iterations::Int64,
					learningrates::Array{Float64, 1},
					weight_decay::Float64,
					noise_type::String,
					noise::Float64,
					lambda::Array{Float64, 1};
					nonlinearity = Array{Function, 1},
					nonlinearity_diff = Array{Function, 1},
					nonlin_diff = true,
					BPuptolayer = net.nl,
					patches = false,
					w_connect = [],
					Földiak_model = false,
					rand_patch_pos = false,
					colorimage = false)

	@showprogress for i in 1:iterations
		sample = inputfunction()
		net.x[1] = deepcopy(sample)
		if noise_type != "none"
			AddNoise!(net, noise)
		end
		forwardprop_Brito_full!(net, lambda, i; nonlinearity = nonlinearity,
			Földiak_model = Földiak_model, rand_patch_pos = rand_patch_pos,
			colorimage = colorimage)
		backprop_sparse_Brito!(net,learningrates; Földiak_model = Földiak_model)
	end
end
