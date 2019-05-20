
#####################################################
#Weight update functions

function backprop!(net::AE, learningrate, weight_decay;
				   nonlinearity_diff = Array{Function, 1},
					 BPuptolayer = net.nl)
	for i in net.nl:-1:2
		nonlinearity_diff[i](net.ax[i], net.e[i])
		BLAS.gemv!('T', 1., net.w[i], net.e[i], 0., net.e[i-1])
	end
	nonlinearity_diff[1](net.ax[1], net.e[1])
	for i in 1:BPuptolayer
		BLAS.ger!(learningrate[i], net.e[i], net.x[i], net.w[i])
		BLAS.axpy!(learningrate[i], net.e[i], net.b[i])
		if weight_decay != 0.
			scale!(net.w[i],1. - weight_decay)
			scale!(net.b[i],1. - weight_decay)
		end
	end
end

function _normalize_inputweights!(weights)
	for j in 1:size(weights)[1]
		weights[j,:] .*= 1. / norm(weights[j,:])
	end
end

#Algorithm for sparse coding as proposed in:
#Brito CSN, GerstnerW(2016) PLoS Comput Biol 12(9).
#Implements the second part of algorithm: weight update
#omitnonlocal: boolian if non local term should be omitted:
#true: will be omitted
#false: will be taken into account
function _backprop_sparse_Brito!(net::AE_sparse, learningrate; omitnonlocal = true, patches = false, w_connect = [])
	if net.nl > 1
		error("Network/AE is deeper than expected for a sparse network (more than one hidden layer)!\n")
	end
	if omitnonlocal
		for i in 1:net.nl
			#BLAS.ger!(learningrate[i],net.ax[i],net.x[i],net.w[i])
			#Above: might be useful for ordinary sparse coding à la mairal which uses .ax and .x differently!
			BLAS.ger!(learningrate[i],net.x[i+1],net.x[i],net.w[i])
			#BLAS.ger!(learningrate[i],net.x[i+1].*(net.x[i+1]-0.1),net.x[i],net.w[i]) #quadratic Hebbian
			_normalize_inputweights!(net.w[i])
	  end
	else
		for i in 1:net.nl
			#BLAS.ger!(learningrate[i],net.ax[i],(net.x[i]-BLAS.gemv('T',net.w[i],net.ax[i])),net.w[i])
			BLAS.ger!(learningrate[i],net.x[i+1],(net.x[i]-BLAS.gemv('T',net.w[i],net.x[i+1])),net.w[i])
			_normalize_inputweights!(net.w[i])
		end
	end
	if patches
		net.w[1] .*= w_connect
	end
end
function _backprop_sparse_Földiak!(net::AE_sparse;
		p = 0.1, #2. / length(net.ax[1]), #0.4
		learningrate_v = 1e-2, learningrate_w = 1e-3, learningrate_thr = 1e-2,
		patches = false, w_connect = [])

	#BLAS.ger!(learningrate_v, Float64.(net.x[2] .> net.b[1]), Float64.(net.x[2] .> net.b[1]), net.A[1])
	BLAS.ger!(learningrate_v,net.x[2],net.x[2],net.A[1])
	net.A[1] .+= - learningrate_v * p^2
	for j in 1:size(net.A[1])[1]
		net.A[1][j,j] = 0. #no self-inhibition
	end
	clamp!(net.A[1],0.,Inf64) #Dale's law
	net.w[1] = Diagonal(1 .- learningrate_w * net.x[2]) * net.w[1]
	BLAS.ger!(learningrate_w,net.x[2],net.x[1],net.w[1])
	#BLAS.axpy!(learningrate_thr,Float64.(net.x[2] .> net.b[1]) .- p, net.b[1])
	BLAS.axpy!(learningrate_thr,net.x[2] .- p, net.b[1]) # ...,Int.(net.x[2] .> net.b[1]) .- p,...
	if patches
		net.w[1] .*= w_connect
	end
end
function backprop_sparse_Brito!(net::AE_sparse, learningrate; omitnonlocal = true, patches = false, w_connect = [], Földiak_model = false)
	if Földiak_model
		_backprop_sparse_Földiak!(net; patches = patches)
	else
		_backprop_sparse_Brito!(net, learningrate; omitnonlocal = omitnonlocal, patches = patches, w_connect = w_connect)
	end
end

function backprop_sparse_Brito!(net::AE_sparse_patchy, learningrate; omitnonlocal = true, Földiak_model = false)
	for AE_sparse_patch in net.AE_sparse_patches
		backprop_sparse_Brito!(AE_sparse_patch, learningrate;
			omitnonlocal = omitnonlocal, patches = false, Földiak_model = Földiak_model)
	end
end
