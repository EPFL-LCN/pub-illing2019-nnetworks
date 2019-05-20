
#####################################################
#forwardprop for different nonlinearities

function forwardprop!(net; nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		BLAS.gemv!('N', 1., net.w[i], net.x[i], 0., net.ax[i])
		BLAS.axpy!(1., net.b[i], net.ax[i])
		nonlinearity[i](net.ax[i], net.x[i+1])
	end
end

function forwardprop_no_rec_linearpart!(net::AE_sparse, i::Int64)
	BLAS.gemv!('N', 1., net.w[i], net.x[i], 0., net.ax[i])
	BLAS.axpy!(1., net.b[i], net.ax[i])
end
function forwardprop_no_rec!(net::AE_sparse, lambdas::Array{Float64, 1}; nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		forwardprop_no_rec_linearpart!(net, i)
		nonlinearity[i](net.ax[i],net.x[i+1],lambdas[i])
		#nonlinearity[i](net.ax[i],net.x[i+1],0)
	end
end
function forwardprop_no_rec!(net::AE_sparse; nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		forwardprop_no_rec_linearpart!(net, i)
		nonlinearity[i](net.ax[i],net.x[i+1],net.b[i])
		#nonlinearity[i](net.ax[i],net.x[i+1],zeros(length(net.b[i]))))
	end
end

#see "g"-function in PhD-Thesis of C. Brito or Rozell2008
function activation_Brito!(input,output,lambda)
	for i in 1:length(input)
		output[i] = clamp(input[i]-lambda,0.,Inf64) #thresholded, linear rectifier
	end
end

#Implementation of 1st part of sparse coding algorithm by C. Brito
#Uses non-local inhibitory connections w^T*w
#epsilon: stop condition for sparsity iteration:
#ratio of update magnitude and quantity magnitude
#r: regression, i.e. implementation of smooth integrator with same fixed point with time constant 1/r
#-> iteration should go on at least 1/r for convergence
function forwardprop_Brito!(net::AE_sparse, lambdas::Array{Float64, 1}; epsilon = 1., r = .1, nonlinearity = Array{Function, 1})
	for i in 1:net.nl
		if nonlinearity[i] != lin!
			error("nonlinearity != linear. Must be linear for sparse coding!")
		else
			#while eps > epsilon
			net.x[i+1][:] = 0.
			net.ax[i][:] = 0.
			n,m = size(net.w[i])
			for k in 1:20
				BLAS.gemv!('N',r,net.w[i],net.x[i]-BLAS.gemv('T',net.w[i],net.x[i+1]),1.0-r,net.ax[i])
				activation_Brito!(net.ax[i],net.x[i+1],lambdas[i])
			end
		end
	end
end

#full implementation with plastic lateral inhibitory connections learned by anti-Hebbian rule
#see Brito's Thesis Chapter 2
#abusing a bit the type AE_sparse by
# - saving/integrating past activations of hidden layer in B-array
# - using A-array for lateral inhibitory connections
function _forwardprop_Brito_full!(net::AE_sparse, lambdas::Array{Float64, 1}, iter::Int64;
										nonlinearity = Array{Function, 1}, iterations = 50, r = 1e-1, memory_decay = 1e-2,
										learningrate_inh = 1e-2, no_recurrence = false)
										#This had worked for sure: iterations = 50, r = 1e-1, memory_decay = 1e-2,
										# #with learningrate 1e-2 and learningrate_inh = 1e-2
	if no_recurrence
		forwardprop_no_rec!(net, lambdas; nonlinearity = [activation_Brito!])
	else
		if iter == -1
			one_over_iter = 0
		else
			one_over_iter = 1. / convert(Float64, iter)
		end
		for i in 1:net.nl
			if nonlinearity[i] != lin!
				error("nonlinearity != linear. Must be linear for sparse coding!")
			else
				net.x[i+1][:] .= 0.
				net.ax[i][:] .= 0.
				n,m = size(net.w[i])
				for j in 1:n
					net.A[i][j,j] = 0. #no self-inhibition
				end
				input = BLAS.gemv('N',net.w[i],net.x[i])
				for k in 1:iterations
					net.ax[i] = r*(input-BLAS.gemv('N',net.A[i],net.x[i+1]))+(1.0-r)*net.ax[i]
					activation_Brito!(net.ax[i],net.x[i+1],lambdas[i])
				end
				if iter != -1 #only learn lateral inhibition weights during overall learning procedure, not during generatehiddenreps!
					if one_over_iter > memory_decay #avoids zero-filling at beginning
						net.B[i][1,:] = (1. - one_over_iter)*net.B[i][1,:] + one_over_iter*net.x[i+1]
					else
						net.B[i][1,:] = (1. - memory_decay)*net.B[i][1,:] + memory_decay*net.x[i+1]
					end
					BLAS.ger!(learningrate_inh,net.x[i+1]-net.B[i][1,:],net.x[i+1],net.A[i])
					clamp!(net.A[i],0.,Inf64) #Dale's law
				end
			end
		end
	end
end
function _forwardprop_Földiak!(net::AE_sparse; no_recurrence = false,
								nonlinearity = relu!, dt = 1e-1, epsilon = 1e-4)
	if no_recurrence
		forwardprop_no_rec!(net; nonlinearity = [nonlinearity])
	else
		net.ax[1] .= 0. # u
		net.x[2] .= 0. # a
		scaling_factor = epsilon/dt
		voltage_incr = scaling_factor*norm(net.ax[1])+1 #+1 to make sure loop is entered
		input_without_recurrence = BLAS.gemv('N',net.w[1],net.x[1])
		while norm(voltage_incr) > scaling_factor*norm(net.ax[1])
			voltage_incr = input_without_recurrence - BLAS.gemv('N',net.A[1],net.x[2]) - net.ax[1]
			BLAS.axpy!(dt, voltage_incr, net.ax[1]) # update membrane potential
			nonlinearity(net.ax[1],net.x[2],net.b[1]) # apply activation function
		end
	end
end
function forwardprop_Brito_full!(net::AE_sparse, lambdas::Array{Float64, 1}, iter::Int64;
									nonlinearity = Array{Function, 1}, no_recurrence = false, Földiak_model = false)
	if Földiak_model
		_forwardprop_Földiak!(net; no_recurrence = no_recurrence)
	else
		_forwardprop_Brito_full!(net, lambdas, iter; nonlinearity = nonlinearity, no_recurrence = no_recurrence)
	end
end

#same for generatehiddenreps (not needed to pay attention on zeros in activities at beginning of training:
#iteration number isn't important)
function forwardprop_Brito_full!(net::AE_sparse, lambdas::Array{Float64, 1};
									nonlinearity = Array{Function, 1}, no_recurrence = false, Földiak_model = false)
	if Földiak_model
		_forwardprop_Földiak!(net; no_recurrence = no_recurrence)
	else
		iter = -1 #codes for iteration number isn't important
		_forwardprop_Brito_full!(net, lambdas, iter; nonlinearity = nonlinearity, no_recurrence = no_recurrence)
	end
end


function getrandompatchpositions(net::AE_sparse_patchy; colorimage = false)
	if colorimage
		im_size = Int(sqrt(length(net.x[1])/3))
	else
		im_size = Int(sqrt(length(net.x[1])))
	end
	nh = length(net.x[2])
	return rand(1:im_size - net.patchsize + 1, nh, 2)
end

function distributeinput!(net::AE_sparse_patchy; rand_patch_pos = false,
	colorimage = false)
	if colorimage
		im_size = Int(sqrt(length(net.x[1])/3))
	else
		im_size = Int(sqrt(length(net.x[1])))
	end
	if rand_patch_pos #patch_positions is globally defined in training.jl
		for i in 1:net.n_AE_sparse_patches
			if colorimage
				net.AE_sparse_patches[i].x[1] = reshape(net.x[1], im_size, im_size, 3)[
					global_patch_positions[i,1]:global_patch_positions[i,1]+net.patchsize-1,
					global_patch_positions[i,2]:global_patch_positions[i,2]+net.patchsize-1,:][:]
			else
				net.AE_sparse_patches[i].x[1] =
					reshape(net.x[1],im_size,im_size)[
					global_patch_positions[i,1]:global_patch_positions[i,1]+net.patchsize-1,
					global_patch_positions[i,2]:global_patch_positions[i,2]+net.patchsize-1][:]
			end
		end
	else
		n_patch = Int(sqrt(net.n_AE_sparse_patches))
		ol = net.patchsize - net.stride
		for i in 1:n_patch
			for j in 1:n_patch
				if colorimage
					net.AE_sparse_patches[(i-1)*n_patch+j].x[1] = reshape(net.x[1],im_size,im_size,3)[(i-1)*net.stride+1 : i*net.patchsize-(i-1)*ol,
					   	(j-1)*net.stride+1 : j*net.patchsize-(j-1)*ol,:][:]
				else
					net.AE_sparse_patches[(i-1)*n_patch+j].x[1] =
						reshape(net.x[1],im_size,im_size)[(i-1)*net.stride+1 : i*net.patchsize-(i-1)*ol,
		              (j-1)*net.stride+1 : j*net.patchsize-(j-1)*ol][:]
				end
			end
		end
	end
end


function forwardprop_Brito_full!(net::AE_sparse_patchy, lambdas, iter;
									nonlinearity = nonlinearity, no_recurrence = false,
									Földiak_model = false, rand_patch_pos = false,
									colorimage = false)
	net.x[end] .= 0
	distributeinput!(net; rand_patch_pos = rand_patch_pos, colorimage = colorimage)
	for i in 1:net.n_AE_sparse_patches
		forwardprop_Brito_full!(net.AE_sparse_patches[i], lambdas, iter; nonlinearity = nonlinearity,
			no_recurrence = no_recurrence, Földiak_model = Földiak_model)
		net.x[2][(i-1)*net.n_out_per_population+1:i*net.n_out_per_population] = deepcopy(net.AE_sparse_patches[i].x[end])
	end
end
#for generatehiddenreps
function forwardprop_Brito_full!(net::AE_sparse_patchy, lambdas;
									nonlinearity = nonlinearity, no_recurrence = false,
									Földiak_model = false, rand_patch_pos = false,
									colorimage = false)
	iter = -1
	forwardprop_Brito_full!(net, lambdas, iter; nonlinearity = nonlinearity,
		no_recurrence = no_recurrence, Földiak_model = Földiak_model, rand_patch_pos = rand_patch_pos,
		colorimage = colorimage)
end
