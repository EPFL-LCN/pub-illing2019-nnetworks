
#####################################################
#Helpers

function getsparsity(input::Array{Float64, 1})
	length(findall(x -> (x == 0),input))/length(input)
end

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

function gethiddenreps(sae)
		global patternindex = rand(1:size(smallimgs)[2])
		#sae.reps[n_ae][1][:, patternindex]
		sae.reps[n_ae-1][1][:, patternindex]
end

function gethiddenrepstest(sae)
		global patternindex = rand(1:size(smallimgstest)[2])
		#sae.reps[n_ae][1][:, patternindex]
		sae.reps[n_ae-1][2][:, patternindex]
end

function _generatehiddenreps!(sae,n_ae::Int64, smallimgs, smallimgstest, conf, func_forwardprop::Function;
		no_recurrence = false, Földiak_model = false, rand_patch_pos = false, colorimage = false)
	if typeof(conf) == configAE
		if n_ae == 1
			@showprogress for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = smallimgs[:, i]
				func_forwardprop(sae.aes[n_ae]; nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
			@showprogress for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = smallimgstest[:, i]
				func_forwardprop(sae.aes[n_ae]; nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
		else
			@showprogress for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][1][:,i])
				func_forwardprop(sae.aes[n_ae]; nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
			@showprogress for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][2][:,i])
				func_forwardprop(sae.aes[n_ae]; nonlinearity = conf.nonlinearity)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[Int(mean([1,sae.aes[n_ae].nl+1]))])
			end
		end
	elseif typeof(conf) == configAE_sparse
		if n_ae == 1
			for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = smallimgs[:, i]
				func_forwardprop(sae.aes[n_ae], conf.lambda; nonlinearity = conf.nonlinearity,
					no_recurrence = no_recurrence, Földiak_model = Földiak_model,
					rand_patch_pos = rand_patch_pos, colorimage = colorimage)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
			for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = smallimgstest[:, i]
				func_forwardprop(sae.aes[n_ae], conf.lambda; nonlinearity = conf.nonlinearity,
					no_recurrence = no_recurrence, Földiak_model = Földiak_model,
					rand_patch_pos = rand_patch_pos, colorimage = colorimage)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
		else
			for i in 1:size(smallimgs)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][1][:,i])
				func_forwardprop(sae.aes[n_ae], conf.lambda; nonlinearity = conf.nonlinearity,
					no_recurrence = no_recurrence, Földiak_model = Földiak_model,
					rand_patch_pos = rand_patch_pos, colorimage = colorimage)
				sae.reps[n_ae][1][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
			for i in 1:size(smallimgstest)[2]
				sae.aes[n_ae].x[1] = deepcopy(sae.reps[n_ae-1][2][:,i])
				func_forwardprop(sae.aes[n_ae], conf.lambda; nonlinearity = conf.nonlinearity,
					no_recurrence = no_recurrence, Földiak_model = Földiak_model,
					rand_patch_pos = rand_patch_pos, colorimage = colorimage)
				sae.reps[n_ae][2][:,i] = deepcopy(sae.aes[n_ae].x[end])
			end
		end
	end
end

#for sparse coding (only 2-layer AEs!)
function generatehiddenreps!(sae::SAE_sparse,n_ae::Int64, smallimgs, smallimgstest, conf::configAE_sparse)
	print(string("compute sparse hidden representations (for train and test sets) of AE nr.",n_ae,"...\n"))
	_generatehiddenreps!(sae,n_ae, smallimgs, smallimgstest, conf, forwardprop!)
end

#for sparse coding à la Brito (only 2-layer AEs!)
function generatehiddenreps_Brito!(sae::SAE_sparse,n_ae::Int64, smallimgs, smallimgstest, conf::configAE_sparse;
		no_recurrence = false, Földiak_model = false, rand_patch_pos = false, colorimage = false)
	print(string("compute sparse hidden representations (for train and test sets) of AE nr.",n_ae,"...\n"))
	_generatehiddenreps!(sae,n_ae, smallimgs, smallimgstest, conf, forwardprop_Brito_full!;
		no_recurrence = no_recurrence, Földiak_model = Földiak_model, rand_patch_pos = rand_patch_pos, colorimage = colorimage) #forwardprop_Brito!
end
function generatehiddenreps_Brito!(sae::SAE_sparse_patchy,n_ae::Int64, smallimgs, smallimgstest, conf::configAE_sparse;
		no_recurrence = false, Földiak_model = false, rand_patch_pos = false, colorimage = false)
	print(string("compute sparse hidden representations (for train and test sets) of AE nr.",n_ae,"...\n"))
	_generatehiddenreps!(sae,n_ae, smallimgs, smallimgstest, conf, forwardprop_Brito_full!;
		no_recurrence = no_recurrence, Földiak_model = Földiak_model, rand_patch_pos = rand_patch_pos, colorimage = colorimage) #forwardprop_Brito!
end



function geterrors(net, imgs, labels; nonlinearity = nonlinearity)
	error = 0
	noftest = size(imgs)[2]
	for i in 1:noftest
		net.x[1] = imgs[:, i]
		forwardprop!(net, nonlinearity = nonlinearity)
		error += findmax(net.x[end])[2] != labels[i] + 1
	end
	error/noftest
end
function printerrors(net, smallimgs, labels, smallimgstest, labelstest, conf)
	error_train = geterrors(net, smallimgs, labels, nonlinearity = conf.nonlinearity)
	error_test = geterrors(net, smallimgstest, labelstest, nonlinearity = conf.nonlinearity)
	print(string(100 * error_train, " % on training set\n"))
	print(string(100 * error_test, " % on test set\n"))

	return error_train, error_test
end

function _rescale_weights!(net, i, input_dim, patch_size, mode)
	if mode == "randn"
		net.w[1][i,:] .*= 3 * input_dim / (patch_size) #rescale weights for same input strength as full connectivity
	elseif mode == "spherical"
		net.w[1][i,:] .*= 1. / norm(net.w[1][i,:]) # normalize random patches: since sampled from randn they lie on the surface of a hypersphere now! norm grows as sqrt(eff. in-dim)
		net.w[1][i,:] .*= 3.#sqrt(patch_size) # heuristic rescaling for best performance
	end
end
function set_connectivity!(net,patch_size; mode = "randn", colorimage = false)
	if colorimage
		input_size = Int(length(net.x[1])/3)
	else
		input_size = length(net.x[1])
	end
	input_dim = Int(sqrt(input_size))
	nr_hidden_neurons = length(net.x[2])
	mask = zeros(input_dim,input_dim) #initialize mask
	mask[1:patch_size,1:patch_size] = 1.
	if colorimage
		for i in 1:nr_hidden_neurons
			shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
			for j in 1:3
				net.w[1][i,(j-1)*input_size+1:j*input_size] = reshape(reshape(net.w[1][i,(j-1)*input_size+1:j*input_size],input_dim,input_dim).*circshift(mask,shifts),input_size)
			end
			_rescale_weights!(net, i, input_dim, patch_size, mode)
		end
	else
		for i in 1:nr_hidden_neurons
			shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
			net.w[1][i,:] = reshape(reshape(net.w[1][i,:],input_dim,input_dim).*circshift(mask,shifts),input_size)
			_rescale_weights!(net, i, input_dim, patch_size, mode)
		end
	end
	w_connect = clamp(ceil(abs(net.w[1])),0,1)
	return w_connect
end
