
#####################################################
#Train functions

#Train classifier in S(D)AE with classifier
function train!(sae,smallimgs,
	 			labels,
				smallimgstest,
				labelstest,
				conf::configClass)
  		errors = zeros(3,length(conf.hidden_sizes),conf.n_inits)
		nets = Array{AE, 1}()
	#get hidden repr. of last AE in stack (n_ae is global!):
	global n_ae = sae.nae+1
	for i in 1:length(conf.hidden_sizes)
		if conf.hidden_sizes[i] == 0
			print("-> shallow classifier...\n")
		else
			print(string("-> classifier with 1 hidden layer with",conf.hidden_sizes[i]," hidden neurons...\n"))
		end
		for j in 1:conf.n_inits
			if conf.hidden_sizes[i] == 0
				net = Classifier([size(sae.reps[end][1])[1], 10])
			else
				net = Classifier([size(sae.reps[end][1])[1], conf.hidden_sizes[i], 10])
			end
      		@time loss = learn!(net, sae, gethiddenreps, getlabel, conf.iterations, conf.learningrates, conf.weight_decay,
                            nonlinearity = conf.nonlinearity, nonlinearity_diff = conf.nonlinearity_diff,
                            nonlin_diff = conf.nonlin_diff)
			push!(nets,deepcopy(net))
			error_train, error_test = printerrors(net, sae.reps[end][1], labels, sae.reps[end][2], labelstest, conf)
      		errors[:,i,j] = [conf.hidden_sizes[i], error_train, error_test]
     	end
  	end
  	errors, nets
end

function trainSAEc_full_patchy!(smallimgs,smallimgstest,labels,labelstest,
									n_out_per_population,conf;
									patchsize = 10,
									n_of_populations = 100,
									stride = 3, im_size = 28,
									no_recurrence_after_training = false,
									Földiak_model = false,
									rand_patch_pos = true,
									colorimage = false)

	n_in = size(smallimgs)[1]
	AE_s_p = AE_sparse_patchy(n_in, n_out_per_population, n_of_populations;
							patchsize = patchsize, stride = stride,
							im_size = im_size, colorimage = colorimage)
	rand_patch_pos && (global global_patch_positions = getrandompatchpositions(AE_s_p; colorimage = colorimage))
	confae = configAE_sparse(conf.configSAE_sparse.task,conf.configSAE_sparse.hidden_sizes[1],
				conf.configSAE_sparse.n_inits,conf.configSAE_sparse.iterations[1],
				conf.configSAE_sparse.learningrates[1],conf.configSAE_sparse.lambdas[1],
				conf.configSAE_sparse.weight_decay,conf.configSAE_sparse.noise_type,
				conf.configSAE_sparse.noise[1],conf.configSAE_sparse.nonlinearity[1],
				conf.configSAE_sparse.nonlinearity_diff[1],true)

	learnAE_Brito_patchy!(AE_s_p, getsmallimg, confae.iterations, confae.learningrates,
						confae.weight_decay, confae.noise_type, confae.noise, confae.lambda;
						nonlinearity = confae.nonlinearity, nonlinearity_diff = confae.nonlinearity_diff,
						nonlin_diff = confae.nonlin_diff, BPuptolayer = AE_s_p.AE_sparse_patches[1].nl,
						Földiak_model = Földiak_model, rand_patch_pos = rand_patch_pos,
						colorimage = colorimage)

	#construct SAE with trained patchy SAE
	sae = SAE_sparse_patchy(AE_s_p.n_AE_sparse_patches,n_in,n_out_per_population;
							n_samples = [size(smallimgs)[2], size(smallimgstest)[2]])
	sae.nae = 1
	sae.aes = [deepcopy(AE_s_p)]

	generatehiddenreps_Brito!(sae,1, smallimgs, smallimgstest, confae;
		no_recurrence = no_recurrence_after_training,
		Földiak_model = Földiak_model, rand_patch_pos = rand_patch_pos,
		colorimage = colorimage)
	for reps in sae.reps[1]
		reps .*= 1/maximum(reps)
	end
	print("train top-end classifier...\n")
	errors, nets = train!(sae,smallimgs,labels,smallimgstest,labelstest,conf.configClass)

	return sae, errors
end
