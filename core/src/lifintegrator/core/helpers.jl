

"""
    firingrates(recorder, start, end; n = "max {neuronindex | neuron spiked}")

Extract firing rates in internval [`start`, `end`] from `recorder`.
"""
function firingrates(rec, start, stop;
                     nneurons = length(rec.spikes) > 0 ?
                        maximum([s[1] for s in rec.spikes]) : 0)
    nneurons == 0 && return [0]
    n = zeros(nneurons)
    for spike in rec.spikes
        if spike[2] > start && spike[2] <= stop
            n[spike[1]] += 1
        end
    end
    1000 .* n /(stop - start) #factor 1000 for unit Hz
end
export firingrates

"""
    ISI

extract inter-spike-intervals
"""
function ISI(rec, start, stop;
              nneurons = length(rec.spikes) > 0 ?
                maximum([s[1] for s in rec.spikes]) : 0)
      nneurons == 0 && return [0]
      ISI = []
      isi = []
      for i in 1:nneurons
        oldspiketime = 0
        for spike in rec.spikes
            if spike[2] > start && spike[2] <= stop
                if spike[1] == i
                    (oldspiketime != 0) && push!(isi,spike[2]-oldspiketime)
                    oldspiketime = spike[2]
                end
            end
        end
        push!(ISI,isi)
      end
      ISI
end
export ISI
"""
    getMNISTinput
"""
function getMNISTinput(smallimgs, labels, threshold, n_hidden, n_out; in_amplitude = 1., index = 0)
    if index == 0
        patternindex = rand(1:size(smallimgs)[2])
    else
        patternindex = index
    end
    input = [in_amplitude*smallimgs[:,patternindex]; zeros(n_hidden); zeros(n_out)] .+ threshold
    return input, labels[patternindex]
end
export getMNISTinput
"""
    MNISTselect
"""
function MNISTselect(images,labels,digits::Array{Int64})
  indices = findall(x -> (x in digits),labels)
  selected_images = images[:,indices]
  selected_labels = labels[indices]
  return selected_images, selected_labels
end
export MNISTselect
"""
    set_connectivity!
"""
function set_connectivity!(net, inputsize, hiddensize, patchsize)
    weights = reshape(deepcopy(net.weights[1:hiddensize*inputsize]),hiddensize,inputsize)
    input_dim = Int(sqrt(inputsize))
    if patchsize != input_dim
    	for i in 1:hiddensize
          mask = zeros(input_dim,input_dim) #initialize mask
          mask[1:patchsize,1:patchsize] .= 1.
          shifts = rand(0:input_dim-patchsize,2) #draw random shifts of mask
          weights[i,:] = reshape(reshape(weights[i,:],input_dim,input_dim) .* circshift(mask,shifts),inputsize)
          weights[i,:] .*= 1 .* input_dim./(patchsize) #rescale weights for same input strength as full connectivity
    	end
    end
    net.weights[1:hiddensize * inputsize] = weights[:]
end
export set_connectivity!
function set_connectivity!(net::RateNet,patch_size; colorimage = false)
  if colorimage
    input_size = Int(length(net.input[1])/3)
  else
    input_size = Int(length(net.input[1]))
  end
  input_dim = Int(sqrt(input_size))
  mask = zeros(input_dim,input_dim) #initialize mask
  mask[1:patch_size,1:patch_size] .= 1.
	if patch_size != input_dim
		nr_hidden_neurons = length(net.input[2])
        if colorimage
      		for i in 1:nr_hidden_neurons
      			shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
      			for j in 1:3
      			  net.weights[1][i,(j-1)*input_size+1:j*input_size] = reshape(reshape(net.weights[1][i,(j-1)*input_size+1:j*input_size],input_dim,input_dim).*circshift(mask,shifts),input_size)
                  net.weights[1][i,(j-1)*input_size+1:j*input_size] .*= 0.5 * input_dim./(patch_size) #rescale weights for same input strength as full connectivity
      			end
      		end
      	else
      		for i in 1:nr_hidden_neurons
                shifts = rand(0:input_dim-patch_size,2)#draw random shifts of mask
          		net.weights[1][i,:] = reshape(reshape(net.weights[1][i,:],input_dim,input_dim).*circshift(mask,shifts),input_size)
                net.weights[1][i,:] .*= 0.5 .* input_dim./(patch_size) #rescale weights for same input strength as full connectivity
                # Factor 0.5 for short tau = 25 ms compared to 50 ms (factor 1000/tau)
      		end
      	end
	end
end
export set_connectivity!
"""
    set_connectivity_gabor!
"""
function set_connectivity_gabor!(net, inputsize, hiddensize, patchsize;
    λrange = [patchsize/4,2*patchsize], ψupperbound = 2π,
    σrange = [patchsize/8,patchsize], γrange = 0, amplitude = 1)
    weights = zeros(hiddensize, inputsize)
    input_dim = Int(sqrt(inputsize))
    for i in 1:hiddensize
        mask = zeros(input_dim,input_dim)
        mask[1:patchsize,1:patchsize] = randMNISTgaborfilterBayesOpt(patchsize, λrange, ψupperbound, σrange, γrange, amplitude)
        shifts = rand(0:input_dim-patchsize,2)
        weights[i,:] = 20 * 2 .* circshift(mask,shifts)[:]
        #weights[i,:] .*= 1. / norm(weights[i,:])
    end
    net.weights[1:hiddensize * inputsize] = weights[:]
end
export set_connectivity_gabor!
function set_connectivity_gabor!(net::RateNet, patchsize; colorimage = false,
      λrange = [patchsize/4,2*patchsize], ψupperbound = 2π,
      σrange = [patchsize/8,patchsize], γrange = 0, amplitude = 1)
    if colorimage
          input_size = Int(length(net.input[1])/3)
    else
          input_size = Int(length(net.input[1]))
    end
	input_dim = Int(sqrt(input_size))
    hiddensize = length(net.input[2])
    if colorimage
        for i in 1:hiddensize
            mask = zeros(input_dim,input_dim)
            mask[1:patchsize,1:patchsize] =
                randMNISTgaborfilterBayesOpt(patchsize, λrange, ψupperbound, σrange, γrange, amplitude)
            shifts = rand(0:input_dim-patchsize,2)
            for j in 1:3
              net.weights[1][i,(j-1)*input_size+1:j*input_size] = circshift(mask,shifts)[:]
            end
        end
    else
        for i in 1:hiddensize
            mask = zeros(input_dim,input_dim)
            mask[1:patchsize,1:patchsize] =
                randMNISTgaborfilterBayesOpt(patchsize, λrange, ψupperbound, σrange, γrange, amplitude)#randMNISTgaborfilter(patchsize)#randgaborfilter(patchsize)
            shifts = rand(0:input_dim-patchsize,2)
            net.weights[1][i,:] = circshift(mask,shifts)[:]
            #net.weights[1][i,:] .*= 1. / norm(net.weights[1][i,:])
            #net.weights[1][i,:] .*= 1. / patchsize
        end
    end
end
export set_connectivity_gabor!
"""
    geterror!
"""
function geterror!(net,imgs,lbls,threshold,n_hidden,n_out,in_amplitude,ntests,testtime)
  print("\n Calculate classification error...\n")
  (ntests > size(imgs)[2]) && error("ntests is higher than number of samples in current data set")
  error = 0.
  rates_test = []
  net_test = deepcopynet(net; recorder = AllSpikesRecorder(), plasticityrule = NoPlasticity)
  endtime = net_test.t
  @showprogress for i in 1:ntests
    endtime += testtime
    input, label = getMNISTinput(imgs, lbls, threshold, n_hidden, n_out;
      in_amplitude = in_amplitude, index = i)
    integratenet!(net_test, input, endtime)
    rates_test = firingrates(net_test.recorder, net_test.t - 0.5*testtime, net_test.t;
                              nneurons = size(imgs)[1]+n_hidden+n_out)
    guess = findmax(rates_test[end-n_out+1:end])[2]
    if guess != Int(label+1) error += 1 end
    deleteoldestspikes!(net_test.recorder, net_test.t)
  end
  error /= ntests*0.01
  return error, net_test.t
end
export geterror!
"""
    deleteoldestspikes!
"""
# For emptying all spikes: empty!(rec.spikes)
function deleteoldestspikes!(rec, until)
    inds = []
    for i in 1:length(rec.spikes)
      (rec.spikes[i][2] <= until) && push!(inds,i)
    end
    deleteat!(rec.spikes,inds)
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


"""
    subtractmean!
"""
function subtractmean!(data)
        m = mean(data, dims = 2)
        d, n = size(data)
        for i in 0:n-1
                BLAS.axpy!(-1., m, 1:d, data, i*d + 1: (i+1) * d)
        end
end
export subtractmean!
"""
    subtractmean
"""
function subtractmean(data)
        m = deepcopy(data)
        subtractmean!(m)
        m
end
