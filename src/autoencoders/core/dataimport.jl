
#############################################################
#Data Import and Preprocesing

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

function subtractmean!(data)
        m = mean(data, dims = 2)
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

function normalizelinewise!(data)
    max = maximum(data, dims = 2)
    d, n = size(data)
    for i in 1:d
        (max[i] != 0.) && (data[i,:] ./= max[i])
    end
    subtractmean!(data)
end

function standardizelinewise!(data)
    means = mean(data, dims = 2)
    stds = std(data, dims = 2)
    d, n = size(data)
    for i in 1:d
        data[i,:] .-= means[i]
        (stds[i] != 0.) && (data[i,:] ./= stds[i])
    end
end

#scale data between [-1,1]
function rescaledata!(data)
        absmax = maximum(abs(data))
        scale!(data, 1. / absmax)
end

# data is dxn array of n d-dimensional random vectors with mean 0
function whiten(data; method = :f_ZCA)
        f = svdfact(data)
        eval(method)(f) * sqrt(size(data, 2) - 1)
end

f_ZCA(f::LinearAlgebra.SVD) = f[:U] * f[:Vt]
f_PCA(f::LinearAlgebra.SVD) = f[:Vt]
