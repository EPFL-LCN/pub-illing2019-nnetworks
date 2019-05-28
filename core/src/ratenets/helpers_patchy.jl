using ProgressMeter, MultivariateStats

function getdatapatch(data, pos, imsize; patchsize = 10, colorimage = false)
    if colorimage
        data_all_colors = reshape(reshape(data[1:imsize^2,:], imsize, imsize, size(data)[2])[pos[1]:pos[1] + patchsize - 1,
            pos[2]:pos[2] + patchsize - 1,:], patchsize^2, size(data)[2])
        for i in 2:3
            data_all_colors = vcat(data_all_colors, reshape(reshape(data[(i-1) * imsize^2 + 1:i * imsize^2,:], imsize, imsize, size(data)[2])[pos[1]:pos[1] + patchsize - 1,
                    pos[2]:pos[2] + patchsize - 1,:], patchsize^2, size(data)[2]))
        end
        data_all_colors
    else
        reshape(reshape(data, imsize, imsize, size(data)[2])[pos[1]:pos[1] + patchsize - 1,
            pos[2]:pos[2] + patchsize - 1,:], patchsize^2, size(data)[2])
    end
end

function getPCAscores(data, datatest, n_hidden_per_patch)
    (n_hidden_per_patch >= size(data)[1]) && error("cannot extract more PCs than (patch) input dimension")
    PCA_fit = fit(PCA, data, pratio = 1., maxoutdim = n_hidden_per_patch)
    return transform(PCA_fit, data), transform(PCA_fit, datatest)
end

function getICAscores(data, datatest, n_hidden_per_patch)
    (n_hidden_per_patch >= size(data)[1]) && error("cannot extract more ICs than (patch) input dimension")
    ICA_fit = fit(ICA, hcat(data,datatest), n_hidden_per_patch, tol = 2.5 * sqrt(n_hidden_per_patch), do_whiten=true)#false) #
    return transform(ICA_fit, data), transform(ICA_fit, datatest)
end

# output: smallimgs-array for shallow net!
function getpatchyhiddenreps(data, datatest, n_patches, n_hidden_per_patch;
    featurefunction = getPCAscores, patchsize = 10, randpos = true, colorimage = false)

    smallimgs = zeros(n_patches * n_hidden_per_patch, size(data)[2])
    smallimgstest = zeros(n_patches * n_hidden_per_patch, size(datatest)[2])

    if colorimage
        input_size = Int(size(data)[1]/3)
    else
        input_size = Int(size(data)[1])
    end
    imsize = Int(sqrt(input_size))
    print(string("calculate unsupervised hidden representations with function: ",string(featurefunction),"\n"))
    @showprogress for i in 1:n_patches
        if !randpos
            error("randpos must be true (in this state of the code)")
        elseif randpos
            if patchsize == imsize
                pos = [1,1]
            else
                pos = rand(1:imsize - patchsize,2)
            end
            patchdata = getdatapatch(data, pos, imsize; patchsize = patchsize, colorimage = colorimage)
            patchdatatest = getdatapatch(datatest, pos, imsize; patchsize = patchsize, colorimage = colorimage)
            smallimgs[(i-1) * n_hidden_per_patch + 1:i * n_hidden_per_patch,:],
             smallimgstest[(i-1) * n_hidden_per_patch + 1:i * n_hidden_per_patch,:] =
                        featurefunction(patchdata, patchdatatest, n_hidden_per_patch)
        end
    end
    return smallimgs ./ maximum(smallimgs), smallimgstest ./ maximum(smallimgs)
end
