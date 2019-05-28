
#####################################################
#nonlinearities

function relu!(inp, outp)
	for i in 1:length(inp)
		outp[i] = max(0, inp[i])
	end
end

function relu!(inp, outp, t)
	for i in 1:length(inp)
		outp[i] = max(0, inp[i] - t[i])
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

function sigm!(inp, outp)
	for i in 1:length(inp)
		outp[i] = 1. / (1. + exp(-inp[i]))
	end
end

function sigm_diff!(ax, error)
	for i in 1:length(ax)
		error[i] *= 1. / (1 + exp(-ax[i]))*(1. - 1. / (1 + exp(-ax[i])))
	end
end

function pwl!(inp, outp, t)
	for i in 1:length(inp)
		outp[i] = clamp(inp[i] - t[i],0.,1.) #piece-wise linear
	end
end

function sigm!(inp, outp, t; λ = 10.) #10
	for i in 1:length(inp)
		outp[i] = 1. / (1. + exp(- λ * (inp[i] - t[i])))
	end
end
