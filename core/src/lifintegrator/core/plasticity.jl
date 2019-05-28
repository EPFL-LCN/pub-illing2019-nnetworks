"""
    struct NoPlasticity end
"""
struct NoPlasticity end
export NoPlasticity
updateweights!(net, i) = Nothing

"""
    mutable struct TeacherSignalPlasticity
        islearning::Bool
        isweightplastic::Array{Bool, 1}
        plasticweights::Array{Int64, 1}
        targets::Array{Float64, 1}
        learningrate::Float64
        weightdecaybound::Float64
    end
"""
@def plasticity_common_fields begin
    islearning::Bool
    isweightplastic::Array{Bool, 1}
    plasticweights::Array{Int64, 1}
    targets::Array{Float64, 1}
    learningrate::Float64
    weightdecaybound::Float64
end
mutable struct TeacherSignalPlasticity
    @plasticity_common_fields
end
export TeacherSignalPlasticity
function TeacherSignalPlasticity(isweightplastic, targets, learningrate; weightdecaybound = Inf)
    TeacherSignalPlasticity(true, isweightplastic, findall(isweightplastic),
                            targets, learningrate, weightdecaybound)
end

"""
    mutable struct TeacherSignalPlasticitySTDP
        islearning::Bool
        isweightplastic::Array{Bool, 1}
        plasticweights::Array{Int64, 1}
        targets::Array{Float64, 1}
        learningrate::Float64
        weightdecaybound::Float64
        lastspiketimesatsynapses::Array{Float64, 1}
        pretracesatlastupdate::Array{Float64, 1}
        posttracesatlastupdate::Array{Float64, 1}
    end
"""
mutable struct TeacherSignalPlasticitySTDP
    @plasticity_common_fields
    lastspiketimesatsynapses::Array{Float64, 1} #time of last spike happening at every synapse
    pretracesatlastupdate::Array{Float64, 1} #trace (as pre-neuron) at last update (pre or post spike) at every synapse
    posttracesatlastupdate::Array{Float64, 1} #trace (as post-neuron) ''
end
export TeacherSignalPlasticitySTDP
function TeacherSignalPlasticitySTDP(isweightplastic, targets, learningrate; weightdecaybound = Inf)
    TeacherSignalPlasticitySTDP(true, isweightplastic, findall(isweightplastic),
                            targets, learningrate, weightdecaybound, zeros(length(isweightplastic)),
                            zeros(length(isweightplastic)), zeros(length(isweightplastic)))
end

################################################################################
# Plasticity Rules
################################################################################

@inline deltarule(lr, pre, post, target) = lr * (target - post) #* pre
export deltarule
@inline function STDPrule(lr, pretracesatlastupdate, posttracesatlastupdate, lastspiketimeatsynapse,
                          target, spiketracetimeconstant, nettime)
    lr * spiketracetimeconstant * pretracesatlastupdate *
    (target * (1 - exp((lastspiketimeatsynapse - nettime) / spiketracetimeconstant))
    - posttracesatlastupdate * sqrt(pi)/2 * erf((nettime - lastspiketimeatsynapse) / spiketracetimeconstant))
    #- posttracesatlastupdate * sqrt(pi) * erf((nettime - lastspiketimeatsynapse) / spiketracetimeconstant))
end

@inline function weightdecay(lr, pre, post, target, weightbound)
  (1 - lr / weightbound)
end

################################################################################
# Weight update
################################################################################

@inline function updateweights!(net::AbstractNet{TeacherSignalPlasticity}, i)
    !net.plasticityrule.islearning && return 0.
    for post in net.posts[i]
        if net.plasticityrule.isweightplastic[post[2]]
            net.weights[post[2]] += deltarule(net.plasticityrule.learningrate,
                                              net.spiketrace[i],
                                              net.spiketrace[post[1]],
                                              net.plasticityrule.targets[post[1]])
            (net.plasticityrule.weightdecaybound != Inf) &&
            (net.weights[post[2]] *= weightdecay(net.plasticityrule.learningrate, net.spiketrace[i], net.spiketrace[post[1]],
                                            net.plasticityrule.targets[post[1]], net.plasticityrule.weightdecaybound))
        end
    end
end

@inline function updateweights!(net::AbstractNet{TeacherSignalPlasticitySTDP}, i)
    for post in net.posts[i] #update all synapses to postsynaptic neurons
        if net.plasticityrule.isweightplastic[post[2]]
            if net.plasticityrule.islearning
              net.weights[post[2]] += STDPrule(net.plasticityrule.learningrate,
                                                net.plasticityrule.pretracesatlastupdate[post[2]], #pretrace at last spike
                                                net.plasticityrule.posttracesatlastupdate[post[2]], #post ''
                                                net.plasticityrule.lastspiketimesatsynapses[post[2]], #time of last spike
                                                net.plasticityrule.targets[post[1]],
                                                net.parameters.spiketrace_timeconstant,
                                                net.t)
            end
            net.plasticityrule.pretracesatlastupdate[post[2]] = net.spiketrace[i] #Save present states and times into memory variables
            net.plasticityrule.posttracesatlastupdate[post[2]] = net.spiketrace[post[1]]
            net.plasticityrule.lastspiketimesatsynapses[post[2]] = net.t
        end
    end
    for pre in net.pres[i] #update all synapses to presynaptic neurons
        if net.plasticityrule.isweightplastic[pre[2]]
            if net.plasticityrule.islearning
              net.weights[pre[2]] += STDPrule(net.plasticityrule.learningrate,
                                                net.plasticityrule.pretracesatlastupdate[pre[2]],
                                                net.plasticityrule.posttracesatlastupdate[pre[2]],
                                                net.plasticityrule.lastspiketimesatsynapses[pre[2]],
                                                net.plasticityrule.targets[i],
                                                net.parameters.spiketrace_timeconstant,
                                                net.t)
            end
            net.plasticityrule.pretracesatlastupdate[pre[2]] = net.spiketrace[pre[1]]
            net.plasticityrule.posttracesatlastupdate[pre[2]] = net.spiketrace[i]
            net.plasticityrule.lastspiketimesatsynapses[pre[2]] = net.t
        end
    end
end
