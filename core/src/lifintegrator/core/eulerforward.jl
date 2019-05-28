mutable struct EulerNet{Tv,Tt,Tw,Tdelay,Tref,TΔt,Tstrace,Tparams,Trecorder,Tplasticity,Tnoise} <: AbstractNet{Tplasticity}
    v::Tv
    n_of_neurons::Int64
    t::Tt
    posts::Array{Array{Tuple{Int64,Int64}, 1}, 1}
    pres::Array{Array{Tuple{Int64,Int64}, 1}, 1}
    weights::Tw
    spiketrace::Tstrace
    spikecandidates::Array{Int64, 1}
    Δt::TΔt
    indelay::Array{Int64, 1}
    inref::Array{Int64, 1}
    delay::Tdelay
    refractory_period::Tref
    parameters::Tparams
    recorder::Trecorder
    plasticityrule::Tplasticity
    noise::Tnoise
end
function EulerNet(net; Δt = .1f0)
    if minimum(net.parameters.delay) == 0
        error("Cannot integrate nets with zero spike transmission delay.
              delay >= Δt")
    end
    EulerNet(net.v, net.n_of_neurons,
             net.t, net.posts, net.pres, net.weights, net.spiketrace,
             Int64[],
             Δt, -ones(Int64, net.n_of_neurons), -ones(Int64, net.n_of_neurons),
             round.(Int64, net.parameters.delay/Δt),
             round.(Int64, net.parameters.refractory_period/Δt),
             net.parameters, net.recorder, net.plasticityrule, net.noise)
end
export EulerNet

"""
    deepcopynet(net;
                recorder = net.recorder,
                plasticityrule = net.plasticityrule)

Copy an EulerNet and optionally replace the `recorder` or the `plasticityrule`.
"""
function deepcopynet(net::EulerNet;
                     recorder = deepcopy(net.recorder),
                     plasticityrule = deepcopy(net.plasticityrule),
                     parameters = deepcopy(net.parameters))
    EulerNet([field == :recorder ? recorder :
         field == :plasticityrule ? plasticityrule :
         field == :parameters ? parameters :
         deepcopy(getfield(net, field)) for field in fieldnames(typeof(net))]...)
end
export deepcopynet


function addspike!(net::EulerNet, t, i)
    net.indelay[i] = getdelay(net.delay, i)
    net.inref[i] = getrefractory_period(net.refractory_period, i)
    (typeof(net.noise) == RectLinEscapeNoise) &&
    (net.noise.endofrefrperiod[i] = t + getrefractory_period(net.parameters.refractory_period, i))
    addspikepostprocess!(net, t, i)
end
@inline function eulerspike!(net::EulerNet, noise::NoNoise, i)
  if abovethreshold(net.v, net.parameters.threshold, i)
      push!(net.spikecandidates, i)
  end
end
@inline getrectlinhazardrate(threshold::Number, v, β, i) = max(0, β * (v[i] - threshold))
@inline getrectlinhazardrate(threshold::AbstractArray, v, β, i) = max(0, β * (v[i] - threshold[i]))
@inline function eulerspike!(net::EulerNet, noise::RectLinEscapeNoise, i)
  if rand() <= net.Δt*getrectlinhazardrate(net.parameters.threshold, net.v, noise.β, i)
    push!(net.spikecandidates, i)
  end
end
function updatenet!(net::EulerNet, input, tmax)
    for eventtime in net.t:net.Δt:tmax
        updatevars!(net, eventtime, input)
        net.t = eventtime
        @inbounds for i in 1:net.n_of_neurons
            eulerspike!(net, net.noise, i)
            if net.indelay[i] == 1
                broadcastspike!(net, i)
                updateweights!(net, i)
            end
            if net.indelay[i] >= 1 net.indelay[i] -= 1 end
            if net.inref[i] == 0
                net.v[i] = getreset(net.parameters.reset, i)
            end
            if net.inref[i] >= 0 net.inref[i] -= 1 end
        end
        processspikecandidates!(net)
    end
    return false
end
