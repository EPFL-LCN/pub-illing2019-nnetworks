__precompile__()
module EventBasedIntegrator
using Statistics, StatsBase, SpecialFunctions, Compat, Distributed, LinearAlgebra

# Macro for common field sharing
macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

struct LIFParams{Ttau,Treset,Tthresh,Tdelay,Tref,Tstrace}
    tau::Ttau
    reset::Treset
    threshold::Tthresh
    delay::Tdelay
    refractory_period::Tref
    spiketrace_timeconstant::Tstrace
end
export LIFParams
"""
    LIFParams(; tau = 10., reset = 0., threshold = 20., delay = .5,
                 refractory_period = 2., spiketrace_timeconstant = 20.)

Each keyword argument accepts either a float or an array of floats with
individual values for each neuron (except `spiketrace_timeconstant` only float).
Choosing individual values for `tau` may increase simulation time a lot.
"""
LIFParams(; tau = 25., reset = 0., threshold = 20., delay = 1e-1,
            refractory_period = 0., spiketrace_timeconstant = 20.) =
LIFParams(tau, reset, threshold, delay, refractory_period,
           spiketrace_timeconstant)

@enum Event SpikeInsert RefractEnd Spike NoEvent

abstract type AbstractNet{Tplasticity} end
mutable struct Net{Tv,Tt,Tw,Tstrace,Tparams,Trecorder,Tplasticity,Tnoise} <: AbstractNet{Tplasticity}
    v::Tv
    n_of_neurons::Int64
    t::Tt
    posts::Array{Array{Tuple{Int64,Int64}, 1}, 1}
    pres::Array{Array{Tuple{Int64,Int64}, 1}, 1}
    weights::Tw
    nextevents::Array{Tuple{Event, Int64, Tt}, 1}
    spikecandidates::Array{Int64, 1}
    recorder::Trecorder
    spiketrace::Tstrace
    parameters::Tparams
    plasticityrule::Tplasticity
    noise::Tnoise
end
export Net
function Net(n_of_neurons, pres, posts, weights, recorder, parameters,
             plasticityrule, noise)
    Net(Float64.(rand(n_of_neurons) .* (parameters.threshold .- parameters.reset) .+
                 parameters.reset),
        n_of_neurons,
        0.,
        pres,
        posts,
        Float64.(weights),
        Tuple{Event, Int64, Float64}[],
        Int64[],
        recorder,
        zeros(n_of_neurons),
        parameters,
        plasticityrule,
        noise)
end

include("escapenoise.jl")
include("eulerforward.jl")
include("recorders.jl")
include("plasticity.jl")
include("ratemodel.jl")
include("helpers.jl")

"""
    function BalancedNet(n_of_neurons, J, g, parameters = LIFParams();
                         f = 3/4,
                         recorder = AllSpikesRecorder(),
                         plasticityrule = NoPlasticity())
"""
function BalancedNet(n_of_neurons, J, g; parameters = LIFParams(),
                     f = 3/4, recorder = AllSpikesRecorder(),
                     plasticityrule = NoPlasticity(),
                     noise = NoNoise())
    posts = [Tuple{Int64,Int64}[] for i in 1:n_of_neurons]
    pres = [Tuple{Int64,Int64}[] for i in 1:n_of_neurons]
    n_of_excitatory = floor(Int64,f*n_of_neurons)
    n_of_inhibitory = floor(Int64, (1-f)*n_of_neurons)
    weights = Float64[J; -g*J]
    for post in 1:n_of_neurons
        for pre in sample(1:n_of_excitatory, div(n_of_excitatory, 10),
                            replace = false)
            push!(posts[pre], (post, 1))
            push!(pres[post], (pre, 1))
        end
        for pre in sample(n_of_excitatory+1:n_of_neurons, div(n_of_inhibitory, 10),
                            replace = false)
            push!(posts[pre], (post, 2))
            push!(pres[post], (pre, 2))
        end
    end
    Net(n_of_neurons, posts, pres, weights, recorder, parameters, plasticityrule, noise)
end
export BalancedNet
"""
    FeedForwardNet(layersizes, parameters = LIFParams();
                   sampleinitw =  (lpre, i, lpost, j) -> .1 * rand(),
                   recorder = AllSpikesRecorder(),
                   plasticityrule = NoPlasticity())
"""
function FeedForwardNet(layersizes, parameters = LIFParams();
                        sampleinitw =  (lpre, i, lpost, j) -> randn() .* 20 ./ sqrt(layersizes[lpre]),
                        recorder = AllSpikesRecorder(),
                        plasticityrule = NoPlasticity(),
                        noise = NoNoise())
    ns = [0; cumsum(layersizes)]
    posts = [Tuple{Int64,Int64}[] for i in 1:ns[end]]
    pres = [Tuple{Int64,Int64}[] for i in 1:ns[end]]
    weights = Float64[]
    weightsindex = 1
    for i in 1:length(layersizes) - 1
        for pre in ns[i] + 1:ns[i+1]
            for post in ns[i+1] + 1:ns[i+2]
                if (w = sampleinitw(i, pre - ns[i], i+1, post - ns[i+1])) != 0
                    push!(posts[pre], (post, weightsindex))
                    push!(pres[post], (pre, weightsindex))
                    push!(weights, w)
                    weightsindex += 1
                end
            end
        end
    end
    Net(ns[end], posts, pres, weights, recorder, parameters, plasticityrule, noise)
end
export FeedForwardNet
"""
    deepcopynet(net;
                recorder = net.recorder,
                plasticityrule = net.plasticityrule)

Copy a net and optionally replace the `recorder` or the `plasticityrule`.
"""
function deepcopynet(net;
                     recorder = deepcopy(net.recorder),
                     plasticityrule = deepcopy(net.plasticityrule),
                     parameters = deepcopy(net.parameters))
    Net([field == :recorder ? recorder :
         field == :plasticityrule ? plasticityrule :
         field == :parameters ? parameters :
         deepcopy(getfield(net, field)) for field in fieldnames(typeof(net))]...)
end
export deepcopynet

let basetypes = (Number, AbstractArray)
    for vargt in basetypes; for inargt in basetypes; for targt in basetypes;
        v = vargt == Number ? :v : :(v[i])
        input = inargt == Number ? :input : :(input[i])
        threshold = targt == Number ? :threshold : :(threshold[i])
        eval(:(@inline arglog(v::$vargt, input::$inargt, threshold::$targt, i::Int64) =
               arglog($v, $input, $threshold)))
    end; end; end
end
@inline function arglog(v, input, threshold)
    threshold >= input && return typeof(v)(Inf)
    (v - input)/(threshold - input)
end

for func in (:gettau, :getrefractory_period, :getdelay, :getreset)
    eval(:(@inline $func(x::Number, i) = x))
    eval(:(@inline $func(x::AbstractArray, i) = x[i]))
end

@inline function nextspiketime(arglog, parameters, i)
    gettau(parameters.tau, i) * log(arglog)
end

function insert_event!(nextevents, event)
    if length(nextevents) == 0 || nextevents[end][3] <= event[3]
        push!(nextevents, event)
        return
    end
    i = length(nextevents)
    while i > 0 && nextevents[i][3] > event[3]
        i -= 1
    end
    insert!(nextevents, i+1, event)
end

@inline function addspikepostprocess!(net, t, i)
    recordspike!(net.recorder, net, (i, t))
    if net.parameters.spiketrace_timeconstant > 0
        net.spiketrace[i] += 1/net.parameters.spiketrace_timeconstant
    end
    net.v[i] = -typeof(net.v[i])(Inf)
end
function addspike!(net, t, i)
    insert_event!(net.nextevents, (RefractEnd, i, t +
                    getrefractory_period(net.parameters.refractory_period, i)))
    (typeof(net.noise) == RectLinEscapeNoise) &&
    (net.noise.endofrefrperiod[i] = t + getrefractory_period(net.parameters.refractory_period, i))
    insert_event!(net.nextevents, (SpikeInsert, i, t +
                                   getdelay(net.parameters.delay, i)))
    addspikepostprocess!(net, t, i)
end

@inline addinput!(gamma, input::Number, v) = v .+= (1 .- gamma) .* input
@inline addinput!(gamma::Number, input::AbstractArray, v) =
    axpy!((1 - gamma), input, v)
@inline function addinput!(gamma::AbstractArray, input::AbstractArray, v)
    v .+= (1 .- gamma) .* input
end

function updatevars!(net, eventtime, input)
    gamma = exp.(-(eventtime - net.t)./net.parameters.tau)
    net.v .*= gamma
    addinput!(gamma, input, net.v)
    if net.parameters.spiketrace_timeconstant > 0
        rmul!(net.spiketrace,
              exp(-(eventtime - net.t)/net.parameters.spiketrace_timeconstant))
    end
    net.t = eventtime
end


@inline function getinext(net, input::Number, threshold::Number)
    if typeof(net.noise) == NoNoise
      v, i = findmax(net.v)
      arglog(v, input, threshold), i
    elseif typeof(net.noise) == RectLinEscapeNoise
      inext = 0; tmin = typeof(net.v[1])(Inf)
      @inbounds for i in 1:net.n_of_neurons
          if (t = getspiketimesample(net.v[i], input, net.parameters.threshold, net.t,
             net.parameters.tau, net.noise.β, net.parameters.reset, (net.v[i] == -Inf), net.noise.endofrefrperiod[i])) < tmin
              tmin = t; inext = i
          end
      end
      return tmin, inext
    end
end
@inline function getvmin(v, input, threshold, range)
    inext = 0; vmin = typeof(v[1])(Inf)
    @inbounds for i in range
        if (vi = arglog(v[i], input, threshold, i)) < vmin
            vmin = vi; inext = i
        end
    end
    return vmin, inext
end
@inline function getvminparallel(v, th, inp)
    nthreads = Threads.nthreads()
    vmin = Inf*ones(nthreads); imin = zeros(Int64, nthreads)
    N = div(length(v), nthreads)
    Threads.@threads for i in 1:nthreads
        if i == nthreads
            range = (i-1)*N + 1:length(v)
        else
            range = (i-1)*N + 1:i * N
        end
        vmin[i], imin[i] = getvmin(v, th, inp, range)
    end
    i0 = findmin(vmin)[2]
    return vmin[i0], imin[i0]
end
@inline function getinext(net, input, threshold)
    if typeof(net.noise) == NoNoise
        return getvminparallel(net.v, input, threshold)
    elseif typeof(net.noise) == RectLinEscapeNoise
      inext = 0; tmin = typeof(net.v[1])(Inf)
      @inbounds for i in 1:net.n_of_neurons
          if (t = getspiketimesample(net.v[i], input[i], net.parameters.threshold[i], net.t,
            net.parameters.tau, net.noise.β, net.parameters.reset, (net.v[i] == -Inf), net.noise.endofrefrperiod[i])) < tmin
              tmin = t; inext = i
          end
      end
      return tmin, inext
    end
end

@inline abovethreshold(v, th::Number, i) = v[i] >= th
@inline abovethreshold(v, th::AbstractArray, i) = v[i] >= th[i]

function broadcastspike!(net, i)
    @inbounds @simd for post in net.posts[i]
        net.v[post[1]] += net.weights[post[2]] # * 1000./net.parameters.tau to match with LIF rate model
        if abovethreshold(net.v, net.parameters.threshold, post[1])
            push!(net.spikecandidates, post[1])
        end
    end
end

nexteventatsamemoment(net) = false
nexteventatsamemoment(net::Net) = (length(net.nextevents) > 0 && net.t == net.nextevents[1][3])
function processspikecandidates!(net)
    if nexteventatsamemoment(net) return end
    @inbounds for i in net.spikecandidates
        if abovethreshold(net.v, net.parameters.threshold, i)
            addspike!(net, net.t, i)
        end
    end
    empty!(net.spikecandidates)
end



# here input = external_input + u_reset
"""
    function integratenet!(net, input, tmax)
"""
function integratenet!(net, input, tmax)
    net.t >= tmax && return
    while updatenet!(net, input, tmax) end
end
export integratenet!

function nextevent(net, input, tmax)
    if length(net.spikecandidates) > 0
        processspikecandidates!(net)
    end
    if length(net.nextevents) > 0 && net.t == net.nextevents[1][3]
        return net.nextevents[1]
    end
    if typeof(net.noise) == NoNoise
      arglog, inext = getinext(net, input, net.parameters.threshold)
    elseif typeof(net.noise) == RectLinEscapeNoise
      tnext, inext = getinext(net, input, net.parameters.threshold)
    end
    if inext == 0
        if length(net.nextevents) == 0
            return (NoEvent, 0, tmax)
        else
            return net.nextevents[1]
        end
    else
        if typeof(net.noise) == NoNoise
          t = net.t + nextspiketime(arglog, net.parameters, inext)
        elseif typeof(net.noise) == RectLinEscapeNoise
          t = tnext
        end
        if length(net.nextevents) == 0 || t < net.nextevents[1][3]
            if t > tmax
                return (NoEvent, 0, tmax)
            else
                return (Spike, inext, t)
            end
        else
            return net.nextevents[1]
        end
    end
end

updatenet!(net, input) = updatenet!(net, input, Inf)
function updatenet!(net, input, tmax)
    updatenetevent!(net, nextevent(net, input, tmax), input, tmax)
end
function updatenetevent!(net, event, input, tmax, doshift = true)
    if event[1] == NoEvent && event[3] == tmax || event[3] > tmax
        updatevars!(net, tmax, input)
        return false
    elseif net.t < event[3]
        updatevars!(net, event[3], input)
    end
    if event[1] == NoEvent
        return true
    elseif event[1] == Spike
        addspike!(net, event[3], event[2])
        return true
    else
        if event[1] == SpikeInsert
            broadcastspike!(net, event[2])
            updateweights!(net, event[2])
        else
            net.v[event[2]] = getreset(net.parameters.reset, event[2])
        end
        if doshift; popfirst!(net.nextevents); end
        return true
    end
end


end
