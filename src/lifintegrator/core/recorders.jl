"""
    NoRecorder()

Records nothing.
"""
struct NoRecorder end
export NoRecorder
recordspike!(recorder, net, spike) = Nothing # Void

mutable struct SpikeRecorder
    neuronindices::Array{Int64, 1}
    timeintervals::Array{Tuple{Float64, Float64}, 1}
    intervalcounter::Int64
    spikes::Array{Tuple{Int64, Float64}, 1}
end
export SpikeRecorder
"""
    SpikeRecorder(neuronindices, timeintervals)

Records all spikes in `neuronindices` and `timeintervals`.
`neuronindices` is an array of indices.
`timeintervals` should be given as `[(start1, end1); (start2, end2); ...]`.
"""
SpikeRecorder(neuronindices, timeintervals) =
    SpikeRecorder(neuronindices, timeintervals, 1, Tuple{Int64, Float64}[], Tuple{Int64, Float64, Float64}[])
function recordspike!(recorder::SpikeRecorder, net, spike)
    !(spike[1] in recorder.neuronindices) && return
    recorder.intervalcounter > length(recorder.timeintervals) && return
    while spike[2] > recorder.timeintervals[recorder.intervalcounter][2]
        recorder.intervalcounter += 1
        recorder.intervalcounter > length(recorder.timeintervals) && return
    end
    if spike[2] > recorder.timeintervals[recorder.intervalcounter][1]
        push!(recorder.spikes, spike)
    end
    return
end
struct AllSpikesRecorder
    spikes::Array{Tuple{Int64, Float64}, 1}
end
export AllSpikesRecorder
"""
    AllSpikesRecorder() = AllSpikesRecorder(Tuple{Int64, Float64}[])

Records all spikes.
"""
AllSpikesRecorder() = AllSpikesRecorder(Tuple{Int64, Float64}[])
recordspike!(recorder::AllSpikesRecorder, net, spike) = push!(recorder.spikes, spike)


struct AllTracesRecorder
    spikes::Array{Tuple{Int64, Float64, Float64}, 1}
end
export AllTracesRecorder
"""
    AllTracesRecorder() = AllTracesRecorder(Tuple{Int64, Float64, Float64}[])

Records all traces at respective spike times.
"""
AllTracesRecorder() = AllTracesRecorder(Tuple{Int64, Float64, Float64}[])
recordspike!(recorder::AllTracesRecorder, net, spike) = push!(recorder.spikes, (spike[1], spike[2], net.spiketrace[spike[1]]))


struct AllStateRecorder
    spiketimes::Array{Float64, 1}
    spikes::Array{Tuple{Int64, Float64}, 1}
    traces::Array{Array{Float64, 1}, 1}
    membranepotentials::Array{Array{Float64, 1}, 1}
    weights::Array{Array{Float64, 1}, 1}
end
export AllStateRecorder
"""
AllStateRecorder() = AllStateRecorder(Array{Float64, 1}(), Tuple{Int64, Float64}[], Array{Array{Float64, 1}, 1}(),
                                      Array{Array{Float64, 1}, 1}(), Array{Array{Float64, 1}, 1}())

Records the state (spiketime, spike (neuronindex, time), traces, membrane potential, weights) at respective spike times.
"""
AllStateRecorder() = AllStateRecorder(Array{Float64, 1}(), Tuple{Int64, Float64}[], Array{Array{Float64, 1}, 1}(),
                                      Array{Array{Float64, 1}, 1}(), Array{Array{Float64, 1}, 1}())
function recordspike!(recorder::AllStateRecorder, net, spike)
  push!(recorder.spiketimes, deepcopy(net.t))
  push!(recorder.spikes, (spike[1], spike[2]))
  push!(recorder.traces, deepcopy(net.spiketrace))
  push!(recorder.membranepotentials, deepcopy(net.v))
  push!(recorder.weights, deepcopy(net.weights))
end
