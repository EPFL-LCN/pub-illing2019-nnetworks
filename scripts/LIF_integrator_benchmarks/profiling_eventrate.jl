# event based vs euler depending on event (i.e. spike) rate

using Pkg; Pkg.activate("./../../core/"); Pkg.instantiate()
push!(LOAD_PATH, string(pwd(),"./../../core/src/lifintegrator/core/"))
using EventBasedIntegrator, LinearAlgebra, ProgressMeter
using TimerOutputs, CPUTime, BenchmarkTools
using HDF5, PyPlot

dts = [0.01, 0.1] # ms
t_end = 1e3 # ms
in_current = 1.51 .+ collect(0:2:14) # 8

V_REST = -65. # * b2.mV
V_RESET = -65. # * b2.mV
FIRING_THRESHOLD = -50. # * b2.mV
MEMBRANE_RESISTANCE = 10. # * b2.Mohm
MEMBRANE_TIME_SCALE = 10. # * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 0. # * b2.ms

include("./eventbasedlifintegrator_defs/eventbasedlifintegrator_defs.jl")

function single_neuron(in_current; dt = 0.1, doEuler = false)
    network_julia, spike_monitor_julia, profiling_time_julia =
    simulate_LIF_neuron(10 * in_current; # should actually be voltage (R = 1 Ω)
                        simulation_time = t_end,
                        doEuler = doEuler,
                        dt = dt,
                        v_rest=V_REST,
                        v_reset=V_RESET,
                        firing_threshold=FIRING_THRESHOLD,
                        membrane_resistance=MEMBRANE_RESISTANCE,
                        membrane_time_scale=MEMBRANE_TIME_SCALE,
                        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD)

    spike_times_julia = [spike[2] for spike in spike_monitor_julia.spikes]
    return profiling_time_julia, length(spike_times_julia)
end

## eventbased

ts_eventbased = zeros(length(in_current))
ns_eventbased = zeros(length(in_current))
for j in 1:length(in_current)
        ts_eventbased[j], ns_eventbased[j] = single_neuron(in_current[j]; doEuler = false)
end


## Euler
ts_euler = zeros(length(dts), length(in_current))
ns_euler = zeros(length(dts), length(in_current))
for i in 1:length(dts)
    for j in 1:length(in_current)
        ts_euler[i,j], ns_euler[i,j] = single_neuron(in_current[j]; dt = dts[i], doEuler = true)
    end
end

fid = h5open("./data/profiling_eventrate.h5", "w")
write(fid, "ts_eventbased", ts_eventbased)
write(fid, "ns_eventbased", ns_eventbased)
write(fid, "ts_euler", ts_euler)
write(fid, "ns_euler", ns_euler)
write(fid, "dts", dts)
close(fid)
#
# figure()
# plot(ns_eventbased, ts_eventbased, label = "eventbased")
# plot(ns_euler[1,:], ts_euler[1,:], label = string("euler, dt=",dts[1])
# plot(ns_euler[2,:], ts_euler[2,:], label = string("euler, dt=",dts[2])
# xlabel("# spikes (during 1 s)")
# ylabel("profiling time [s]")
# legend()
#
# savefig("./figures/profiling_eventrate.pdf")
