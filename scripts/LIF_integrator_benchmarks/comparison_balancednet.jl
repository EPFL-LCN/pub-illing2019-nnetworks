dt = .01
t_end = 1e3 # ms
in_current = 2.

ns_of_neurons = 10 .^ [1,2,3,4]

V_REST = -65. # * b2.mV
V_RESET = -65. # * b2.mV
FIRING_THRESHOLD = -50. # * b2.mV
MEMBRANE_RESISTANCE = 10. # * b2.Mohm
MEMBRANE_TIME_SCALE = 10. # * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 0. # * b2.ms

###############################################################################
## EventBasedIntegrator
###############################################################################

using Pkg; Pkg.activate("./../../core/"); Pkg.instantiate()
push!(LOAD_PATH, string(pwd(),"./../../core/src/lifintegrator/core/"))
using EventBasedIntegrator, LinearAlgebra, ProgressMeter
using TimerOutputs
using CPUTime, BenchmarkTools, HDF5

include("./eventbasedlifintegrator_defs/eventbasedlifintegrator_defs.jl")

function balanced_julia(n_of_neurons; doEuler = false)
    network_julia, spike_monitor_julia, profiling_time_julia =
        simulate_balanced_network(10 * in_current; # should actually be voltage (R = 1 Ω)
                            n_of_neurons = n_of_neurons,
                            simulation_time = t_end,
                            doEuler = doEuler, dt = dt,
                            v_rest=V_REST,
                            v_reset=V_RESET,
                            firing_threshold=FIRING_THRESHOLD,
                            membrane_resistance=MEMBRANE_RESISTANCE,
                            membrane_time_scale=MEMBRANE_TIME_SCALE,
                            abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD)

    spike_times_julia = [spike[2] for spike in spike_monitor_julia.spikes]
    return profiling_time_julia, spike_times_julia
end

# event based

fid = h5open("./data/balancednet_julia.h5", "w")
write(fid, "ns_of_neurons", ns_of_neurons)
profiling_times_julia = zeros(length(ns_of_neurons))
for i in 1:length(ns_of_neurons)
    t, spts = balanced_julia(ns_of_neurons[i]; doEuler = false)
    write(fid, string("spike_train_",ns_of_neurons[i]), spts)
    profiling_times_julia[i] = t
end
write(fid, "profiling_times", profiling_times_julia)
close(fid)

# euler

fid = h5open("./data/balancednet_julia_euler.h5", "w")
write(fid, "ns_of_neurons", ns_of_neurons)
profiling_times_julia_euler = zeros(length(ns_of_neurons))
for i in 1:length(ns_of_neurons)
    t, spts = balanced_julia(ns_of_neurons[i]; doEuler = true)
    write(fid, string("spike_train_",ns_of_neurons[i]), spts)
    profiling_times_julia_euler[i] = t
end
write(fid, "profiling_times", profiling_times_julia_euler)
close(fid)

# get the (2) weights for init of python network?

###############################################################################
## brian2 using C++ (cpp_standalone) and exact integration
###############################################################################

using PyCall

py"""
import sys
import brian2 as b2

sys.path.append('/Users/Bernd/Documents/PhD/Drafts/draft_mnistbenchmarks/BioPlausibleShallowDeepLearning/scripts/LIF_integrator_benchmarks/brian2_defs/')
from brian2_defs import simulate_balanced_network
"""

function balanced_brian2(n_of_neurons; first_init = false)
    py"""
    import sys
    import brian2 as b2

    if $first_init:
        b2.set_device('cpp_standalone', build_on_run=False)
    else:
        #brian2.__init__.clear_cache()
        b2.device.reinit()
        b2.device.activate()
        b2.set_device('cpp_standalone', build_on_run=False)
    spike_monitor_brian2, profiling_summary = simulate_balanced_network($in_current * b2.namp,
                            n_of_neurons = $n_of_neurons,
                            simulation_time = $t_end * b2.ms,
                            dt = $dt,
                            v_rest=$V_REST * b2.mV,
                            v_reset=$V_RESET * b2.mV,
                            firing_threshold=$FIRING_THRESHOLD * b2.mV,
                            membrane_resistance=$MEMBRANE_RESISTANCE * b2.Mohm,
                            membrane_time_scale=$MEMBRANE_TIME_SCALE * b2.ms,
                            abs_refractory_period=$ABSOLUTE_REFRACTORY_PERIOD * b2.ms)
    """
    spike_monitor_brian2, profiling_summary = py"spike_monitor_brian2", py"profiling_summary"
    profiling_times = []
    for x in profiling_summary push!(profiling_times, x[2][1]) end
    profiling_time_brian2 = sum(profiling_times)

    spike_monitor_brian2 = py"spike_monitor_brian2"
    spike_times_brian2 = 1e3 .* spike_monitor_brian2.spike_trains()[0]
    return profiling_time_brian2, spike_times_brian2
end

fid = h5open("./data/balancednet_brian2.h5", "w")
write(fid, "ns_of_neurons", ns_of_neurons)
profiling_times_brian2 = zeros(length(ns_of_neurons))
for i in 1:length(ns_of_neurons)
    t, spts = balanced_brian2(ns_of_neurons[i]; first_init = (i == 1))
    write(fid, string("spike_train_",ns_of_neurons[i]), spts)
    profiling_times_brian2[i] = t
end
write(fid, "profiling_times", profiling_times_brian2)
close(fid)

###############################################################################

#print(string("Time elapsed brian2 (profiling)[ms]: ", 1e3 .* profiling_time_brian2,"\n"))
#print(string("Time elapsed julia (median_of_profiling) [ms]: ", 1e3 .* profiling_time_julia,"\n"))
