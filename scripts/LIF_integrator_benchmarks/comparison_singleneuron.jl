
dts = 2.0 .^ collect(-12:2:2) # collect(-9:1:3)
t_end = 1e3 # ms
in_current = 5. # 8

V_REST = -65. # * b2.mV
V_RESET = -65. # * b2.mV
FIRING_THRESHOLD = -50. # * b2.mV
MEMBRANE_RESISTANCE = 10. # * b2.Mohm
MEMBRANE_TIME_SCALE = 10. # * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 0. # * b2.ms

#doEuler_julia = false

###############################################################################
## EventBasedIntegrator
###############################################################################

using Pkg; Pkg.activate("./../../core/"); Pkg.instantiate()
push!(LOAD_PATH, string(pwd(),"./../../core/src/lifintegrator/core/"))
using EventBasedIntegrator, LinearAlgebra, ProgressMeter
using TimerOutputs, CPUTime, BenchmarkTools
using HDF5

include("./eventbasedlifintegrator_defs/eventbasedlifintegrator_defs.jl")
function single_neuron_julia(dt; doEuler = false)
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
    return profiling_time_julia, spike_times_julia
end

# Event-based

fid = h5open("./data/spike_trains_single_neuron_julia.h5", "w")
write(fid, "dts", dts)
profiling_times_julia = zeros(length(dts))
for i in 1:length(dts)
    t, spts = single_neuron_julia(dts[i]; doEuler = false)
    write(fid, string("spike_train_",dts[i]), spts)
    profiling_times_julia[i] = t
end
write(fid, "profiling_times", profiling_times_julia)
close(fid)

# Euler-forward

fid = h5open("./data/spike_trains_single_neuron_julia_euler.h5", "w")
write(fid, "dts", dts)
profiling_times_julia = zeros(length(dts))
for i in 1:length(dts)
    t, spts = single_neuron_julia(dts[i]; doEuler = true)
    write(fid, string("spike_train_",dts[i]), spts)
    profiling_times_julia[i] = t
end
write(fid, "profiling_times", profiling_times_julia)
close(fid)

###############################################################################
## brian2 using C++ (cpp_standalone) and exact integration
###############################################################################

using PyCall

py"""
import sys
import brian2 as b2

sys.path.append('/Users/Bernd/Documents/PhD/Drafts/draft_mnistbenchmarks/BioPlausibleShallowDeepLearning/scripts/LIF_integrator_benchmarks/brian2_defs/')
from brian2_defs import simulate_LIF_neuron
"""

function single_neuron_brian2(dt; first_init = false)
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

    spike_monitor_brian2, profiling_summary = simulate_LIF_neuron($in_current * b2.namp,
                            simulation_time = $t_end * b2.ms,
                            dt=$dt,
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

    spike_times_brian2 = 1e3 .* spike_monitor_brian2.spike_trains()[0]
    return profiling_time_brian2, spike_times_brian2
end

fid = h5open("./data/spike_trains_single_neuron_brian2.h5", "w")
write(fid, "dts", dts)
profiling_times_brian2 = zeros(length(dts))
for i in 1:length(dts)
    t, spts = single_neuron_brian2(dts[i]; first_init = (i == 1))
    write(fid, string("spike_train_",dts[i]), spts)
    profiling_times_brian2[i] = t
end
write(fid, "profiling_times", profiling_times_brian2)
close(fid)

###############################################################################

# n_spikes_julia =length(spike_times_julia)
# n_spikes_brian2 = length(spike_times_brian2)
# n_spikes = [n_spikes_julia, n_spikes_brian2]
# less_spikes = findmin(n_spikes)
#
# print(string("Time elapsed brian2 (profiling)[ms]: ", 1e3 .* profiling_time_brian2,"\n"))
# print(string("Time elapsed julia (median_of_profiling) [ms]: ", 1e3 .* profiling_time_julia,"\n"))
# print("\n")
# print(string("#spikes brian2: ", n_spikes_brian2,"\n"))
# print(string("#spikes julia: ", n_spikes_julia,"\n"))
# print(string("relative spike count discrepancy: ", abs(n_spikes_julia - n_spikes_brian2) * 100 ./ n_spikes_julia, " %"))
# print("\n")
# print(string("relative abs. error in spike times (Σ |errors| / #spikes * 1/ΔT = Σ |errors| / sim. time) between spike times: \n",
#     sum(abs.(spike_times_julia[1:less_spikes[1]] .- spike_times_brian2[1:less_spikes[1]])) * 100 ./ t_end, " %"))
# print("\n")
