
dt = 0.1
t_end = 1e3 # ms
in_current = 2.

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
using TimerOutputs, CPUTime, BenchmarkTools

include("./eventbasedlifintegrator_defs/eventbasedlifintegrator_defs.jl")

network_julia, spike_monitor_julia, time_elapsed_wallclock_julia, time_elapsed_cpu_julia =
    simulate_LIF_neuron(10 * in_current; # should actually be voltage (R = 1 Ω)
                        simulation_time = t_end,
                        doEuler = false, dt = dt,
                        v_rest=V_REST,
                        v_reset=V_RESET,
                        firing_threshold=FIRING_THRESHOLD,
                        membrane_resistance=MEMBRANE_RESISTANCE,
                        membrane_time_scale=MEMBRANE_TIME_SCALE,
                        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD)

spike_times_julia = [spike[2] for spike in spike_monitor_julia.spikes]

###############################################################################
## brian2 using C++ (cpp_standalone) and exact integration
###############################################################################

using PyCall

py"""
import sys
import brian2 as b2

set_device('cpp_standalone', build_on_run=False)

sys.path.append('/Users/Bernd/Documents/PhD/Drafts/draft_mnistbenchmarks/BioPlausibleShallowDeepLearning/scripts/LIF_integrator_benchmarks/brian2_defs/')
from brian2_defs import simulate_LIF_neuron

spike_monitor_brian2, time_elapsed_wallclock_brian2, time_elapsed_cpu_brian2, max_profiling_time = simulate_LIF_neuron($in_current * b2.namp,
                        simulation_time = $t_end * b2.ms,
                        dt=$dt,
                        v_rest=$V_REST * b2.mV,
                        v_reset=$V_RESET * b2.mV,
                        firing_threshold=$FIRING_THRESHOLD * b2.mV,
                        membrane_resistance=$MEMBRANE_RESISTANCE * b2.Mohm,
                        membrane_time_scale=$MEMBRANE_TIME_SCALE * b2.ms,
                        abs_refractory_period=$ABSOLUTE_REFRACTORY_PERIOD * b2.ms)
"""
spike_monitor_brian2, time_elapsed_wallclock_brian2, time_elapsed_cpu_brian2, max_profiling_time =
    py"spike_monitor_brian2", py"time_elapsed_wallclock_brian2", py"time_elapsed_cpu_brian2", py"max_profiling_time"
spike_times_brian2 = 1e3 .* spike_monitor_brian2.spike_trains()[0]

###############################################################################

n_spikes_julia =length(spike_times_julia)
n_spikes_brian2 = length(spike_times_brian2)
n_spikes = [n_spikes_julia, n_spikes_brian2]
less_spikes = findmin(n_spikes)

print(string("Time elapsed julia (wallclock, cpu) [ms]: ", 1e3 .* [time_elapsed_wallclock_julia, time_elapsed_cpu_julia],"\n"))
print(string("Time elapsed brian2 (wallclock, cpu, max_of_profiling)[ms]: ", 1e3 .* [time_elapsed_wallclock_brian2, time_elapsed_cpu_brian2, max_profiling_time],"\n"))
print("\n")
print(string("#spikes julia: ", n_spikes_julia,"\n"))
print(string("#spikes brian2: ", n_spikes_brian2,"\n"))
print(string("relative spike count discrepancy: ", abs(n_spikes_julia - n_spikes_brian2) * 100 ./ n_spikes_julia, " %"))
print("\n")
print(string("relative abs. error in spike times (Σ |errors| / #spikes * 1/ΔT = Σ |errors| / sim. time) between spike times: \n",
    sum(abs.(spike_times_julia[1:less_spikes[1]] .- spike_times_brian2[1:less_spikes[1]])) * 100 ./ t_end, " %"))
print("\n")
