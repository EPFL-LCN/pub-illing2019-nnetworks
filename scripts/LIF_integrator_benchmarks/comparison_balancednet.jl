dt = .01
t_end = 1e3 # ms
in_current = 0.

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
using CPUTime

include("./eventbasedlifintegrator_defs/eventbasedlifintegrator_defs.jl")

network_julia, spike_monitor_julia, time_elapsed_wallclock_julia, time_elapsed_cpu_julia =
    simulate_balanced_network(10 * in_current; # should actually be voltage (R = 1 Ω)
                        simulation_time = t_end,
                        doEuler = false, dt = dt,
                        v_rest=V_REST,
                        v_reset=V_RESET,
                        firing_threshold=FIRING_THRESHOLD,
                        membrane_resistance=MEMBRANE_RESISTANCE,
                        membrane_time_scale=MEMBRANE_TIME_SCALE,
                        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD)

spike_times_julia = [spike[2] for spike in spike_monitor_julia.spikes]
# get the (2) weights for init of python network?

###############################################################################
## brian2 using C++ (cpp_standalone) and exact integration
###############################################################################

using PyCall

py"""
import sys
import brian2 as b2

set_device('cpp_standalone', build_on_run=False)

sys.path.append('/Users/Bernd/Documents/PhD/Drafts/draft_mnistbenchmarks/BioPlausibleShallowDeepLearning/scripts/LIF_integrator_benchmarks/brian2_defs/')
from brian2_defs import simulate_balanced_network

spike_monitor_brian2 = simulate_balanced_network()
"""

spike_monitor_brian2 = py"spike_monitor_brian2"
spike_times_brian2 = 1e3 .* spike_monitor_brian2.spike_trains()[0]

###############################################################################

print(string("Time elapsed julia (wallclock, cpu) [ms]: ", 1e3 .* [time_elapsed_wallclock_julia, time_elapsed_cpu_julia],"\n"))
