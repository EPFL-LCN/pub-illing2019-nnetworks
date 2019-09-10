import brian2 as b2
from brian2 import NeuronGroup, Synapses
from brian2.monitors import SpikeMonitor
from random import sample
from numpy import random
from timeit import default_timer as timer

def simulate_LIF_neuron(input_current,
                        simulation_time=5 * b2.ms,
                        dt = 0.01,
                        v_rest=-70 * b2.mV,
                        v_reset=-65 * b2.mV,
                        firing_threshold=-50 * b2.mV,
                        membrane_resistance=10. * b2.Mohm,
                        membrane_time_scale= 8. * b2.ms,
                        abs_refractory_period=2.0 * b2.ms):


    # differential equation of Leaky Integrate-and-Fire model
    # eqs = """
    # dv/dt =
    # ( -(v-v_rest) + membrane_resistance * input_current(t,i) ) / membrane_time_scale : volt (unless refractory)
    # """
    eqs = """
    dv/dt =
    ( -(v-v_rest) + membrane_resistance * input_current ) / membrane_time_scale : volt (unless refractory)
    """

    # LIF neuron using Brian2 library
    neuron = b2.NeuronGroup(
        1, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
        refractory=abs_refractory_period, method="exact") # "euler"
    neuron.v = v_rest  # set initial value

    # monitoring membrane potential of neuron and injecting current
    spike_monitor = b2.SpikeMonitor(neuron)

    b2.defaultclock.dt = dt * b2.ms

    # run the simulation
    start = timer()
    b2.run(simulation_time, profile=True)
    end = timer()
    time_elapsed = end - start

    print("\n")
    print("brian2 profiling summary (listed by time consumption):\n")
    print(b2.profiling_summary())
    return spike_monitor, time_elapsed
