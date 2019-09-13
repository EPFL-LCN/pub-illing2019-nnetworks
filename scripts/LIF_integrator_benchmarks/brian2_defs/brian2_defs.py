import brian2 as b2
from brian2 import NeuronGroup, Synapses
from brian2.monitors import SpikeMonitor
from random import sample
from numpy import random, floor
from timeit import default_timer as timer
import time
import copy

def simulate_LIF_neuron(input_current,
                        simulation_time=5. * b2.ms,
                        dt = 0.01,
                        v_rest=-70 * b2.mV,
                        v_reset=-65 * b2.mV,
                        firing_threshold=-50 * b2.mV,
                        membrane_resistance=10. * b2.Mohm,
                        membrane_time_scale= 8. * b2.ms,
                        abs_refractory_period=2.0 * b2.ms):


    b2.defaultclock.dt = dt * b2.ms

    # differential equation of Leaky Integrate-and-Fire model
    # eqs = """
    # dv/dt =
    # ( -(v-v_rest) + membrane_resistance * input_current(t,i) ) / membrane_time_scale : volt (unless refractory)
    # """
    eqs = """
    dv/dt =
    ( -(v-v_rest) + membrane_resistance * input_current ) / membrane_time_scale : volt (unless refractory)
    """

    neuron = b2.NeuronGroup(
        1, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
        refractory=abs_refractory_period, method="exact") # "euler" / "exact"
    neuron.v = v_rest  # set initial value

    network = b2.core.network.Network(neuron)
    # run before for compiling (JIT compile time out of timing)
    #network.run(simulation_time, profile=True)

    spike_monitor = b2.SpikeMonitor(neuron)
    network.add(spike_monitor)
    neuron.v = v_rest

    #start_wallclock = time.time()
    #start_cpu = time.clock() # timer()

    network.run(simulation_time, profile=True)

    #end_cpu = time.clock() # timer()
    #end_wallclock = time.time()
    #time_elapsed_wallclock = end_wallclock - start_wallclock
    #time_elapsed_cpu = end_cpu - start_cpu

    b2.device.build(directory='output', clean=True, compile=True, run=True, debug=False)

    print("\n")
    print("brian2 profiling summary (listed by time consumption):\n")
    print(b2.profiling_summary())

    return spike_monitor, network.get_profiling_info() # time_elapsed_wallclock, time_elapsed_cpu,


def simulate_balanced_network(input_current=0. * b2.namp,
        simulation_time=1000.*b2.ms,
        dt = 0.01,
        n_of_neurons=1000,
        f = 3/4,
        connection_probability=0.1,
        w0=0.1 * b2.mV,
        g=4.,
        membrane_resistance=10. * b2.Mohm,
        poisson_input_rate=13. * b2.Hz,
        v_rest=0. * b2.mV,
        v_reset=10. * b2.mV,
        firing_threshold=20. * b2.mV,
        membrane_time_scale=20. * b2.ms,
        abs_refractory_period=2.0 * b2.ms,
        monitored_subset_size=100,
        random_vm_init=False):

    b2.defaultclock.dt = dt * b2.ms

    N_Excit = int(floor(f*n_of_neurons))
    N_Inhib = int(floor((1-f)*n_of_neurons))

    J_excit = w0
    J_inhib = -g*w0

    lif_dynamics = """
    dv/dt =
    ( -(v-v_rest) + membrane_resistance * input_current ) / membrane_time_scale : volt (unless refractory)
    """

    neurons = NeuronGroup(
        N_Excit+N_Inhib, model=lif_dynamics,
        threshold="v>firing_threshold", reset="v=v_reset", refractory=abs_refractory_period,
        method="linear") # "exact"?
    if random_vm_init:
        neurons.v = random.uniform(v_rest/b2.mV, high=firing_threshold/b2.mV, size=(N_Excit+N_Inhib))*b2.mV
    else:
        neurons.v = v_rest
    excitatory_population = neurons[:N_Excit]
    inhibitory_population = neurons[N_Excit:]

    exc_synapses = Synapses(excitatory_population, target=neurons, on_pre="v += J_excit", delay=dt * b2.ms)
    exc_synapses.connect(p=connection_probability)

    inhib_synapses = Synapses(inhibitory_population, target=neurons, on_pre="v += J_inhib", delay=dt * b2.ms)
    inhib_synapses.connect(p=connection_probability)

    monitored_subset_size = min(monitored_subset_size, (N_Excit+N_Inhib))
    idx_monitored_neurons = sample(range(N_Excit+N_Inhib), monitored_subset_size)
    spike_monitor = b2.SpikeMonitor(neurons, record=idx_monitored_neurons)

    network = b2.core.network.Network(b2.core.magic.collect())

    start_wallclock = time.time()
    start_cpu = time.clock() # timer()

    network.run(simulation_time, profile=True)

    end_cpu = time.clock() # timer()
    end_wallclock = time.time()
    time_elapsed_wallclock = end_wallclock - start_wallclock
    time_elapsed_cpu = end_cpu - start_cpu

    device.build(directory='output', compile=True, run=True, debug=False)

    print("\n")
    print("brian2 profiling summary (listed by time consumption):\n")
    print(b2.profiling_summary())
    return spike_monitor, time_elapsed_wallclock, time_elapsed_cpu, network.get_profiling_info()[0][1]
