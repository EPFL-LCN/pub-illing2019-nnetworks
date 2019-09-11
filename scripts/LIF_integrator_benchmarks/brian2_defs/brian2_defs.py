import brian2 as b2
from brian2 import NeuronGroup, Synapses
from brian2.monitors import SpikeMonitor
from random import sample
from numpy import random
from timeit import default_timer as timer
import time

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
        refractory=abs_refractory_period, method="exact") # "euler" / "exact"
    neuron.v = v_rest  # set initial value

    # monitoring membrane potential of neuron and injecting current
    spike_monitor = b2.SpikeMonitor(neuron)

    b2.defaultclock.dt = dt * b2.ms

    network = b2.core.network.Network(b2.core.magic.collect())
    # run the simulation
    start_wallclock = time.time()
    start_cpu = time.clock() # timer()
    network.run(simulation_time, profile=True)
    end_cpu = time.clock() # timer()
    end_wallclock = time.time()
    time_elapsed_wallclock = end_wallclock - start_wallclock
    time_elapsed_cpu = end_cpu - start_cpu

    print("\n")
    print("brian2 profiling summary (listed by time consumption):\n")
    print(b2.profiling_summary())
    return spike_monitor, time_elapsed_wallclock, time_elapsed_cpu, network.get_profiling_info()[0][1]

def simulate_balanced_network(
        N_Excit=5000,
        N_Inhib=None,
        N_extern=N_POISSON_INPUT,
        connection_probability=CONNECTION_PROBABILITY_EPSILON,
        w0=SYNAPTIC_WEIGHT_W0,
        g=RELATIVE_INHIBITORY_STRENGTH_G,
        synaptic_delay=SYNAPTIC_DELAY,
        poisson_input_rate=POISSON_INPUT_RATE,
        w_external=None,
        v_rest=V_REST,
        v_reset=V_RESET,
        firing_threshold=FIRING_THRESHOLD,
        membrane_time_scale=MEMBRANE_TIME_SCALE,
        abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD,
        monitored_subset_size=100,
        random_vm_init=False,
        sim_time=100.*b2.ms):
    """
    Fully parametrized implementation of a sparsely connected network of LIF neurons (Brunel 2000)

    Args:
        N_Excit (int): Size of the excitatory popluation
        N_Inhib (int): optional. Size of the inhibitory population.
            If not set (=None), N_Inhib is set to N_excit/4.
        N_extern (int): optional. Number of presynaptic excitatory poisson neurons. Note: if set to a value,
            this number does NOT depend on N_Excit and NOT depend on connection_probability (this is different
            from the book and paper. Only if N_extern is set to 'None', then N_extern is computed as
            N_Excit*connection_probability.
        connection_probability (float): probability to connect to any of the (N_Excit+N_Inhib) neurons
            CE = connection_probability*N_Excit
            CI = connection_probability*N_Inhib
            Cexternal = N_extern
        w0 (float): Synaptic strength J
        g (float): relative importance of inhibition. J_exc = w0. J_inhib = -g*w0
        synaptic_delay (Quantity): Delay between presynaptic spike and postsynaptic increase of v_m
        poisson_input_rate (Quantity): Poisson rate of the external population
        w_external (float): optional. Synaptic weight of the excitatory external poisson neurons onto all
            neurons in the network. Default is None, in that case w_external is set to w0, which is the
            standard value in the book and in the paper Brunel2000.
            The purpose of this parameter is to see the effect of external input in the
            absence of network feedback(setting w0 to 0mV and w_external>0).
        v_rest (Quantity): Resting potential
        v_reset (Quantity): Reset potential
        firing_threshold (Quantity): Spike threshold
        membrane_time_scale (Quantity): tau_m
        abs_refractory_period (Quantity): absolute refractory period, tau_ref
        monitored_subset_size (int): nr of neurons for which a VoltageMonitor is recording Vm
        random_vm_init (bool): if true, the membrane voltage of each neuron is initialized with a
            random value drawn from Uniform(v_rest, firing_threshold)
        sim_time (Quantity): Simulation time

    Returns:
        (rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons)
        PopulationRateMonitor: Rate Monitor
        SpikeMonitor: SpikeMonitor for ALL (N_Excit+N_Inhib) neurons
        StateMonitor: membrane voltage for a selected subset of neurons
        list: index of monitored neurons. length = monitored_subset_size
    """
    if N_Inhib is None:
        N_Inhib = int(N_Excit/4)
    if N_extern is None:
        N_extern = int(N_Excit*connection_probability)
    if w_external is None:
        w_external = w0

    J_excit = w0
    J_inhib = -g*w0

    lif_dynamics = """
    dv/dt = -(v-v_rest) / membrane_time_scale : volt (unless refractory)"""

    network = NeuronGroup(
        N_Excit+N_Inhib, model=lif_dynamics,
        threshold="v>firing_threshold", reset="v=v_reset", refractory=abs_refractory_period,
        method="linear") # "exact"?
    if random_vm_init:
        network.v = random.uniform(v_rest/b2.mV, high=firing_threshold/b2.mV, size=(N_Excit+N_Inhib))*b2.mV
    else:
        network.v = v_rest
    excitatory_population = network[:N_Excit]
    inhibitory_population = network[N_Excit:]

    exc_synapses = Synapses(excitatory_population, target=network, on_pre="v += J_excit", delay=synaptic_delay)
    exc_synapses.connect(p=connection_probability)

    inhib_synapses = Synapses(inhibitory_population, target=network, on_pre="v += J_inhib", delay=synaptic_delay)
    inhib_synapses.connect(p=connection_probability)

    external_poisson_input = PoissonInput(target=network, target_var="v", N=N_extern,
                                          rate=poisson_input_rate, weight=w_external)

    # collect data of a subset of neurons:
    monitored_subset_size = min(monitored_subset_size, (N_Excit+N_Inhib))
    idx_monitored_neurons = sample(range(N_Excit+N_Inhib), monitored_subset_size)
    rate_monitor = PopulationRateMonitor(network)
    # record= some_list is not supported? :-(
    spike_monitor = SpikeMonitor(network, record=idx_monitored_neurons)

    b2.run(sim_time)
    return spike_monitor
