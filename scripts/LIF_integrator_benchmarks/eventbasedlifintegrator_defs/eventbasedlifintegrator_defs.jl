
function simulate_LIF_neuron(input_current;
                                simulation_time = 5,
                                dt = dt,
                                v_rest=-70, # * b2.mV
                                v_reset=-65, # * b2.mV,
                                firing_threshold=-50, # * b2.mV,
                                membrane_resistance=10., # * b2.Mohm,
                                membrane_time_scale=8., # * b2.ms,
                                abs_refractory_period=2.0, # * b2.ms,
                                doEuler = false)

    params = LIFParams(; tau = membrane_time_scale, reset = v_reset,
                            threshold = firing_threshold, delay = dt,
                            refractory_period = abs_refractory_period)
    network = FeedForwardNet([1], params;
                                recorder = AllSpikesRecorder(),
                                plasticityrule = NoPlasticity)

    doEuler && (network = EulerNet(network; Δt = dt))

    network.v[1] = v_rest

    input = input_current + v_reset # network.parameters.reset # This causes HUGE slowdown?!

    start_wallclock = time_ns()
    start_cpu = CPUtime_us()

    integratenet!(network, input, simulation_time)

    end_cpu = CPUtime_us()
    end_wallclock = time_ns()
    time_elapsed_wallclock = (end_wallclock - start_wallclock) / 1e9
    time_elapsed_cpu = (end_cpu - start_cpu) / 1e6

    return network, network.recorder, time_elapsed_wallclock, time_elapsed_cpu
end

function simulate_balanced_network(input_current;
                                n_of_neurons = 10^3,
                                J = 0.2,
                                g = 5.0,
                                simulation_time = 5,
                                dt = dt,
                                v_rest=-70, # * b2.mV
                                v_reset=-65, # * b2.mV,
                                firing_threshold=-50, # * b2.mV,
                                membrane_resistance=10., # * b2.Mohm,
                                membrane_time_scale=8., # * b2.ms,
                                abs_refractory_period=2.0, # * b2.ms,
                                doEuler = false)
    params = LIFParams(; tau = membrane_time_scale, reset = v_reset,
                            threshold = firing_threshold, delay = dt,
                            refractory_period = abs_refractory_period)
    network = BalancedNet(n_of_neurons, J, g; parameters = params)

    doEuler && (network = EulerNet(network; Δt = dt))
    start_wallclock = time_ns()
    start_cpu = CPUtime_us()

    integratenet!(network, input_current, simulation_time)

    end_cpu = CPUtime_us()
    end_wallclock = time_ns()
    time_elapsed_wallclock = (end_wallclock - start_wallclock) / 1e9
    time_elapsed_cpu = (end_cpu - start_cpu) / 1e6

    return network, network.recorder, time_elapsed_wallclock, time_elapsed_cpu
end
