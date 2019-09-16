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
    network_init = FeedForwardNet([1], params;
                                recorder = AllSpikesRecorder(),
                                plasticityrule = NoPlasticity)

    doEuler && (network_init = EulerNet(network_init; Δt = dt))
    network_init.v[1] = v_rest
    network = deepcopy(network_init)

    input = input_current + network_init.parameters.reset

    b = @benchmark integratenet!(x[1], x[2], x[3]) setup = (x = (deepcopy($network_init), copy($input), copy($simulation_time))) evals = 1;
    profiling_time = BenchmarkTools.median(b).time

    integratenet!(network, input, simulation_time)
    rec = deepcopy(network.recorder)
    return network, rec, profiling_time / 1e9
end

function simulate_balanced_network(input_current;
                                n_of_neurons = 10^3,
                                J = 0.2,
                                g = 5.0,
                                f = 3/4,
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
    network_init = BalancedNet(n_of_neurons, J, g; f = f, parameters = params)

    doEuler && (network_init = EulerNet(network_init; Δt = dt))
    network_init.v[:] .= v_rest
    network = deepcopy(network_init)

    input = input_current + network_init.parameters.reset

    b = @benchmark integratenet!(x[1], x[2], x[3]) setup = (x = (deepcopy($network_init), copy($input), copy($simulation_time))) evals = 1;
    profiling_time = BenchmarkTools.median(b).time

    integratenet!(network, input, simulation_time)
    rec = deepcopy(network.recorder)
    return network, rec, profiling_time / 1e9
end
