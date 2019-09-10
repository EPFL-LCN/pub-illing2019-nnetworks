
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
    a = Int(deepcopy(eval(network.parameters.reset)))
    input = input_current + a # network.parameters.reset # This causes HUGE slowdown?!
    time_elapsed = @timed integratenet!(network, input, simulation_time)
    return network, network.recorder, time_elapsed[2]
end
