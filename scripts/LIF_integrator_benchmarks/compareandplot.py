import numpy as np
import pyspike as spk
from pyspike import SpikeTrain
import matplotlib.pyplot as plt
import h5py

def get_isi_distance(spike_train_1, spike_train_2):
    return spk.isi_profile(spike_train_1, spike_train_2).avrg()

def get_spike_distance(spike_train_1, spike_train_2):
    return spk.spike_profile(spike_train_1, spike_train_2).avrg()

##############################################################################
## single neuron

t_end = 1000. # ms

dts = np.array(list(h5py.File('./data/spike_trains_single_neuron_julia.h5')['dts']))
ts_julia = np.array(list(h5py.File('./data/spike_trains_single_neuron_julia.h5')['profiling_times']))
ts_julia_euler = np.array(list(h5py.File('./data/spike_trains_single_neuron_julia_euler.h5')['profiling_times']))
ts_brian2 = np.array(list(h5py.File('./data/spike_trains_single_neuron_brian2.h5')['profiling_times']))

distances_brian2 = np.zeros(dts.size) # ISI between brian2 and event-based julia
distances_euler = np.zeros(dts.size) # ISI between euler julia and event-based julia

distances_spike_brian2 = np.zeros(dts.size) # ISI between brian2 and event-based julia
distances_spike_euler = np.zeros(dts.size) # ISI between euler julia and event-based julia

spike_count_error_brian2 = np.zeros(dts.size) # rel. distance between spike counts
spike_count_error_julia = np.zeros(dts.size)

for i in range(0,dts.size):
    spike_times_julia = list(h5py.File('./data/spike_trains_single_neuron_julia.h5')['spike_train_'+str(dts[i])])
    spike_train_julia = SpikeTrain(np.array(spike_times_julia), [0.0, t_end])

    spike_times_julia_euler = list(h5py.File('./data/spike_trains_single_neuron_julia_euler.h5')['spike_train_'+str(dts[i])])
    spike_train_julia_euler = SpikeTrain(np.array(spike_times_julia_euler), [0.0, t_end])

    spike_times_brian2 = list(h5py.File('./data/spike_trains_single_neuron_brian2.h5')['spike_train_'+str(dts[i])])
    spike_train_brian2 = SpikeTrain(np.array(spike_times_brian2), [0.0, t_end])

    distances_brian2[i] = get_isi_distance(spike_train_julia, spike_train_brian2)
    distances_euler[i] = get_isi_distance(spike_train_julia, spike_train_julia_euler)
    distances_spike_brian2[i] = get_spike_distance(spike_train_julia, spike_train_brian2)
    distances_spike_euler[i] = get_spike_distance(spike_train_julia, spike_train_julia_euler)
    spike_count_error_brian2[i] = float(abs(len(spike_times_brian2) - len(spike_times_julia))) / len(spike_times_julia) * 100
    spike_count_error_julia[i] = float(abs(len(spike_times_julia_euler) - len(spike_times_julia))) / len(spike_times_julia) * 100

print('Number of spikes per second (julia event-based): '+str(np.array(list(h5py.File('./data/spike_trains_single_neuron_julia.h5')['spike_train_'+str(dts[i])])).size))

plt.figure()
plt.loglog(dts, ts_brian2, label = 'brian2')
plt.loglog(dts, ts_julia_euler, label = 'julia')
plt.loglog(dts, ts_julia, label = 'julia event based')
plt.xlabel('dt for integration [ms]')
plt.ylabel('profiling time [ms]')
plt.legend()
plt.title('Single LIF neuron, T = 1 s, approx. 280 Hz')
plt.savefig('./figures/singleneuron_t_vs_dt.pdf')

plt.figure()
plt.semilogx(dts, spike_count_error_brian2, label = 'brian2')
plt.semilogx(dts, spike_count_error_julia, label = 'julia euler')
plt.xlabel('dt for integration [ms]')
plt.ylabel('rel. spike count error [%]')
plt.legend()
plt.title('Single LIF neuron, T = 1 s, approx. 280 Hz')
plt.savefig('./figures/singleneuron_spikecounterror.pdf')

plt.figure()
plt.loglog(dts, distances_brian2, label = 'brian2')
plt.loglog(dts, distances_euler, label = 'julia euler')
plt.xlabel('dt for integration [ms]')
plt.ylabel('ISI distance')
plt.legend()
plt.title('Single LIF neuron, T = 1 s, approx. 280 Hz')
plt.savefig('./figures/singleneuron_ISI_vs_dt.pdf')

plt.figure()
plt.loglog(dts, distances_spike_brian2, label = 'brian2')
plt.loglog(dts, distances_spike_euler, label = 'julia euler')
plt.xlabel('dt for integration [ms]')
plt.ylabel('SPIKE distance')
plt.legend()
plt.title('Single LIF neuron, T = 1 s, approx. 280 Hz')
plt.savefig('./figures/singleneuron_SPIKE_vs_dt.pdf')

plt.figure()
plt.loglog(distances_brian2, ts_brian2 / np.mean(ts_julia), label = 'brian2')
plt.loglog(distances_euler, ts_julia_euler / np.mean(ts_julia), label = 'julia euler')
plt.xlabel('ISI distance')
plt.ylabel('profiling time / profiling time (julia_eventbased)')
plt.legend()
plt.title('Single LIF neuron, T = 1 s, approx. 280 Hz')
plt.savefig('./figures/singleneuron_ISI_vs_t.pdf')

plt.figure()
plt.loglog(distances_spike_brian2, ts_brian2 / np.mean(ts_julia), label = 'brian2')
plt.loglog(distances_spike_euler, ts_julia_euler / np.mean(ts_julia), label = 'julia euler')
plt.xlabel('SPIKE distance')
plt.ylabel('profiling time / profiling time (julia_eventbased)')
plt.legend()
plt.title('Single LIF neuron, T = 1 s, approx. 280 Hz')
plt.savefig('./figures/singleneuron_SPIKE_vs_t.pdf')

##############################################################################
## Events rate profiling

dts_eventrate = np.array(list(h5py.File('./data/profiling_eventrate.h5')['dts']))
ts_eventbased = np.array(list(h5py.File('./data/profiling_eventrate.h5')['ts_eventbased']))
ns_eventbased = np.array(list(h5py.File('./data/profiling_eventrate.h5')['ns_eventbased']))
ts_euler = np.array(list(h5py.File('./data/profiling_eventrate.h5')['ts_euler']))
ns_euler = np.array(list(h5py.File('./data/profiling_eventrate.h5')['ns_euler']))

plt.figure()
plt.semilogy(ns_eventbased, ts_eventbased, label = 'eventbased')
plt.semilogy(ns_euler[:,0], ts_euler[:,0], label = 'euler, dt='+str(dts_eventrate[0]))
plt.semilogy(ns_euler[:,1], ts_euler[:,1], label = 'euler, dt='+str(dts_eventrate[1]))
plt.xlabel('spike frequency[Hz]')
plt.ylabel('profiling time [s]')
plt.legend()
plt.title('Single LIF neuron, T = 1 s')
plt.savefig('./figures/singleneuron_eventrateprofiling.pdf')

##############################################################################
## balanced net

ns_of_neurons = np.array(list(h5py.File('./data/balancednet_julia.h5')['ns_of_neurons']))
ts_balanced_julia = np.array(list(h5py.File('./data/balancednet_julia.h5')['profiling_times']))
ts_balanced_julia_euler = np.array(list(h5py.File('./data/balancednet_julia_euler.h5')['profiling_times']))
ts_balanced_brian2 = np.array(list(h5py.File('./data/balancednet_brian2.h5')['profiling_times']))

plt.figure()
plt.loglog(ns_of_neurons, ts_balanced_julia, label = 'julia eventbased')
plt.loglog(ns_of_neurons, ts_balanced_julia_euler, label = 'julia euler')
plt.loglog(ns_of_neurons, ts_balanced_brian2, label = 'brian2')
plt.legend()
plt.xlabel('number of neurons')
plt.ylabel('profiling times [s]')
plt.title('Balanced Net (Brunel 2000), dt = 0.01, T = 1 s')
plt.savefig('./figures/balancednet.pdf')

plt.show()
