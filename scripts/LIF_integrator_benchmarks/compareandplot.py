import numpy as np
import pyspike as spk
from pyspike import SpikeTrain
import matplotlib.pyplot as plt
import h5py

def get_isi_distance(spike_train_1, spike_train_2):
    return spk.isi_profile(spike_train_1, spike_train_2).avrg()

def get_spike_distance(spike_train_1, spike_train_2):
    return spk.spike_profile(spike_train_1, spike_train_2).avrg()

t_end = 1000. # ms

dts = np.array(list(h5py.File('./data/spike_trains_single_neuron_julia.h5')['dts']))
ts_julia = np.array(list(h5py.File('./data/spike_trains_single_neuron_julia.h5')['profiling_times']))
ts_julia_euler = np.array(list(h5py.File('./data/spike_trains_single_neuron_julia_euler.h5')['profiling_times']))
ts_brian2 = np.array(list(h5py.File('./data/spike_trains_single_neuron_brian2.h5')['profiling_times']))

distances_brian2 = np.zeros(dts.size) # ISI between brian2 and event-based julia
distances_euler = np.zeros(dts.size) # ISI between euler julia and event-based julia
distances_spike_brian2 = np.zeros(dts.size) # ISI between brian2 and event-based julia
distances_spike_euler = np.zeros(dts.size) # ISI between euler julia and event-based julia
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

print('Number of spikes per second (julia event-based): '+str(np.array(list(h5py.File('./data/spike_trains_single_neuron_julia.h5')['spike_train_'+str(dts[i])])).size))

plt.figure()
plt.loglog(distances_brian2, ts_brian2 / np.mean(ts_julia), label = 'exp = brian2')
plt.loglog(distances_euler, ts_julia_euler / np.mean(ts_julia), label = 'exp = julia euler')
plt.xlabel('ISI distance')
plt.ylabel('t_exp / t_julia')
plt.legend()
plt.show()

plt.figure()
plt.loglog(distances_spike_brian2, ts_brian2 / np.mean(ts_julia), label = 'exp = brian2')
plt.loglog(distances_spike_euler, ts_julia_euler / np.mean(ts_julia), label = 'exp = julia euler')
plt.xlabel('SPIKE distance')
plt.ylabel('t_exp / t_julia')
plt.legend()
plt.show()
