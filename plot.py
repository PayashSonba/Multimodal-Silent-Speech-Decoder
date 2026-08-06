import mne
import matplotlib.pyplot as plt
import os

for root, dirs, files in os.walk(r"D:\Payash\ds007808-download"):
    for file in files:
        if file.endswith(".edf"):
            print(os.path.join(root, file))

raw = mne.io.read_raw_edf(
    r"D:\Payash\ds007808-download\sub-02\ses-20241126\eeg\sub-02_ses-20241126_task-listening_acq-pangolin_run-03_eeg.edf",
    preload=True
)

print(raw)

for ch in raw.ch_names:
    print(ch)


# Plot first 10 channels for 10 seconds
raw.plot(duration=300, n_channels=10, scalings='auto', title='EEG Data', show=True, block=True)
# Get data and times for the first channel
data, times = raw[:10]

# Plot the first channel data
plt.figure(figsize=(15,8))
for i in range(10):
    plt.plot(times, data[i] + i*0.002, label=raw.ch_names[i])
plt.plot(times, data.T)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.title("EEG Channels Data")
plt.legend(loc ='upper right')
plt.show()


# Plot the sensor locations
raw.plot_sensors(show_names=True)

# Plot the power spectral density (PSD) of the data
raw.compute_psd().plot()

# Plot the events 
events, event_id = mne.events_from_annotations(raw)

mne.viz.plot_events(events, sfreq=raw.info['sfreq'])

# Create epochs from the raw data
events, event_id = mne.events_from_annotations(raw)

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=-0.2,
    tmax=1.0,
    preload=True
)

epochs.plot()
plt.show()

# Create an evoked object by averaging the epochs
evoked = epochs.average()

evoked.plot()
plt.show()
# Plot the topomap of the evoked response at specific time points
evoked.plot_topomap(times=[0.2,0.4,0.6])

# Plot the EMG and EOG channels
raw.pick("emg").plot()
plt.show()
raw.pick("eog").plot()
plt.show()