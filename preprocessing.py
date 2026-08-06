import pandas as pd
import mne


# Load EDF

edf_file = r"D:\Payash\ds007808-download\sub-01\ses-20230829\eeg\sub-01_ses-20230829_task-speechopen_acq-pangolin_run-06_eeg.edf"

raw = mne.io.read_raw_edf(edf_file, preload=False)

# Load channel TSV

tsv_file = r"D:\Payash\ds007808-download\sub-01\ses-20230829\eeg\sub-01_ses-20230829_task-speechopen_acq-pangolin_run-06_channels.tsv"

# Load channel information from TSV file and set channel types in the raw object

channels = pd.read_csv(tsv_file, sep="\t")

print(channels.head())
channel_types = {}

# Separate them into EEG, EMG, EOG, and TRIG channels
for _, row in channels.iterrows():

    ch_name = row["name"]
    ch_type = row["type"].strip().upper()

    if ch_type == "EEG":
        channel_types[ch_name] = "eeg"

    elif ch_type == "EMG":
        channel_types[ch_name] = "emg"

    elif ch_type == "EOG":
        channel_types[ch_name] = "eog"

    elif ch_type == "TRIG":
        channel_types[ch_name] = "stim"

    else:
        channel_types[ch_name] = "misc"

raw.set_channel_types(channel_types)



# PREPROCESSING
# Make separate copies for EEG, EMG and EOG
raw_eeg = raw.copy().pick("eeg")
raw_emg = raw.copy().pick("emg")
raw_eog = raw.copy().pick("eog")

# EEG PREPROCESSING
print("\nPreprocessing EEG...")

# 1. Remove 50 Hz power-line noise
raw_eeg.notch_filter(freqs=50)

# 2. Common Average Reference (CAR)
raw_eeg.set_eeg_reference("average")

# 3. Band-pass filter (2–120 Hz)
raw_eeg.filter(l_freq=2, h_freq=120)

# 4. Resample to 240 Hz
raw_eeg.resample(240)

print("EEG preprocessing completed.")

# EMG PREPROCESSING
print("\nPreprocessing EMG...")

# Remove power-line noise
raw_emg.notch_filter(freqs=50)

# Band-pass filter
raw_emg.filter(l_freq=2, h_freq=120)

# Resample
raw_emg.resample(240)

print("EMG preprocessing completed.")

# Plot EEG Channels
eeg_picks = mne.pick_types(raw_eeg.info, eeg=True)

print(f"Number of EEG channels: {len(eeg_picks)}")

raw_eeg.plot(
    picks=eeg_picks,
    duration=10,
    n_channels=20,
    scalings='auto',
    title='EEG Signals',
    show=True,
    block=True
)

# Plot EMG Channels
emg_picks = mne.pick_types(raw_emg.info, emg=True)

print(f"Number of EMG channels: {len(emg_picks)}")

raw_emg.plot(
    picks=emg_picks,
    duration=10,
    n_channels=len(emg_picks),
    scalings='auto',
    title='EMG Signals',
    show=True,
    block=True
)


from collections import Counter

types = [raw.get_channel_types(picks=ch)[0] for ch in raw.ch_names]

print("\nChannel Type Summary")
print(Counter(types))

print("\nEMG Channels")

for ch in raw.ch_names:
    if raw.get_channel_types(picks=ch)[0] == "emg":
        print(ch)
    
print("\nEOG Channels")

for ch in raw.ch_names:
    if raw.get_channel_types(picks=ch)[0] == "eog":
        print(ch)


print("\nTrigger Channel")

for ch in raw.ch_names:
    if raw.get_channel_types(picks=ch)[0] == "stim":
        print(ch)     

