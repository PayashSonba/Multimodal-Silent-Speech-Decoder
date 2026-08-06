# import mne
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import Counter

# # =====================================================
# # LOAD EDF FILE
# # =====================================================

# edf_path = r"D:\Payash\ds007808-download\sub-02\ses-20241126\eeg\sub-02_ses-20241126_task-listening_acq-pangolin_run-01_eeg.edf"

# raw = mne.io.read_raw_edf(edf_path, preload=True)

# # =====================================================
# # BASIC INFORMATION
# # =====================================================

# print("="*60)
# print("RAW INFORMATION")
# print("="*60)

# print(raw)
# print(raw.info)

# print("\nTotal Channels :", len(raw.ch_names))
# print("Recording Duration :", raw.times[-1], "seconds")
# print("Sampling Frequency :", raw.info["sfreq"], "Hz")

# print("\nChannel Types")
# print(Counter(raw.get_channel_types()))

# print("\nFirst 20 Channel Names")
# print(raw.ch_names)

# # =====================================================
# # RAW EEG VIEWER
# # =====================================================

# raw.plot(
#     duration=60,
#     n_channels=10,
#     scalings="auto",
#     title="Raw EEG Signal",
#     block=True
# )

# # =====================================================
# # PLOT FIRST 10 EEG CHANNELS
# # =====================================================

# data, times = raw[:10]

# plt.figure(figsize=(18,8))

# offset = 0.002

# for i in range(10):
#     plt.plot(
#         times,
#         data[i] + i*offset,
#         label=raw.ch_names[i]
#     )

# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")
# plt.title("First 10 EEG Channels")
# plt.legend(loc="upper right")
# plt.grid(True)

# plt.show()

# # =====================================================
# # POWER SPECTRAL DENSITY
# # =====================================================

# raw.compute_psd().plot()

# # =====================================================
# # CHECK FOR EMG / EOG
# # =====================================================

# types = Counter(raw.get_channel_types())

# print("\nAvailable Signal Types")
# print(types)

# if "emg" in types:

#     emg = raw.copy().pick("emg")
#     emg.plot(title="EMG Signals")

# else:
#     print("\nNo EMG channels found in this EDF file.")

# if "eog" in types:

#     eog = raw.copy().pick("eog")
#     eog.plot(title="EOG Signals")

# else:
#     print("\nNo EOG channels found in this EDF file.")

# # =====================================================
# # EVENTS
# # =====================================================

# try:

#     events, event_id = mne.events_from_annotations(raw)

#     print("\nEvent IDs")
#     print(event_id)

#     if len(events) > 0:

#         mne.viz.plot_events(
#             events,
#             sfreq=raw.info["sfreq"]
#         )

#         # =============================================
#         # CREATE EPOCHS
#         # =============================================

#         epochs = mne.Epochs(
#             raw,
#             events,
#             event_id,
#             tmin=-0.2,
#             tmax=1.0,
#             preload=True
#         )

#         epochs.plot()

#         # =============================================
#         # AVERAGE EEG
#         # =============================================

#         evoked = epochs.average()

#         evoked.plot()

#         # =============================================
#         # TOPOGRAPHIC MAP
#         # =============================================

#         try:
#             evoked.plot_topomap(times=[0.2,0.4,0.6])

#         except Exception as e:
#             print("\nCannot plot topomap.")
#             print(e)

#     else:

#         print("\nNo events found.")

# except Exception as e:

#     print("\nNo annotations/events available.")
#     print(e)

# print("\nProgram Finished Successfully.")
import os

folder = r"D:\Payash\ds007808-download\sub-01\ses-20230829\beh"

for f in sorted(os.listdir(folder)):
    print(f)
import pandas as pd

events = pd.read_csv(
    r"D:\Payash\ds007808-download\sub-01\ses-20230829\eeg\sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_events.tsv",
    sep="\t"
)

print(events.columns)
print(events.head(10))    

from pathlib import Path

root = Path(r"D:\Payash\ds007808-download")

for f in sorted(root.glob("**/*events.tsv")):
    print(f.name)   