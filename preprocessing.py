
import pandas as pd
import numpy as np
import mne
from pathlib import Path


class Preprocessor:

    def __init__(self, run_info):

        """
        run_info is one dictionary returned by dataset.py

        Example:
        dataset[0]
        """

        self.run = run_info

        self.raw = None

    # --------------------------------------------------
    # Load EDF
    # --------------------------------------------------

    def load_raw(self):

        print(f"\nLoading EDF\n{self.run['edf']}")

        self.raw = mne.io.read_raw_edf(
            self.run["edf"],
            preload=True,
            verbose=False
        )

        return self.raw

    # --------------------------------------------------
    # Read channels.tsv
    # --------------------------------------------------

    def assign_channel_types(self):

        print("Assigning channel types...")

        channels = pd.read_csv(
            self.run["channels"],
            sep="\t"
        )

        mapping = {}

        for _, row in channels.iterrows():

            name = row["name"]
            ch_type = str(row["type"]).upper()

            if ch_type == "EEG":
                mapping[name] = "eeg"

            elif ch_type == "EMG":
                mapping[name] = "emg"

            elif ch_type == "EOG":
                mapping[name] = "eog"

            elif ch_type == "TRIG":
                mapping[name] = "stim"

            else:
                mapping[name] = "misc"

        self.raw.set_channel_types(mapping)

    # --------------------------------------------------
    # Store channel groups
    # --------------------------------------------------

    def extract_channel_groups(self):

        self.eeg_picks = mne.pick_types(
            self.raw.info,
            eeg=True
        )

        self.emg_picks = mne.pick_types(
            self.raw.info,
            emg=True
        )

        self.eog_picks = mne.pick_types(
            self.raw.info,
            eog=True
        )

        self.stim_picks = mne.pick_types(
            self.raw.info,
            stim=True
        )

        self.misc_picks = mne.pick_types(
            self.raw.info,
            misc=True
        )

        print("\nChannel Groups")

        print(f"EEG  : {len(self.eeg_picks)}")
        print(f"EMG  : {len(self.emg_picks)}")
        print(f"EOG  : {len(self.eog_picks)}")
        print(f"Stim : {len(self.stim_picks)}")
        print(f"Misc : {len(self.misc_picks)}")

    def verify_channels(self):

        assert len(self.eeg_picks) == 128, \
            "Unexpected number of EEG channels"

        assert len(self.emg_picks) == 4, \
            "Unexpected number of EMG channels"

        assert len(self.eog_picks) == 2, \
            "Unexpected number of EOG channels"

        assert len(self.stim_picks) == 1, \
            "Unexpected number of Trigger channels"

    # --------------------------------------------------
    # Quality Control
    # --------------------------------------------------

    def quality_control(self):

        print("\n==========================")
        print("QUALITY REPORT")
        print("==========================")

        sfreq = self.raw.info["sfreq"]

        duration = self.raw.times[-1]

        print(f"Subject      : {self.run['subject']}")
        print(f"Session      : {self.run['session']}")
        print(f"Run          : {self.run['run']}")

        print(f"Sampling Hz  : {sfreq}")

        print(f"Duration (s) : {duration:.2f}")

        print()

        print(f"EEG Channels : {len(self.eeg_picks)}")
        print(f"EMG Channels : {len(self.emg_picks)}")
        print(f"EOG Channels : {len(self.eog_picks)}")
        print(f"Stim         : {len(self.stim_picks)}")
        print(f"Misc         : {len(self.misc_picks)}")

        data = self.raw.get_data()

        print()

        print(f"Shape        : {data.shape}")

        print(f"NaN Values   : {np.isnan(data).sum()}")

        print(f"Minimum      : {data.min():.4e}")

        print(f"Maximum      : {data.max():.4e}")

        flat = []

        for i, ch in enumerate(self.raw.ch_names):

            if np.std(data[i]) < 1e-12:

                flat.append(ch)

        print(f"Flat Channels: {len(flat)}")

        if flat:

            print(flat)    

    # --------------------------------------------------
    # Notch Filter
    # --------------------------------------------------

    def notch_filter(self):

        print("Applying 50 Hz notch filter...")

        self.raw.notch_filter(
            freqs=[50, 100],
            verbose=False
    )

    # --------------------------------------------------
    # EEG Reference
    # --------------------------------------------------

    def rereference(self):

        print("Applying Common Average Reference...")

        self.raw.set_eeg_reference(
            ref_channels="average",
            projection=False,
            verbose=False
        )

    # --------------------------------------------------
    # EEG Bandpass
    # --------------------------------------------------

    def filter_eeg(self):

        eeg = mne.pick_types(
            self.raw.info,
            eeg=True
        )

        print(f"Filtering {len(eeg)} EEG channels...")

        self.raw.filter(
            l_freq=2,
            h_freq=120,
            picks=eeg,
            verbose=False
        )

    # --------------------------------------------------
    # EMG Bandpass
    # --------------------------------------------------

    def filter_emg(self):

        emg = mne.pick_types(
            self.raw.info,
            emg=True
        )

        if len(emg):

            print(f"Filtering {len(emg)} EMG channels...")

            self.raw.filter(
                l_freq=20,
                h_freq=120,
                picks=emg,
                verbose=False
            )

    # --------------------------------------------------
    # EOG Bandpass
    # --------------------------------------------------

    def filter_eog(self):

        eog = mne.pick_types(
            self.raw.info,
            eog=True
        )

        if len(eog):

            print(f"Filtering {len(eog)} EOG channels...")

            self.raw.filter(
                l_freq=1,
                h_freq=20,
                picks=eog,
                verbose=False
            )

    # --------------------------------------------------
    # Resample
    # --------------------------------------------------

    def resample(self):

        print("Resampling to 240 Hz...")

        self.raw.resample(
            sfreq=240,
            verbose=False
        )

    # --------------------------------------------------
    # Full pipeline
    # --------------------------------------------------

    def process(self):


            self.load_raw()

            self.assign_channel_types()

            self.extract_channel_groups()

            self.verify_channels()

            self.quality_control()

            self.notch_filter()

            self.filter_eeg()

            self.filter_emg()

            self.filter_eog()

            self.rereference()

            self.resample()

            self.quality_control()

            self.metadata = {

                "sampling_rate": self.raw.info["sfreq"],

                "duration": self.raw.times[-1],

                "eeg_channels": len(self.eeg_picks),

                "emg_channels": len(self.emg_picks),

                "eog_channels": len(self.eog_picks),

                "stim_channels": len(self.stim_picks)
            }

            return {

                "raw": self.raw,

                "eeg_picks": self.eeg_picks,

                "emg_picks": self.emg_picks,

                "eog_picks": self.eog_picks,

                "stim_picks": self.stim_picks,

                "metadata": self.metadata
            }
                



if __name__ == "__main__":

    from dataset import OpenNeuroDataset

    DATASET_PATH = r"D:\Payash\ds007808-download"

    dataset = OpenNeuroDataset(DATASET_PATH)

    run = dataset[0]

    processor = Preprocessor(run)

    processed = processor.process()

    raw = processed["raw"]

    print("\n===================================")

    print(raw)

    print(raw.info)

    print("\nMetadata")

    print(processed["metadata"])
