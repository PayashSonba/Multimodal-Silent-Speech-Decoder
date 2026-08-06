#   Import libraries

import numpy as np
from pathlib import Path
import scipy.signal as signal

# Create feature extraction class

class FeatureExtractor:

    def __init__(self, trial_file):

        self.trial_file = Path(trial_file)

        self.data = np.load(
            self.trial_file,
            allow_pickle=True
        )

        self.eeg = self.data["eeg"]

        self.emg = self.data["emg"]

        self.eog = self.data["eog"]

        self.label = str(self.data["label"])

        self.trial_type = str(self.data["trial_type"])

        self.sfreq = float(self.data["sampling_rate"])

    # Create EEG Frequency Band Extraction Method
    BANDS = {

        "delta": (1,4),

        "theta": (4,8),

        "alpha": (8,13),

        "beta": (13,30),

        "gamma": (30,70)

    }        

    # Bandpass filter function
    def bandpass_filter(
        self,
        signal_data,
        low,
        high
    ):

        nyquist = self.sfreq / 2

        b, a = signal.butter(
            4,
            [low/nyquist, high/nyquist],
            btype="band"
        )

        return signal.filtfilt(
            b,
            a,
            signal_data,
            axis=1
        )

    # Multi-Band EEG feature extraction function
    def extract_multiband(self):

        self.multiband = {}

        print("\nExtracting EEG Bands")

        for name, (low, high) in self.BANDS.items():

            filtered = self.bandpass_filter(
                self.eeg,
                low,
                high
            )

            self.multiband[name] = filtered

            print(
                f"{name:<6}",
                filtered.shape
            )

    # Differentail Entropy Feature Extraction Function
    def differential_entropy(self, x):

        variance = np.var(
            x,
            axis=1
        )

        variance = np.maximum(
            variance,
            1e-10
        )

        return 0.5 * np.log(
            2*np.pi*np.e*variance
        )            

    # Extract Differential Entropy Features from every bands
    def extract_entropy(self):

        entropy = []

        print("\nComputing Differential Entropy")

        for band in self.multiband.values():

            de = self.differential_entropy(
                band
            )

            entropy.append(de)

        self.entropy = np.concatenate(
            entropy
        )

        print(
            "Entropy shape:",
            self.entropy.shape
        )

    # Pipeline function to run the feature extraction methods    
    def run(self):

        self.extract_multiband()

        self.extract_entropy()

        return {
            "entropy_features": self.entropy,
            "label": self.label,
            "trial_type": self.trial_type,
            "sampling_rate": self.sfreq
        }

# Test the feature extraction class
if __name__ == "__main__":

    trial = Path(
       "outputs",
        "segmented",
        "sub-01",
        "ses-20230829",
        "run-01",
        "trial_0000.npz"
    )

    extractor = FeatureExtractor(trial)

    extractor.run()            