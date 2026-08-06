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

    



    #-------------------------------------------------------------------------------------
    # EMG Feature Extraction 
    # RMS Feature Extraction Function
    def compute_rms(self, signal):

        return np.sqrt(np.mean(signal ** 2))

    # MAV(Mean Absolute Value) Feature Extraction Function
    def compute_mav(self, signal):

        return np.mean(np.abs(signal))

    # Waveform Length Feature Extraction Function
    def compute_waveform_length(self, signal):

        return np.sum(np.abs(np.diff(signal)))

    # Extract EMG Features Function
    def extract_emg_features(self):

        print("\nExtracting EMG Features")

        features = []

        for channel in self.emg:

            rms = self.compute_rms(channel)

            mav = self.compute_mav(channel)

            wl = self.compute_waveform_length(channel)

            features.extend([rms, mav, wl])

        self.emg_features = np.array(features)

        print("EMG Feature Shape:", self.emg_features.shape)



    #-------------------------------------------------------------------------------------   

    # EOG Feature Extraction
    # MEAN , STD , Max Absolute Amplitude Feature Extraction Function
    def extract_eog_features(self):

        print("\nExtracting EOG Features")

        features = []

        for channel in self.eog:

            mean = np.mean(channel)

            std = np.std(channel)

            peak = np.max(np.abs(channel))

            features.extend([mean, std, peak])

        self.eog_features = np.array(features)

        print("EOG Feature Shape:", self.eog_features.shape)

    # Pipeline function to run the feature extraction methods    
    # Pipeline function to run the feature extraction methods
    def run(self):

        self.extract_multiband()

        self.extract_entropy()

        self.extract_emg_features()

        self.extract_eog_features()

        # ---------------------------------------
        # Feature Fusion
        # ---------------------------------------

        self.fused_features = np.concatenate(

            [
                self.entropy,
                self.emg_features,
                self.eog_features
            ]

        )

        print("Fused Feature Shape:", self.fused_features.shape)

        return {

            "entropy_features": self.entropy,

            "emg_features": self.emg_features,

            "eog_features": self.eog_features,

            "fused_features": self.fused_features,

            "label": self.label,

            "trial_type": self.trial_type,

            "sampling_rate": self.sfreq

        }

    # Save the extracted features to a .npz file
    def save_features(self, output_file):

        np.savez_compressed(

            output_file,

            entropy_features=self.entropy,

            emg_features=self.emg_features,

            eog_features=self.eog_features,

            fused_features=self.fused_features,

            label=self.label,

            trial_type=self.trial_type,

            sampling_rate=self.sfreq

        )

        print(f"\nSaved features to:\n{output_file}")
        
# Test the feature extraction class
if __name__ == "__main__":

    from pathlib import Path

    ROOT = Path("outputs") / "segmented"

    trial_files = sorted(ROOT.rglob("trial_*.npz"))

    print(f"\nFound {len(trial_files)} segmented trials")

    processed = 0
    failed = 0

    for i, trial in enumerate(trial_files, start=1):

        print("\n" + "=" * 60)
        print(f"Trial {i}/{len(trial_files)}")
        print(trial)

        try:

            extractor = FeatureExtractor(trial)

            features = extractor.run()

            relative = trial.relative_to(ROOT)

            output_file = (
                Path("outputs")
                / "features"
                / relative.parent
                / relative.name.replace("trial", "feature")
            )

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            extractor.save_features(output_file)

            processed += 1

        except Exception as e:

            failed += 1

            print("FAILED")
            print(e)

    print("\n" + "=" * 60)
    print("\n========== SUMMARY ==========")

    print("EEG :", features["entropy_features"].shape)

    print("EMG :", features["emg_features"].shape)

    print("EOG :", features["eog_features"].shape)

    print("Fused :", features["fused_features"].shape)

    print("Label :", features["label"])

    print("Trial :", features["trial_type"])