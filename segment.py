import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import Preprocessor


class Segmenter:

    def __init__(self, run_info):

        self.run = run_info

        processor = Preprocessor(run_info)

        processed = processor.process()

        self.raw = processed["raw"]

        self.eeg = processed["eeg_picks"]

        self.emg = processed["emg_picks"]

        self.eog = processed["eog_picks"]

        self.sfreq = processed["metadata"]["sampling_rate"]

    # Load events.tsv

    def load_events(self):

        self.events = pd.read_csv(
            self.run["events"],
            sep="\t"
        )

        print(f"\nLoaded {len(self.events)} events")

    # Segment trials

    def create_trials(self):

        trials = []

        for _, row in self.events.iterrows():

            onset = float(row["onset"])

            duration = float(row["duration"])

            start = int(onset * self.sfreq)

            stop = int((onset + duration) * self.sfreq)

            eeg = self.raw.get_data(
                picks=self.eeg,
                start=start,
                stop=stop
            )

            emg = self.raw.get_data(
                picks=self.emg,
                start=start,
                stop=stop
            )

            eog = self.raw.get_data(
                picks=self.eog,
                start=start,
                stop=stop
            )

            trials.append({

                "eeg": eeg,

                "emg": emg,

                "eog": eog,

                "label": row["value"],

                "trial_type": row["trial_type"],

                "onset": onset,

                "duration": duration
            })

        self.trials = trials

        print(f"Created {len(trials)} trials")

    # Save trials

    def save_trials(self):

        out_dir = (
            Path("outputs")
            / "segmented"
            / self.run["subject"]
            / self.run["session"]
            / self.run["run"]
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for i, trial in enumerate(self.trials):

            np.savez_compressed(

                out_dir / f"trial_{i:04d}.npz",

                eeg=trial["eeg"],

                emg=trial["emg"],

                eog=trial["eog"],

                label=trial["label"],

                trial_type=trial["trial_type"],

                onset=trial["onset"],

                duration=trial["duration"],

                sampling_rate=self.sfreq
            )

        print(f"Saved {len(self.trials)} trials")

    # Pipeline

    def run_pipeline(self):

        self.load_events()

        self.create_trials()

        self.save_trials()


if __name__ == "__main__":

    from dataset import OpenNeuroDataset

    DATASET = r"D:\Payash\ds007808-download"

    dataset = OpenNeuroDataset(DATASET)

    TOTAL = len(dataset)

    processed = 0
    failed = 0
    total_trials = 0

    # -------------------------------
    # CONFIGURATION
    # -------------------------------

    MODE = "subset"
    MAX_RUNS = 10

    # debug
    if MODE == "debug":
        runs = dataset[:1]

    # subset
    elif MODE == "subset":
        runs = dataset[:MAX_RUNS]

    # full
    else:
        runs = dataset

    # -------------------------------
    # PROCESS
    # -------------------------------

    for i, run in enumerate(runs, start=1):

        print("\n" + "=" * 60)
        print(f"Run {i}/{len(runs)}")
        print("=" * 60)

        print(run["subject"])
        print(run["session"])
        print(run["run"])

        try:

            segmenter = Segmenter(run)

            segmenter.run_pipeline()

            processed += 1

            total_trials += len(segmenter.trials)

        except Exception as e:

            failed += 1

            print("FAILED")

            print(e)

    print("\n")
    print("=" * 60)
    print("SEGMENTATION SUMMARY")
    print("=" * 60)

    print("Runs processed :", processed)
    print("Runs failed    :", failed)
    print("Trials created :", total_trials)