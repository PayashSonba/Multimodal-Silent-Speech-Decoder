
from pathlib import Path
from torch.utils.data import Dataset
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


class OpenNeuroDataset(Dataset):

    def __init__(self, dataset_root, subjects=None):

        """
        Parameters
        ----------
        dataset_root : str or Path
            Root directory of OpenNeuro dataset.

        subjects : list or None
            Example:
                ["sub-01"]
                ["sub-01","sub-02"]
                None -> Load all subjects
        """

        self.dataset_root = Path(dataset_root)
        self.subjects = subjects
        self.runs = []

        self.scan_dataset()

    # Scan Dataset

    def scan_dataset(self):

        pattern = "sub-*/ses-*/eeg/*_eeg.edf"

        edf_files = sorted(
            self.dataset_root.glob(pattern)
        )

        logging.info(f"Found {len(edf_files)} EDF recordings")

        for edf in edf_files:

            subject = edf.parts[-4]

            if self.subjects is not None:

                if subject not in self.subjects:
                    continue

            session = edf.parts[-3]

            filename = edf.stem.replace("_eeg", "")

            eeg_folder = edf.parent
            beh_folder = eeg_folder.parent / "beh"

            run = {

                "subject": subject,

                "session": session,

                "run": filename.split("_")[-1],

                "edf": edf,

                "events": eeg_folder / (filename + "_events.tsv"),

                "channels": eeg_folder / (filename + "_channels.tsv"),

                "eeg_json": eeg_folder / (filename + "_eeg.json"),

                "wav": beh_folder /
                (
                    filename.replace("_acq-pangolin", "")
                    + "_recording-vocal_beh.wav"
                ),

                "audio_json": beh_folder /
                (
                    filename.replace("_acq-pangolin", "")
                    + "_recording-vocal_beh.json"
                )
            }

            # Verify files exist

            missing = []

            for key in [
                "edf",
                "events",
                "channels",
                "eeg_json",
                "wav",
                "audio_json"
            ]:

                if not run[key].exists():
                    missing.append(key)

             # Check required files only
            required = [
                    "edf",
                    "events",
                    "channels"
                ]

            optional = [
                    "wav",
                    "audio_json",
                    "eeg_json"
                ]

            required_missing = [
                        key for key in required
                        if not run[key].exists()
                    ]
        
            if required_missing:
                        logging.warning(
                            f"Skipping {filename} "
                            f"(missing {required_missing})"
                        )
                        continue
        
                    # Optional files
            for key in optional:
                        if not run[key].exists():
                            run[key] = None

            self.runs.append(run)

        logging.info(
            f"Valid runs : {len(self.runs)}"
        )

    # Required by PyTorch

    def __len__(self):

        return len(self.runs)

    def __getitem__(self, idx):

        return self.runs[idx]

    # Summary

    def summary(self):

        subjects = Counter()

        sessions = {}

        for run in self.runs:

            subjects[run["subject"]] += 1

            sessions.setdefault(
                run["subject"],
                set()
            )

            sessions[run["subject"]].add(
                run["session"]
            )

        print("\n==========================")
        print("DATASET SUMMARY")
        print("==========================")

        print(f"Total Runs : {len(self)}")

        for subject in sorted(subjects):

            print(

                f"{subject}"

                f" | Runs: {subjects[subject]}"

                f" | Sessions: {len(sessions[subject])}"

            )


# --------------------------------------------------------
# Test
# --------------------------------------------------------

if __name__ == "__main__":

    DATASET_PATH = r"D:\Payash\ds007808-download"

    dataset = OpenNeuroDataset(
        DATASET_PATH,
        subjects=None          # Change to ["sub-01"] for debugging
    )

    dataset.summary()

    print("\nFirst Sample\n")

    sample = dataset[0]

    for key, value in sample.items():

        print(f"{key:12s}: {value}")