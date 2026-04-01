from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    DATA_ROOT, PATIENT_IDS, TARGET_CHANNELS,
    L_FREQ, H_FREQ, NOTCH_FREQ,
    WINDOW_SIZE_SEC, STRIDE_SEC, OVERLAP_LABEL_THRESHOLD,
    WINDOWS_FILE, LABELS_FILE, META_FILE
)
from src.summary_parser import parse_patient_summary
from src.edf_loader import load_and_prepare_edf
from src.windowing import create_windows
from src.saver import save_outputs


def run_preprocessing():
    all_X = []
    all_y = []
    all_meta = []

    for patient_id in PATIENT_IDS:
        patient_dir = DATA_ROOT / patient_id
        summary_file = patient_dir / f"{patient_id}-summary.txt"

        if not patient_dir.exists():
            print(f"[WARNING] Missing patient folder: {patient_dir}")
            continue

        if not summary_file.exists():
            print(f"[WARNING] Missing summary file: {summary_file}")
            continue

        seizure_map = parse_patient_summary(summary_file)

        edf_files = sorted(patient_dir.glob("*.edf"))
        print(f"\nProcessing {patient_id}: {len(edf_files)} EDF files")

        for edf_path in edf_files:
            edf_name = edf_path.name
            seizure_intervals = seizure_map.get(edf_name, [])

            try:
                data, sfreq = load_and_prepare_edf(
                    file_path=edf_path,
                    target_channels=TARGET_CHANNELS,
                    l_freq=L_FREQ,
                    h_freq=H_FREQ,
                    notch_freq=NOTCH_FREQ
                )

                X_file, y_file, meta_file = create_windows(
                    data=data,
                    sfreq=sfreq,
                    seizure_intervals=seizure_intervals,
                    window_size_sec=WINDOW_SIZE_SEC,
                    stride_sec=STRIDE_SEC,
                    overlap_threshold=OVERLAP_LABEL_THRESHOLD
                )

                for row in meta_file:
                    row["patient_id"] = patient_id
                    row["file_name"] = edf_name

                all_X.append(X_file)
                all_y.append(y_file)
                all_meta.extend(meta_file)

                print(
                    f"  {edf_name}: windows={len(y_file)}, "
                    f"seizure_windows={int(y_file.sum())}, "
                    f"non_seizure_windows={len(y_file) - int(y_file.sum())}"
                )

            except Exception as e:
                print(f"[ERROR] Failed on {edf_name}: {e}")

    if not all_X:
        raise RuntimeError("No windows were created. Check data paths and channel consistency.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    save_outputs(
        X=X,
        y=y,
        metadata=all_meta,
        windows_file=WINDOWS_FILE,
        labels_file=LABELS_FILE,
        meta_file=META_FILE
    )

    print("\nPreprocessing complete")
    print(f"Saved X: {WINDOWS_FILE}")
    print(f"Saved y: {LABELS_FILE}")
    print(f"Saved metadata: {META_FILE}")
    print(f"Total windows: {len(y)}")
    print(f"Positive windows: {int(y.sum())}")
    print(f"Negative windows: {len(y) - int(y.sum())}")