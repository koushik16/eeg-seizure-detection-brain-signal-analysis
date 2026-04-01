from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"

# Patients to include
PATIENT_IDS = [
    "chb01", "chb02", "chb03", "chb05", "chb06",
    "chb07", "chb08", "chb09", "chb10", "chb11",
    "chb12", "chb13", "chb14", "chb15", "chb16",
    "chb17", "chb18", "chb20", "chb21", "chb22",
    "chb23", "chb24"
]

# Signal settings
L_FREQ = 0.5
H_FREQ = 40.0
NOTCH_FREQ = 60.0

# Windowing settings
WINDOW_SIZE_SEC = 5
STRIDE_SEC = 2
OVERLAP_LABEL_THRESHOLD = 0.2  # fraction of window overlapping seizure

# Standard bipolar montage channels to keep
TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]

# Output files
WINDOWS_FILE = OUTPUT_ROOT / "X_windows.npy"
LABELS_FILE = OUTPUT_ROOT / "y_labels.npy"
META_FILE = OUTPUT_ROOT / "metadata.csv"