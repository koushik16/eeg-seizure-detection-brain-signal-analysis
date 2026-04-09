from pathlib import Path

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "predictions"

# -----------------------------
# Input / signal settings
# -----------------------------
EXPECTED_SFREQ = 256
WINDOW_SEC = 5
STEP_SEC = 2.5

WINDOW_SAMPLES = int(EXPECTED_SFREQ * WINDOW_SEC)
STEP_SAMPLES = int(EXPECTED_SFREQ * STEP_SEC)

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
NOTCH_FREQ = 60.0
APPLY_NOTCH = True

# -----------------------------
# Channel order
# -----------------------------
EXPECTED_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]

N_CHANNELS = len(EXPECTED_CHANNELS)

# -----------------------------
# Normalization stats
# -----------------------------
TRAIN_MEAN_PATH = ARTIFACTS_DIR / "train_channel_mean.npy"
TRAIN_STD_PATH = ARTIFACTS_DIR / "train_channel_std.npy"

# -----------------------------
# Model
# -----------------------------
MODEL_PATH = ARTIFACTS_DIR / "best_eeg_cnnlstm.pth"
CLASSIFICATION_THRESHOLD = 0.75
MIN_CONSECUTIVE_POSITIVE_WINDOWS = 5

# -----------------------------
# Runtime
# -----------------------------
BATCH_SIZE = 64

# -----------------------------
# Save-intermediate names
# -----------------------------
PREPROCESSED_SIGNAL_NAME = "preprocessed_signal.npy"
WINDOWS_NAME = "windows.npy"
NORMALIZED_WINDOWS_NAME = "normalized_windows.npy"
WINDOW_PROBS_NAME = "window_probabilities.npy"
SUMMARY_NAME = "summary.json"
CHANNELS_NAME = "channels_used.json"