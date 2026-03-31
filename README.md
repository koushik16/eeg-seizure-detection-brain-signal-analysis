# EEG Seizure Detection and Brain Signal Analysis

This project develops an end-to-end machine learning pipeline for detecting epileptic seizures from EEG (Electroencephalogram) signals. It covers raw signal processing, temporal segmentation, feature preparation, and classification under highly imbalanced conditions.

---

## Overview

Electroencephalography (EEG) records electrical activity of the brain using electrodes placed on the scalp. These signals are inherently noisy, high-dimensional, and temporally complex. Seizure detection from EEG is a challenging time-series classification problem due to:

- Multi-channel signal structure (spatial + temporal dependencies)
- Noise and physiological artifacts
- Patient-specific variability
- Extreme class imbalance (very few seizure events)

This project builds a structured and reproducible pipeline to address these challenges.

---

## Dataset

### Source
- CHB-MIT Scalp EEG Database (PhysioNet)
- Public dataset widely used for seizure detection research

### Data Characteristics

- **Subjects:** Pediatric patients with intractable seizures
- **Recordings:** Multiple EEG sessions per subject
- **File Format:** `.edf` (European Data Format)
- **Sampling Frequency:** ~256 Hz
- **Channels:** Typically 18–23 EEG channels per recording

Each `.edf` file contains:
- Continuous multi-channel EEG signal
- Recording duration (~1 hour per file in many cases)

---

### Seizure Annotations

Seizure events are provided via summary files and/or `.seizures` annotations:

Each seizure includes:
- **Start time (seconds)**
- **End time (seconds)**

Example:
Seizure Start Time: 1732 seconds
Seizure End Time: 1772 seconds

Some files contain:
- No seizures (normal recordings)
- Multiple seizures within a single file

---

### Data Organization

Typical structure:

chb01/
├── chb01_01.edf
├── chb01_02.edf
├── …
├── chb01_15.edf
├── chb01-summary.txt

Key points:
- Summary files describe seizure intervals per recording
- Channel naming may vary slightly across files
- Some channels may have duplicates or inconsistencies

---

### Challenges in the Dataset

- **Severe Class Imbalance**
  - Seizure windows are extremely rare compared to normal activity
  - Example scale:
    - Total windows: ~500,000+
    - Seizure windows: ~2,000–3,000

- **Signal Noise**
  - Powerline interference (50/60 Hz)
  - Motion artifacts
  - Electrode noise

- **Channel Variability**
  - Slight differences in channel naming/order
  - Requires normalization across recordings

- **Temporal Complexity**
  - Seizures exhibit evolving patterns over time
  - Not always sharply defined boundaries

---

## Pipeline

### 1. Data Loading
- Load `.edf` files using MNE
- Extract signal matrix (channels × time)
- Standardize channel names and ordering

---

### 2. Signal Preprocessing
- Bandpass filter: 0.5–40 Hz (retain brain-relevant frequencies)
- Notch filter: ~60 Hz (remove powerline noise)
- Signal normalization and cleaning

---

### 3. Windowing
- Segment continuous signals into fixed windows (e.g., 5 seconds)
- Optional overlap between windows
- Assign labels:
  - **1 → seizure window**
  - **0 → non-seizure window**

---

### 4. Feature Representation
- Raw time-series windows
- Future extensions:
  - Spectrograms (time-frequency)
  - Wavelet transforms

---

### 5. Modeling

#### Baseline
- Logistic Regression

#### Planned Models
- CNN (spatial + frequency features)
- LSTM (temporal dependencies)
- CNN-LSTM hybrid models

---

### 6. Evaluation

Due to extreme class imbalance:

Metrics used:
- **Sensitivity (Recall for seizure class)** → critical metric
- **Specificity**
- **ROC-AUC**

Accuracy is monitored but not relied upon as the primary metric.

---

## Current Status

- EEG preprocessing pipeline implemented
- Window-based dataset created (~500k+ samples)
- Class imbalance quantified and analyzed
- Baseline model evaluated (Logistic Regression)
- Observed limitations in detecting minority class (seizures)

---

## Project Structure

eeg-seizure-detection-brain-signal-analysis/
│
├── data/                  # (ignored) raw and processed EEG data
├── notebooks/             # exploratory analysis and experiments
├── preprocess.py          # signal preprocessing
├── windowing.py           # window creation and labeling
├── config.json            # parameters (window size, filters, etc.)
├── requirements.txt       # dependencies
└── README.md

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/eeg-seizure-detection-brain-signal-analysis.git
cd eeg-seizure-detection-brain-signal-analysis

pip install -r requirements.txt

Usage

Run preprocessing pipeline:
python preprocess.py
python windowing.py

Or explore via notebooks:
jupyter notebook

Key Challenges
	•	Handling extreme class imbalance
	•	Maintaining signal integrity during preprocessing
	•	Capturing temporal patterns effectively
	•	Generalizing across patients

⸻

Future Work
	•	Class imbalance strategies (class weights, focal loss, resampling)
	•	Deep learning models (CNN, LSTM, Transformers)
	•	Time-frequency modeling (spectrograms, wavelets)
	•	Model interpretability (SHAP, saliency maps)
	•	Real-time seizure detection systems

⸻

Tech Stack
	•	Python
	•	MNE (EEG processing)
	•	NumPy, Pandas
	•	Scikit-learn
	•	PyTorch (planned)
	•	Matplotlib / Seaborn

⸻

Notes
	•	Raw EEG data is not included due to size constraints
	•	Dataset must be downloaded from PhysioNet:
https://physionet.org/content/chbmit/1.0.0/

⸻

Author

Koushik Reddy Parukola
Research Associate, NLP Lab — Indiana University