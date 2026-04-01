import re
from pathlib import Path


def parse_patient_summary(summary_path: Path):
    """
    Parse a CHB-MIT patient summary file and return:
    {
        "chb01_03.edf": [(2996, 3036)],
        "chb01_04.edf": [],
        ...
    }
    """
    seizure_map = {}

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    current_file = None
    current_intervals = []

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        file_match = re.match(r"File Name:\s*(.+)", line)
        if file_match:
            if current_file is not None:
                seizure_map[current_file] = current_intervals
            current_file = file_match.group(1).strip()
            current_intervals = []
            continue

        start_match = re.match(r"Seizure Start Time:\s*(\d+)\s*seconds", line)
        end_match = re.match(r"Seizure End Time:\s*(\d+)\s*seconds", line)

        if start_match:
            start_time = int(start_match.group(1))
            current_intervals.append([start_time, None])

        if end_match:
            end_time = int(end_match.group(1))
            if current_intervals and current_intervals[-1][1] is None:
                current_intervals[-1][1] = end_time

    if current_file is not None:
        seizure_map[current_file] = current_intervals

    # convert lists to tuples and clean malformed entries
    cleaned = {}
    for fname, intervals in seizure_map.items():
        valid = []
        for start, end in intervals:
            if start is not None and end is not None and end > start:
                valid.append((start, end))
        cleaned[fname] = valid

    return cleaned