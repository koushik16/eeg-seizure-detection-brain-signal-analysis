from __future__ import annotations
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLASSIFICATION_THRESHOLD, MIN_CONSECUTIVE_POSITIVE_WINDOWS
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSIFICATION_THRESHOLD
from src.preprocessing.loader import load_edf
from src.preprocessing.channels import reorder_signal_channels, validate_required_channels
from src.preprocessing.filters import apply_filters
from src.preprocessing.windowing import create_windows
from src.preprocessing.normalize import load_normalization_stats, apply_channelwise_zscore
from src.inference.model_loader import load_model
from src.inference.predictor import predict_window_probabilities
from src.inference.aggregation import summarize_predictions
from src.inference.saver import save_intermediate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG seizure prediction from EDF file")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input EDF file",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save preprocessed signal, windows, probabilities, and summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input EDF file not found: {input_path}")

    # 1. Load EDF
    data, sfreq, raw_channel_names = load_edf(input_path)

    # 2. Validate and reorder channels
    validate_required_channels(raw_channel_names)
    data, ordered_channels = reorder_signal_channels(data, raw_channel_names)

    # 3. Filter
    filtered_data = apply_filters(data, sfreq)

    # 4. Window
    windows, window_ranges = create_windows(filtered_data)

    # 5. Normalize
    mean, std = load_normalization_stats()
    normalized_windows = apply_channelwise_zscore(windows, mean, std)

    # 6. Load model
    model, device = load_model()

    # 7. Predict
    window_probs = predict_window_probabilities(model, normalized_windows, device)

    # 8. Aggregate
    summary = summarize_predictions(
        window_probs=window_probs,
        window_ranges=window_ranges,
        threshold=CLASSIFICATION_THRESHOLD,
        sfreq=sfreq,
        min_consecutive_positive_windows=MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    )

    # 9. Print result

    print(f"Input file: {input_path}")
    print(f"Windows processed: {summary['num_windows']}")
    print(f"Max seizure probability: {summary['seizure_probability']:.6f}")
    print(f"Mean window probability: {summary['mean_window_probability']:.6f}")
    print(f"Positive windows: {summary['num_positive_windows']}/{summary['num_windows']}")
    print(f"Prediction: {summary['prediction']}")


    if summary["prediction"] == "YES":
        print("\nConsecutive positive time frames used for YES decision:")
        for run in summary["decision_runs"]:
            print(
                f"  Windows {run['start_window_index']} to {run['end_window_index']} | "
                f"{run['start_sec']:.2f}s to {run['end_sec']:.2f}s | "
                f"count={run['num_windows']} | "
                f"max_p={run['max_probability_in_run']:.4f} | "
                f"mean_p={run['mean_probability_in_run']:.4f}"
            )

 #   if summary["num_positive_windows"] > 0:
 #       print("\nPositive window time frames:")
 #       for window in summary["positive_windows"]:
 #           print(
 #               f"  Window {window['window_index']}: "
 #               f"{window['start_sec']:.2f}s to {window['end_sec']:.2f}s "
 #               f"(p={window['probability']:.4f})"
#            )

 #   if summary["num_positive_windows"] > 0:
 #       first_hit = summary["positive_windows"][0]
 #       print(
 #           f"First positive window: "
 #           f"{first_hit['start_sec']:.2f}s to {first_hit['end_sec']:.2f}s "
 #           f"(p={first_hit['probability']:.4f})"
#        )

    # 10. Optional save
    if args.save_intermediate:
        out_dir = save_intermediate_outputs(
            input_path=input_path,
            channels_used=ordered_channels,
            preprocessed_signal=filtered_data,
            windows=windows,
            normalized_windows=normalized_windows,
            window_probabilities=window_probs,
            summary=summary,
        )
        print(f"Saved intermediate outputs to: {out_dir}")


if __name__ == "__main__":
    main()