import numpy as np
import pandas as pd
from pathlib import Path


def save_outputs(X, y, metadata, windows_file, labels_file, meta_file):
    Path(windows_file).parent.mkdir(parents=True, exist_ok=True)

    np.save(windows_file, X)
    np.save(labels_file, y)

    df = pd.DataFrame(metadata)
    df.to_csv(meta_file, index=False)