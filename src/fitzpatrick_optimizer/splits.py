import numpy as np
import pandas as pd


def assign_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError(
            "train_fraction and val_fraction must leave a positive test fraction"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    train_end = max(1, int(len(indices) * train_fraction))
    val_end = max(train_end + 1, int(len(indices) * (train_fraction + val_fraction)))
    val_end = min(val_end, len(indices) - 1)

    split_values = np.empty(len(df), dtype=object)
    split_values[indices[:train_end]] = "train"
    split_values[indices[train_end:val_end]] = "val"
    split_values[indices[val_end:]] = "test"

    result = df.copy()
    result["split"] = split_values
    return result
