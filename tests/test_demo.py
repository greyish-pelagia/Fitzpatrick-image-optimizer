from pathlib import Path

import pandas as pd

from fitzpatrick_optimizer.demo import create_demo_dataset


def test_create_demo_dataset_writes_images_and_labels(tmp_path):
    output_dir = tmp_path / "demo"

    csv_path = create_demo_dataset(output_dir, count=3)

    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert set(df.columns) == {
        "training_image",
        "ground_truth_image",
        "Fitzpatrick scale",
    }
    for path in df["training_image"].tolist() + df["ground_truth_image"].tolist():
        assert Path(path).exists()
