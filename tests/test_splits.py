import pandas as pd

from fitzpatrick_optimizer.splits import assign_split


def test_assign_split_is_deterministic_and_complete():
    df = pd.DataFrame(
        {
            "training_image": [f"input-{i}.jpg" for i in range(20)],
            "ground_truth_image": [f"target-{i}.jpg" for i in range(20)],
            "Fitzpatrick scale": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4] * 2,
        }
    )

    first = assign_split(df, seed=7)
    second = assign_split(df, seed=7)

    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "val", "test"}
    assert len(first) == 20
