import importlib.metadata
import subprocess
import sys


def test_package_imports():
    import fitzpatrick_optimizer

    assert fitzpatrick_optimizer.__version__ == "0.1.0"


def test_distribution_metadata_has_real_description():
    metadata = importlib.metadata.metadata("fitzpatrick-image-optimizer")

    assert metadata["Name"] == "fitzpatrick-image-optimizer"
    assert "experimental" in metadata["Summary"].lower()


def test_train_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "fitzpatrick_optimizer.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "residual-filter" in result.stdout
    assert "illumination-unet" in result.stdout


def test_evaluate_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "fitzpatrick_optimizer.evaluate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--split" in result.stdout
    assert "--metrics_json" in result.stdout


def test_infer_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "fitzpatrick_optimizer.infer", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--output_dir" in result.stdout
