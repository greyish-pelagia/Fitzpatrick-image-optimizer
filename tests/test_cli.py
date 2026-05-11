import importlib.metadata


def test_package_imports():
    import fitzpatrick_optimizer

    assert fitzpatrick_optimizer.__version__ == "0.1.0"


def test_distribution_metadata_has_real_description():
    metadata = importlib.metadata.metadata("fitzpatrick-image-optimizer")

    assert metadata["Name"] == "fitzpatrick-image-optimizer"
    assert "experimental" in metadata["Summary"].lower()
