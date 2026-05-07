import pathlib
import subprocess
import sys


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts/openpi_utils/resolve_training_yaml_metadata.py"


def _run_metadata_script(config_path: pathlib.Path) -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata: dict[str, str] = {}
    for line in completed.stdout.strip().splitlines():
        key, value = line.split("=", 1)
        metadata[key] = value
    return metadata


def test_resolve_training_yaml_metadata_with_inheritance(tmp_path: pathlib.Path) -> None:
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        "\n".join(
            [
                "dataset:",
                "  repo_id: lerobot_datasets/task6891011_level12_260304",
                "  data_dir: /groups/gch51606/lerobot_datasets/task6891011_level12_260304",
                "  assets_dir: ./assets/pi05_hsr_task6891011_level12_260304_finetune",
                "  asset_id: lerobot_datasets/task6891011_level12_260304",
                "model:",
                "  type: pi05",
                "gpu:",
                "  num_gpus: 16",
                "checkpoints:",
                "  base_model:",
                "    pi05: gs://openpi-assets/checkpoints/pi05_base/params",
            ]
        ),
        encoding="utf-8",
    )
    child_yaml = tmp_path / "child.yaml"
    child_yaml.write_text(
        "\n".join(
            [
                "_base_: base.yaml",
                "experiment:",
                "  name: derived_config",
            ]
        ),
        encoding="utf-8",
    )

    metadata = _run_metadata_script(child_yaml)

    assert metadata["DATASET_DATA_DIR"] == "/groups/gch51606/lerobot_datasets/task6891011_level12_260304"
    assert metadata["DATASET_REPO_ID"] == "lerobot_datasets/task6891011_level12_260304"
    assert metadata["DATASET_ASSETS_DIR"] == "./assets/pi05_hsr_task6891011_level12_260304_finetune"
    assert metadata["DATASET_ASSET_ID"] == "lerobot_datasets/task6891011_level12_260304"
    assert metadata["GPU_NUM_GPUS"] == "16"
    assert metadata["BASE_MODEL_URL"] == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert metadata["DATASET_HF_HOME"] == "/groups/gch51606"


def test_resolve_training_yaml_metadata_without_inheritance(tmp_path: pathlib.Path) -> None:
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        "\n".join(
            [
                "dataset:",
                "  repo_id: datasets/example_set",
                "  data_dir: /mnt/cache/datasets/example_set",
                "model:",
                "  type: pi0",
                "gpu:",
                "  num_gpus: 8",
                "checkpoints:",
                "  base_model:",
                "    pi0: gs://openpi-assets/checkpoints/pi0_base/params",
            ]
        ),
        encoding="utf-8",
    )

    metadata = _run_metadata_script(config_yaml)

    assert metadata["DATASET_DATA_DIR"] == "/mnt/cache/datasets/example_set"
    assert metadata["DATASET_REPO_ID"] == "datasets/example_set"
    assert metadata["DATASET_ASSETS_DIR"] == ""
    assert metadata["DATASET_ASSET_ID"] == ""
    assert metadata["GPU_NUM_GPUS"] == "8"
    assert metadata["BASE_MODEL_URL"] == "gs://openpi-assets/checkpoints/pi0_base/params"
    assert metadata["DATASET_HF_HOME"] == "/mnt/cache"
