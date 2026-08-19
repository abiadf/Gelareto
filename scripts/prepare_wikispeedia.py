"""Prepare and verify the local Wikispeedia graph/navigation benchmark."""

from topo.config import DATASET_CONFIGS
from topo.ml_tda_wikispeedia import load_wikispeedia, validate_wikispeedia


if __name__ == "__main__":
    data = load_wikispeedia(DATASET_CONFIGS["wikispeedia"])
    for key, value in validate_wikispeedia(data).items():
        print(f"{key}: {value}")
