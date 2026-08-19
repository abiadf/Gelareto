"""Build and verify the CPU-friendly Retailrocket category-sequence cache."""

from topo.config import DATASET_CONFIGS
from topo.ml_tda_retailrocket import load_retailrocket, validate_retailrocket


if __name__ == "__main__":
    data = load_retailrocket(DATASET_CONFIGS["retailrocket"])
    for key, value in validate_retailrocket(data).items():
        print(f"{key}: {value}")
