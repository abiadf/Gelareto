#!/usr/bin/env bash
set -euo pipefail

datasets="bouncing_rings,bouncing_disks,orbiting_rings,orbiting_disks,moving_mnist,lorenz96,electric_devices,glioblastoma,hela"
lambdas=(0 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.5 0.7 1)

for geo_lambda in "${lambdas[@]}"; do
  uv run python -m topo.ml_runner \
    --scenario geo_latent_spectrum \
    --dataset "$datasets" \
    --seeds 0,1,2,3,4 \
    --geo-ae-lambda "$geo_lambda" \
    --geo-ae-epochs 3 \
    --geo-ae-pair-batch-size 64 \
    --decoder-type mlp \
    --output-dir results/geoae_pca_sweep

  uv run python -m topo.ml_runner \
    --scenario geo_latent_tda \
    --dataset "$datasets" \
    --seeds 0,1,2,3,4 \
    --modes z \
    --horizon 5 \
    --geo-ae-lambda "$geo_lambda" \
    --geo-ae-epochs 3 \
    --geo-ae-pair-batch-size 64 \
    --decoder-type mlp \
    --latent-tda-window 15 \
    --output-dir results/geoae_lambda_extra
done

uv run python scripts/make_geoae_lambda_pca_plot.py
uv run python scripts/make_geoae_lambda_plot.py
