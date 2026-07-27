#!/usr/bin/env bash
set -euo pipefail

# Profiles PCA/KPCA, spectral, RFF, and topology controls for latent forecasting.
# Run once after the corresponding predictors/encoders already exist, so
# --reuse-predictor avoids timing retraining noise.

mkdir -p latex_tables

uv run python -m topo.ml_runner \
  --scenario latent_tda,geo_latent_tda,topo_latent_tda \
  --dataset bouncing_disks,bouncing_rings,orbiting_disks,orbiting_rings,moving_mnist,lorenz96,electric_devices,glioblastoma,hela \
  --seeds 0 \
  --modes z,z_pca_h1,z_kpca_h1,z_fuse_h1,z_fuse_pi_h1,z_fuse_landscape_h1,z_fuse_perslay_h1,z_fuse_pca_h1,z_fuse_kpca_h1,z_fuse_laplacian_h1,z_fuse_diffusion_h1,z_fuse_rff_h1 \
  --horizon 5 \
  --predictor-type lstm \
  --latent-tda-window 15 \
  --latent-tda-bins 16 \
  --profile-run \
  --profile-sizes 128 \
  --profile-warmup-runs 1 \
  --profile-repeats 3 \
  --reuse-predictor \
  "$@" | tee latex_tables/profiling.txt
