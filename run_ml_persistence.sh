#!/usr/bin/env bash
set -euo pipefail

# Modes:
#   sequence:   none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   latent_tda: z,z_latent_h0,z_latent_h1,z_latent_both
#   aux_tda:    none,aux_h0,aux_h1,aux_both
#   pixel_tda:  none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift

args=(
  --scenario sequence                 # aux_tda | latent_tda | pixel_tda | sequence
  --dataset bouncing_balls             # moving_mnist | davis_images | celltracking_fluo | bouncing_balls
  --seeds 0,1,2                  # comma-separated seeds
  --modes none,h0,h1,both,h0_shuffle,h0_noise,h0_shift,h0_zero # comma-separated modes; see above
  --aux-tda-lambda 1                 # aux_tda: auxiliary Betti loss weight
  --predict-steps-ahead 5
  # --retrain-encoder
  --epochs 10

  # --device auto                    # auto | cpu | cuda | mps
  # --predict-steps-ahead 5          # override forecast horizon
  # --latent-dim 64                  # override encoder latent dimension
  # --hidden-dim 64                  # override LSTM hidden dimension
  # --epochs 10                      # predictor epochs; scenario-specific where applicable
  # --learning-rate 3e-4             # predictor learning rate
  # --betti-scale 15                 # image-space Betti curve normalization
  # --n-steps 25                     # image-space Betti curve bins
  # --ae-frame-batch-size 256        # autoencoder frame batch size
  # --ae-max-frames-per-epoch 8192   # autoencoder frame subsample cap
  # --force-rebuild-data-cache       # bouncing_balls: rebuild cached clips
  # --num-train-clips 512            # bouncing_balls: train clips
  # --num-test-clips 128             # bouncing_balls: test clips
  # --retrain-encoder                # ignore existing encoder checkpoint
  # --reuse-predictor                # load predictor checkpoint if present
  # --latent-tda-window 20           # latent_tda: trajectory window
  # --latent-tda-bins 16             # latent_tda: Betti bins
  # --latent-tda-max-train 200       # latent_tda: train clip subset
  # --latent-tda-max-test 100        # latent_tda: test clip subset
  # --recompute-latent-tda-features  # latent_tda: rebuild feature cache
  # --pixel-tda-batch-size 32        # pixel_tda: predictor batch size
  # --pixel-tda-fg-weight 10.0       # pixel_tda: foreground loss weight
  # --pixel-tda-fg-threshold 0.05    # pixel_tda: foreground threshold
  # --output-dir results/ml_persistence # output folder
  # --no-save                        # print only, no CSV/config output
)

uv run python -m topo.ml_runner "${args[@]}" "$@"
