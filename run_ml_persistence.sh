#!/usr/bin/env bash
set -euo pipefail

# Modes:
#   sequence:   none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   latent_tda: z,z_latent_h0,z_latent_h1,z_latent_both,z_latent_h0_zero,z_latent_h0_shuffle,z_latent_h0_noise,z_latent_h0_shift,z_latent_h1_zero,z_latent_h1_shuffle,z_latent_h1_noise,z_latent_h1_shift,z_latent_both_zero,z_latent_both_shuffle,z_latent_both_noise,z_latent_both_shift
#   aux_tda:    none,aux_h0,aux_h1,aux_both
#   pixel_tda:  none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   topo_pixel_z: z-only topo-AE latent history -> future frame X
#   topo_latent_tda: same ones as latent_tda, but with a topo-regularized AE encoder
#   topo_sequence: same as sequence, but with a topo-regularized AE encoder

args=(
  --scenario sequence                 # aux_tda | sequence | topo_sequence | latent_tda | topo_latent_tda | pixel_tda | topo_pixel_z
  --dataset celltracking_fluo             # moving_mnist | davis_images | celltracking_fluo | bouncing_balls
  --seeds 0,1,2,3,4,5,6,7,8,9,10                  # comma-separated seeds
  --modes none #z,z_latent_h1,z_latent_h1_zero,z_latent_h1_shuffle,z_latent_h1_noise,z_latent_h1_shift,z_latent_h0_zero # comma-separated modes; see above
  --aux-tda-lambda 1                 # aux_tda: auxiliary Betti loss weight
  --predict-steps-ahead 5
  --retrain-encoder
  --epochs 10
  --topo-ae-lambda 1
  --topo-ae-epochs 3
  --topo-ae-pair-batch-size 64

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
  # --topo-ae-lambda 0.1             # topo_*: shape-preservation regularizer weight
  # --topo-ae-epochs 3               # topo_*: topo-AE pretraining epochs
  # --topo-ae-pair-batch-size 64     # topo_*: batch size for pairwise-distance regularizer
  # --pixel-tda-batch-size 32        # pixel_tda: predictor batch size
  # --pixel-tda-fg-weight 10.0       # pixel_tda: foreground loss weight
  # --pixel-tda-fg-threshold 0.05    # pixel_tda: foreground threshold
  # --output-dir results/ml_persistence # output folder
  # --no-save                        # print only, no CSV/config output
)

uv run python -m topo.ml_runner "${args[@]}" "$@"
