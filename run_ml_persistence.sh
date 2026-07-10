#!/usr/bin/env bash
set -euo pipefail

# Scenarios:
# real_tda: AE z-history + optional frame/real-space TDA -> future z
# topo_real_tda: topoAE z-history + optional frame/real-space TDA -> future z
# latent_tda: AE z-history + optional latent-trajectory TDA -> future z
# topo_latent_tda: topoAE z-history + optional latent-trajectory TDA -> future z
# decode_z: AE z-history -> future z -> decoded future X
# topo_decode_z: topoAE z-history -> future z -> decoded future X
# pixel_tda: AE z-history + optional frame TDA -> future X
# topo_pixel_tda: topoAE z-history + optional frame TDA -> future X
# aux_tda: AE z-history -> future z, with optional auxiliary Betti prediction head/loss
# topo_* models: same as their non-topo counterparts, but with a topo-regularized AE encoder

# Modes:
#   real_tda:      none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   topo_real_tda: same as real_tda
#   latent_tda: z,z_latent_h0,z_latent_h1,z_latent_both,z_latent_h0_zero,z_latent_h0_shuffle,z_latent_h0_noise,z_latent_h0_shift,z_latent_h1_zero,z_latent_h1_shuffle,z_latent_h1_noise,z_latent_h1_shift,z_latent_both_zero,z_latent_both_shuffle,z_latent_both_noise,z_latent_both_shift
#   topo_latent_tda: same ones as latent_tda
#   pixel_tda:  none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   topo_pixel_tda: same as pixel_tda; none is topoAE z-only -> future X
#   decode_z/topo_decode_z: z_decode only
#   aux_tda:    none,aux_h0,aux_h1,aux_both

# Synthetic datasets:
#   bouncing_balls: hollow/ring objects with Lorenz-like irregular motion
#   bouncing_disks: filled objects with Lorenz-like irregular motion
#   orbiting_rings: hollow/ring objects with periodic circular/elliptical orbit motion
#   orbiting_disks: filled objects with periodic circular/elliptical orbit motion

args=(
  --scenario topo_real_tda             # aux_tda | real_tda | topo_real_tda | latent_tda | topo_latent_tda | pixel_tda | topo_pixel_tda | decode_z | topo_decode_z
  --dataset orbiting_rings             # moving_mnist | davis_images | celltracking_fluo | bouncing_balls | bouncing_disks | orbiting_rings | orbiting_disks
  --seeds 0,1,2 #,3,4                  # comma-separated seeds
  --modes none,h0,h1,both,h0_zero,h1_zero,both_zero #,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift #z,z_latent_h0,z_latent_h1,z_latent_both #z,z_latent_h0,z_latent_h1,z_latent_both #none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift # comma-separated modes; see above
  --horizon 10
  # --retrain-encoder
  --predictor-epochs 10
  --topo-ae-lambda 0.1
  --topo-ae-epochs 3
  --topo-ae-pair-batch-size 64
  # --latent-tda-window 30
  # --latent-tda-bins 16
  # --aux-tda-lambda 1                 # aux_tda: auxiliary Betti loss weight

  # --device auto                    # auto | cpu | cuda | mps
  # --horizon 5          # override forecast horizon
  # --latent-dim 64                  # override encoder latent dimension
  # --hidden-dim 64                  # override LSTM hidden dimension
  # --predictor-epochs 10          # predictor training epochs; scenario-specific where applicable
  # --learning-rate 3e-4             # predictor learning rate
  # --real-tda-scale 15            # real/frame-space Betti curve normalization
  # --real-tda-bins 25             # real/frame-space Betti curve bins
  # --ae-frame-batch-size 256        # autoencoder frame batch size
  # --ae-max-frames-per-epoch 8192   # autoencoder frame subsample cap
  # --force-rebuild-data-cache       # synthetic datasets: rebuild cached clips
  # --num-train-clips 512            # synthetic datasets: train clips
  # --num-test-clips 128             # synthetic datasets: test clips
  # --retrain-encoder                # ignore existing encoder checkpoint
  # --reuse-predictor                # load predictor checkpoint if present
  # --latent-tda-window 20           # latent_tda: trajectory window
  # --latent-tda-bins 16             # latent_tda: Betti bins
  # --latent-tda-max-train 200       # latent_tda: train clip subset
  # --latent-tda-max-test 100        # latent_tda: test clip subset
  # --recompute-latent-tda-features  # latent_tda: rebuild feature cache
  # --topo-ae-lambda 0.1             # topo_*: shape-preservation regularizer weight
  # --topo-ae-epochs 3             # topo_*: topo-AE pretraining epochs
  # --topo-ae-pair-batch-size 64     # topo_*: batch size for pairwise-distance regularizer
  # --pixel-tda-batch-size 32        # pixel_tda: predictor batch size
  # --pixel-tda-fg-weight 10.0       # pixel_tda: foreground loss weight
  # --pixel-tda-fg-threshold 0.05    # pixel_tda: foreground threshold
  # --output-dir results/ml_persistence # output folder
  # --no-save                        # print only, no CSV/config output
)

uv run python -m topo.ml_runner "${args[@]}" "$@"
