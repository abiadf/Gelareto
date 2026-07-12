#!/usr/bin/env bash
set -euo pipefail

# Scenarios:
# real_tda: AE z-history + optional frame/real-space TDA -> future z
# geo_real_tda: geoAE z-history + optional frame/real-space TDA -> future z
# topo_real_tda: topoAE z-history + optional frame/real-space TDA -> future z
# latent_tda: AE z-history + optional latent-trajectory TDA -> future z
# geo_latent_tda: geoAE z-history + optional latent-trajectory TDA -> future z
# topo_latent_tda: topoAE z-history + optional latent-trajectory TDA -> future z
# decode_z: AE z-history -> future z -> decoded future X
# geo_decode_z: geoAE z-history -> future z -> decoded future X
# topo_decode_z: topoAE z-history -> future z -> decoded future X
# pixel_tda: AE z-history + optional frame TDA -> future X
# geo_pixel_tda: geoAE z-history + optional frame TDA -> future X
# topo_pixel_tda: topoAE z-history + optional frame TDA -> future X
# aux_tda: AE z-history -> future z, with optional auxiliary Betti prediction head/loss
# geo_* models use the old pairwise-distance geometry proxy; topo_* models use H0 persistence-signature AE

# Modes:
#   real_tda:      none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   geo_real_tda/topo_real_tda: same as real_tda
#   latent_tda: z,z_temporal_stats,z_temporal_stats_latent_h0,z_temporal_stats_latent_h1,z_temporal_stats_latent_both,z_latent_h0,z_latent_h1,z_latent_both,z_latent_h0_zero,z_latent_h0_shuffle,z_latent_h0_noise,z_latent_h0_shift,z_latent_h1_zero,z_latent_h1_shuffle,z_latent_h1_noise,z_latent_h1_shift,z_latent_both_zero,z_latent_both_shuffle,z_latent_both_noise,z_latent_both_shift
#   geo_latent_tda/topo_latent_tda: same as latent_tda
#   pixel_tda:  none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   geo_pixel_tda/topo_pixel_tda: same as pixel_tda; none is AE z-only -> future X
#   decode_z/geo_decode_z/topo_decode_z: z_decode only
#   aux_tda:    none,aux_h0,aux_h1,aux_both

# Synthetic datasets:
#   bouncing_rings: hollow/ring objects with Lorenz-like irregular motion
#   bouncing_disks: filled objects with Lorenz-like irregular motion
#   orbiting_rings: hollow/ring objects with periodic circular/elliptical orbit motion
#   orbiting_disks: filled objects with periodic circular/elliptical orbit motion
#
# Encoder workflow:
#   First fair run for a scenario/dataset: add --include-retrain-encoder and list all modes.
#   The encoder is refreshed once per seed, frozen, then reused for every mode including none/z.
#   Later reruns: remove --include-retrain-encoder to reuse the existing frozen encoder checkpoint.
# Stacked runs print each scenario/dataset as they finish, then print combined tables at the end.

args=(
  --scenario real_tda,geo_real_tda  # comma-separated allowed; options: aux_tda | real_tda | geo_real_tda | topo_real_tda | latent_tda | geo_latent_tda | topo_latent_tda | pixel_tda | geo_pixel_tda | topo_pixel_tda | decode_z | geo_decode_z | topo_decode_z
  --dataset moving_mnist,celltracking_fluo #bouncing_disks,bouncing_rings,orbiting_disks,orbiting_rings # comma-separated allowed; moving_mnist | davis_images | celltracking_fluo | bouncing_rings | bouncing_disks | orbiting_rings | orbiting_disks
  --seeds 0,1,2,3,4                  # comma-separated seeds
  --modes none,h0,h1,both #,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift #,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift #z,z_latent_h0,z_latent_h1,z_latent_both #z,z_latent_h0,z_latent_h1,z_latent_both #none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift # comma-separated modes; see above
  --horizon 5
  --include-retrain-encoder        # train encoder once per seed, freeze it, then run all modes fairly
  --predictor-epochs 10
  --geo-ae-lambda 0.1
  --topo-ae-lambda 0.1
  --geo-ae-epochs 3
  --topo-ae-epochs 3
  --geo-ae-pair-batch-size 64
  --topo-ae-pair-batch-size 64
  --topo-ae-distance signature         # signature | wasserstein
  --latent-tda-window 15
  --recompute-latent-tda-features
  # --latent-tda-bins 16
  # --aux-tda-lambda 1                 # aux_tda: auxiliary Betti loss weight

  # --device auto                    # auto | cpu | cuda | mps
  # --horizon 5                      # override forecast horizon
  # --latent-dim 64                  # override encoder latent dimension
  # --hidden-dim 64                  # override LSTM hidden dimension
  # --predictor-epochs 10            # predictor training epochs; scenario-specific where applicable
  # --learning-rate 3e-4             # predictor learning rate
  # --real-tda-scale 15              # real/frame-space Betti curve normalization
  # --real-tda-bins 25               # real/frame-space Betti curve bins
  # --ae-frame-batch-size 256        # autoencoder frame batch size
  # --ae-max-frames-per-epoch 8192   # autoencoder frame subsample cap
  # --force-rebuild-data-cache       # synthetic datasets: rebuild cached clips
  # --num-train-clips 512            # synthetic datasets: train clips
  # --num-test-clips 128             # synthetic datasets: test clips
  # --include-retrain-encoder        # refresh shared encoder once per seed, then freeze for all modes
  # --retrain-encoder                # same as --include-retrain-encoder
  # --reuse-predictor                # load predictor checkpoint if present
  # --latent-tda-window 20           # latent_tda: trajectory window
  # --latent-tda-bins 16             # latent_tda: Betti bins
  # --latent-tda-max-train 200       # latent_tda: train clip subset
  # --latent-tda-max-test 100        # latent_tda: test clip subset
  # --recompute-latent-tda-features  # latent_tda: rebuild feature cache
  # --geo-ae-lambda 0.1              # geo_*: pairwise-distance geometry proxy weight
  # --geo-ae-epochs 3                # geo_*: geo-AE pretraining epochs
  # --geo-ae-pair-batch-size 64      # geo_*: minibatch size for geometry proxy
  # --topo-ae-lambda 0.1             # topo_*: H0 persistence regularizer weight
  # --topo-ae-epochs 3               # topo_*: topo-AE pretraining epochs
  # --topo-ae-pair-batch-size 64     # topo_*: minibatch size for persistence signature
  # --topo-ae-distance signature     # topo_*: signature | wasserstein
  # --pixel-tda-batch-size 32        # pixel_tda: predictor batch size
  # --pixel-tda-fg-weight 10.0       # pixel_tda: foreground loss weight
  # --pixel-tda-fg-threshold 0.05    # pixel_tda: foreground threshold
  # --output-dir results/ml_persistence # output folder
  # --no-save                        # print only, no CSV/config output
)

uv run python -m topo.ml_runner "${args[@]}" "$@"
