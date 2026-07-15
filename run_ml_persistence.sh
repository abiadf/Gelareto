#!/usr/bin/env bash
set -euo pipefail

# Scenarios:
# real_tda: AE z-history + optional frame/real-space TDA -> future z
# geo_real_tda: geoAE z-history + optional frame/real-space TDA -> future z
# topo_real_tda: topoAE z-history + optional frame/real-space TDA -> future z
# latent_tda: AE z-history + optional latent-trajectory TDA -> future z
# geo_latent_tda: geoAE z-history + optional latent-trajectory TDA -> future z
# topo_latent_tda: topoAE z-history + optional latent-trajectory TDA -> future z
# vae_latent_tda: VAE encoder z-history + optional latent-trajectory TDA -> future z
# byol_latent_tda: BYOL-style encoder z-history + optional latent-trajectory TDA -> future z
# vjepa_latent_tda: frozen V-JEPA video embeddings + optional latent-trajectory TDA -> future V-JEPA embedding
# decode_z: AE z-history -> future z -> decoded future X
# geo_decode_z: geoAE z-history -> future z -> decoded future X
# topo_decode_z: topoAE z-history -> future z -> decoded future X
# pixel_tda: AE z-history + optional frame TDA -> future X
# geo_pixel_tda: geoAE z-history + optional frame TDA -> future X
# topo_pixel_tda: topoAE z-history + optional frame TDA -> future X
# aux_tda: AE z-history -> future z, with optional auxiliary Betti prediction head/loss
# geo_* models use the old pairwise-distance geometry proxy; topo_* models use H0 persistence-signature AE.
# VAE/BYOL/V-JEPA are representation baselines for latent_tda; --predictor-type xlstm swaps the temporal predictor.

# Modes:
#   real_tda:   none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   geo_real_tda/topo_real_tda: same as real_tda
#   latent_tda: z,z_temporal_stats,z_temporal_stats_h0,z_temporal_stats_h1,z_temporal_stats_both,z_h0,z_h1,z_both,z_pi_h0,z_pi_h1,z_pi_both,z_landscape_h0,z_landscape_h1,z_landscape_both,z_perslay_h0,z_perslay_h1,z_perslay_both,z_temporal_stats_pi_h1,z_temporal_stats_landscape_h1,z_temporal_stats_perslay_h1,topo_h1,topo_pi_h1,z_fuse_h1,z_fuse_pi_h1,z_fuse_landscape_h1,z_fuse_perslay_h1, plus *_zero,*_shuffle,*_noise,*_shift controls
#   geo_latent_tda/topo_latent_tda/vae_latent_tda/byol_latent_tda/vjepa_latent_tda: same as latent_tda
#   pixel_tda:  none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   geo_pixel_tda/topo_pixel_tda: same as pixel_tda; none is AE z-only -> future X
#   decode_z/geo_decode_z/topo_decode_z: z_decode only
#   aux_tda:    none,aux_h0,aux_h1,aux_both

# Synthetic/video/time-series datasets:
#   bouncing_rings: hollow/ring objects with Lorenz-like irregular motion
#   bouncing_disks: filled objects with Lorenz-like irregular motion
#   orbiting_rings: hollow/ring objects with periodic circular/elliptical orbit motion
#   orbiting_disks: filled objects with periodic circular/elliptical orbit motion
#   lorenz96: synthetic multivariate Lorenz-96 trajectories rasterized as heatmap frames
#   electric_devices: UCR/Aeon ElectricDevices time-series samples rendered as sparkline clips
#   noisy_frames: iid random-frame negative control; topology should not reliably help
#   glioblastoma/hela: CTC TIFF sequences windowed as full-frame clips by default
#
# Encoder workflow:
#   First fair run for a scenario/dataset: add --include-retrain-encoder and list all modes.
#   The encoder is refreshed once per seed, frozen, then reused for every mode including none/z.
#   Later reruns: remove --include-retrain-encoder to reuse the existing frozen encoder checkpoint.
# Stacked runs print each scenario/dataset as they finish, then print combined tables at the end.
# Use --decoder-type conv for new paper-quality decoded-image runs; default mlp preserves old results/checkpoints.

args=(
  --scenario latent_tda,geo_latent_tda,vjepa_latent_tda #,topo_latent_tda #,vae_latent_tda,byol_latent_tda,vjepa_latent_tda # options: aux_tda real_tda geo_real_tda topo_real_tda latent_tda geo_latent_tda topo_latent_tda vae_latent_tda byol_latent_tda vjepa_latent_tda pixel_tda geo_pixel_tda topo_pixel_tda decode_z geo_decode_z topo_decode_z
  --dataset bouncing_disks,bouncing_rings,orbiting_disks,orbiting_rings #,moving_mnist,lorenz96,electric_devices,lorenz96,noisy_frames,celltracking_fluo,glioblastoma,hela,bouncing_disks,bouncing_rings,orbiting_disks,orbiting_rings,moving_mnist
  --seeds 0,1,2,3,4                 # comma-separated seeds
  --modes z,topo_h1,topo_pi_h1,z_fuse_h1,z_fuse_pi_h1,z_fuse_landscape_h1,z_fuse_perslay_h1 #z,z_pi_h0,z_pi_h1,z_pi_both,z_landscape_h0,z_landscape_h1,z_landscape_both,z_perslay_h0,z_perslay_h1,z_perslay_both,z_temporal_stats,z_temporal_stats_h0,z_temporal_stats_h1,z_temporal_stats_both,z_temporal_stats_pi_h1,z_temporal_stats_landscape_h1,z_temporal_stats_perslay_h1,topo_h1,topo_pi_h1,z_fuse_h1,z_fuse_pi_h1,z_fuse_landscape_h1,z_fuse_perslay_h1 # plus *_zero,*_shuffle,*_noise,*_shift controls
  --horizon 5
  --include-retrain-encoder         # train encoder once per seed, freeze it, then run all modes fairly
  --ae-epochs 10                    # baseline AE pretraining epochs
  --predictor-epochs 40 #10
  --predictor-type lstm             # lstm | xlstm; xlstm is a lightweight gated recurrent benchmark
  --geo-ae-lambda 0.1
  --topo-ae-lambda 0.1
  --geo-ae-epochs 3
  --topo-ae-epochs 3
  --geo-ae-pair-batch-size 64
  --topo-ae-pair-batch-size 64
  --topo-ae-distance signature         # signature | wasserstein
  --decoder-type mlp                   # mlp: old flat decoder | conv: convolutional upsampling decoder
  --latent-tda-window 15
  --recompute-latent-tda-features
  # --vae-beta 0.001                 # vae_latent_tda: KL weight
  # --vae-epochs 10                  # vae_latent_tda: pretraining epochs; default falls back to --ae-epochs
  # --byol-epochs 10                 # byol_latent_tda: pretraining epochs; default falls back to --ae-epochs
  # --byol-noise-std 0.05            # byol_latent_tda: two-view augmentation noise
  # --vjepa-repo facebook/vjepa2-vitl-fpc64-256 # vjepa_latent_tda: Hugging Face checkpoint
  # --vjepa-batch-size 1             # vjepa_latent_tda: lower this if GPU memory is tight
  # --vjepa-num-frames 16            # vjepa_latent_tda: frames per context window ending at each time step
  # --latent-tda-bins 16
  # --aux-tda-lambda 1                 # aux_tda: auxiliary Betti loss weight

  # --device auto                    # auto | cpu | cuda | mps
  # --horizon 5                      # override forecast horizon
  # --latent-dim 64                  # override encoder latent dimension
  # --hidden-dim 64                  # override LSTM hidden dimension
  # --predictor-epochs 10            # predictor training epochs; scenario-specific where applicable
  # --predictor-type lstm            # lstm | xlstm temporal predictor
  # --learning-rate 3e-4             # predictor learning rate
  # --real-tda-scale 15              # real/frame-space Betti curve normalization
  # --real-tda-bins 25               # real/frame-space Betti curve bins
  # --ae-epochs 10                   # baseline AE pretraining epochs
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
