#!/usr/bin/env bash
set -euo pipefail

# Scenarios:
# real_tda: AE z-history + optional frame/real-space TDA -> future z
# geo_real_tda: geoAE z-history + optional frame/real-space TDA -> future z
# topo_real_tda: topoAE z-history + optional frame/real-space TDA -> future z
# latent_tda: AE z-history + optional latent-trajectory TDA -> future z
# latent_stability/geo_latent_stability/topo_latent_stability: noised-input diagnostic measuring latent Hausdorff and diagram bottleneck changes
# representation_fidelity/geo_representation_fidelity/topo_representation_fidelity: input-to-latent metric distortion and H0/H1 bottleneck fidelity
# geo_latent_tda: geoAE z-history + optional latent-trajectory TDA -> future z
# topo_latent_tda: topoAE z-history + optional latent-trajectory TDA -> future z
# vae_latent_tda: VAE encoder z-history + optional latent-trajectory TDA -> future z
# byol_latent_tda: BYOL-style encoder z-history + optional latent-trajectory TDA -> future z
# vjepa_latent_tda: frozen V-JEPA video embeddings + optional latent-trajectory TDA -> future V-JEPA embedding
# clip_latent_tda: frozen CLIP frame embeddings + optional latent-trajectory TDA -> future CLIP embedding
# simvp: SimVP-style pixel predictor; frames is the standard pixel baseline, z_* modes add AE latent/TDA conditioning
# latent_classification: classify complete synthetic trajectories by motion regime or object type using latent/TDA summaries
# video3d_tda: AE z-history + streaming 3D cubical H0/H1/H2 features from growing frame volumes -> future X
# decode_z: AE z-history -> future z -> decoded future X
# geo_decode_z: geoAE z-history -> future z -> decoded future X
# topo_decode_z: topoAE z-history -> future z -> decoded future X
# pixel_tda: AE z-history + optional frame TDA -> future X
# geo_pixel_tda: geoAE z-history + optional frame TDA -> future X
# topo_pixel_tda: topoAE z-history + optional frame TDA -> future X
# aux_tda: AE z-history -> future z, with optional auxiliary Betti prediction head/loss
# geo_* models use the old pairwise-distance geometry proxy; topo_* models use H0 persistence-signature AE.
# VAE/BYOL/V-JEPA/CLIP are representation baselines for latent_tda; SimVP/video3d_tda are pixel-space video prediction baselines.

# Modes:
#   real_tda:   none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   geo_real_tda/topo_real_tda: same as real_tda
#   latent_tda: z,z_temporal_stats,z_temporal_stats_h0,z_temporal_stats_h1,z_temporal_stats_both,z_h0,z_h1,z_both,z_gate_both,z_pca_h0,z_pca_h1,z_pca_both,z_kpca_h0,z_kpca_h1,z_kpca_both,z_laplacian_h1,z_diffusion_h1,z_rff_h1,z_pi_h0,z_pi_h1,z_pi_both,z_landscape_h0,z_landscape_h1,z_landscape_both,z_perslay_h0,z_perslay_h1,z_perslay_both,z_temporal_stats_pi_h1,z_temporal_stats_landscape_h1,z_temporal_stats_perslay_h1,z_temporal_stats_pca_h1,z_temporal_stats_kpca_h1,topo_h1,topo_pi_h1,topo_pca_h1,topo_kpca_h1,z_fuse_h1,z_fuse_both,z_fuse_pi_h1,z_fuse_landscape_h1,z_fuse_perslay_h1,z_fuse_pca_h1,z_fuse_kpca_h1,z_fuse_laplacian_h1,z_fuse_diffusion_h1,z_fuse_rff_h1, plus *_zero,*_shuffle,*_noise,*_shift controls
#   geo_latent_tda/topo_latent_tda/vae_latent_tda/byol_latent_tda/vjepa_latent_tda: same as latent_tda
#   simvp: frames, plus latent-TDA modes such as z,z_temporal_stats,z_fuse_h1,z_fuse_pi_h1
#   latent_classification: z,h0,h1,both,z_h0,z_h1,z_both,z_h0_shuffle,z_h1_shuffle
#   video3d_tda: none,h0,h1,h2,h0_h1,h0_h2,h1_h2,all, plus *_zero,*_shuffle,*_noise,*_shift controls
#   pixel_tda:  none,h0,h1,both,h0_zero,h1_zero,both_zero,h0_shuffle,h1_shuffle,both_shuffle,h0_noise,h1_noise,both_noise,h0_shift,h1_shift,both_shift
#   geo_pixel_tda/topo_pixel_tda: same as pixel_tda; none is AE z-only -> future X
#   decode_z/geo_decode_z/topo_decode_z: z_decode only
#   aux_tda:    none,aux_h0,aux_h1,aux_both

# Synthetic/video/time-series datasets:
#   bouncing_rings: hollow/ring objects with Lorenz-like irregular motion
#   bouncing_disks: filled objects with Lorenz-like irregular motion
#   orbiting_rings: hollow/ring objects with periodic circular/elliptical orbit motion
#   orbiting_disks: filled objects with periodic circular/elliptical orbit motion
#   synthetic_motion_classification: matched four-source corpus for motion/object classification
#   character_trajectories: raw 3D pen trajectories with 20 character labels (downloaded through aeon)
#   moving_mnist: MNIST digits with random translation and rotation, rendered as sparkline clips
#   lorenz96: synthetic multivariate Lorenz-96 trajectories rasterized as heatmap frames
#   electric_devices: UCR/Aeon ElectricDevices time-series samples rendered as sparkline clips
#   noisy_frames: iid random-frame negative control; topology should not reliably help
#   glioblastoma/hela: CTC TIFF sequences windowed as full-frame clips by default

# Encoder workflow:
#   First fair run for a scenario/dataset: add --include-retrain-encoder and list all modes.
#   The encoder is refreshed once per seed, frozen, then reused for every mode including none/z.
#   Later reruns: remove --include-retrain-encoder to reuse the existing frozen encoder checkpoint.
# Stacked runs print each scenario/dataset as they finish, then print combined tables at the end.
# Use --decoder-type conv for new paper-quality decoded-image runs; default mlp preserves old results/checkpoints.
#
# Quick hyperparameter tuning protocol:
#   uv run python scripts/tune_latent_z_hparams.py --device auto
#   Then uncomment --hparam-file below. The tuner selects dataset-level predictor
#   params using z-only validation MSE, then all modes/scenarios reuse those params.

args=(
  # External raw-trajectory classification. CharacterTrajectories is resampled
  # to length 100; VR persistence is computed directly in its 3D signal space.
  --scenario latent_classification
  --dataset character_trajectories
  --seeds 0,1,2,3,4                 # matched seeds required by the paired test
  # z: latent trajectory summary; h0/h1/both: topology only; z_h*: fusion.
  # z_h1_shuffle is the equal-width specificity control for genuine H1.
  # Focused confirmation run; use the complete mode list above for a full table.
  --modes z,z_h0,z_h0_shuffle
  --predictor-epochs 100             # classifier epochs
  --hidden-dim 64                    # classifier hidden width
  --latent-tda-window 20
  --latent-tda-bins 16
  # For the matched synthetic version instead use:
  # --dataset synthetic_motion_classification --ae-epochs 10
  # --stability-noise-levels 0,0.01,0.03,0.05,0.10 # use for latent_stability scenario
  # --vae-beta 0.001                 # vae_latent_tda: KL weight
  # --vae-epochs 10                  # vae_latent_tda: pretraining epochs; default falls back to --ae-epochs
  # --byol-epochs 10                 # byol_latent_tda: pretraining epochs; default falls back to --ae-epochs
  # --byol-noise-std 0.05            # byol_latent_tda: two-view augmentation noise
  # --vjepa-repo facebook/vjepa2-vitl-fpc64-256 # vjepa_latent_tda: Hugging Face checkpoint
  # --vjepa-batch-size 1             # vjepa_latent_tda: lower this if GPU memory is tight
  # --vjepa-num-frames 16            # vjepa_latent_tda: frames per context window ending at each time step
  # --clip-repo openai/clip-vit-base-patch32 # clip_latent_tda: lightweight frozen CLIP frame encoder
  # --clip-batch-size 64
  # --clip-image-size 224
  # --simvp-input-frames 5           # simvp: past frames used to predict the horizon frame
  # --pixel-tda-batch-size 16        # simvp/pixel_tda minibatch size
  # --video3d-tda-bins 16            # video3d_tda: bins per H0/H1/H2 Betti curve
  # --video3d-tda-scale 15           # video3d_tda: normalizes Betti counts before concatenation
  # --video3d-tda-boundary-slices 2  # video3d_tda: rolling damage-region radius recorded in stream state
  # --recompute-video3d-tda-features   # video3d_tda: refresh cached H0/H1/H2 volume features
  # --latent-tda-bins 16
  # --aux-tda-lambda 1                 # aux_tda: auxiliary Betti loss weight
  # --num-train-clips 128 \
  # --num-test-clips 64 \
  # --profile-run                    # save profile.csv with wall time, peak memory, throughput, topo fraction, and mode overhead
  # --profile-sizes 16,32,64,128     # scaling curve: rerun with train/test and latent-TDA clip caps set to each size

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
  # --profile-run                    # enable profiling metrics; multiple --modes are profiled as separate runs
  # --profile-sizes 16,32,64,128     # profile scaling over clip caps; use one small dataset first
  # --output-dir results/ml_persistence # output folder
  # --no-save                        # print only, no CSV/config output
)

printf 'Running topo.ml_runner with args:\n'
printf '  %q\n' "${args[@]}" "$@"
uv run python -m topo.ml_runner "${args[@]}" "$@"
