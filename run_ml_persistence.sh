#!/usr/bin/env bash
set -euo pipefail

# Final paper experiment (CPU-safe):
#   latent_tda: baseline AE
#   geo_latent_tda: Euclidean distance-preserving GeoAE
#   manifold_mixed_geo_latent_tda: mixed-curvature GeoAE
# Modes: z is latent-only; z_both appends H0/H1 persistence features.
# The default run uses the product-manifold metric for the manifold model's VR.
# After it finishes, run the Euclidean-VR control shown below.
#
# Main scenario reference:
#   real_tda / geo_real_tda / topo_real_tda:
#     Predict future z using optional persistence computed directly on frames.
#   latent_tda:
#     Standard AE followed by latent-trajectory persistence and forecasting.
#   geo_latent_tda:
#     Euclidean distance-preserving GeoAE followed by latent TDA.
#   topo_latent_tda:
#     Topology-regularized AE followed by latent TDA.
#   mixed_geo_latent_tda:
#     Mixed-curvature GeoAE; latent persistence still uses ordinary Euclidean distance.
#   manifold_mixed_geo_latent_tda:
#     The same mixed-curvature encoder with a selectable Euclidean or product-manifold
#     distance for the Vietoris--Rips filtration. This is the proposed full method.
#   triangle_curvature:
#     Training-data kNN triangle diagnostic used to propose H/S/E signatures.
#   routed_mixed_geo_latent_tda:
#     Experimental per-relation routing ablation; retained, but not a main method.
#   pwgeo_latent_tda:
#     Experimental persistence-weighted geometry-loss ablation.
#   latent_stability / representation_fidelity:
#     Diagnostics for noise stability and input-to-latent metric/topology fidelity.
#   decode_z / geo_decode_z / topo_decode_z:
#     Forecast in latent space and decode the predicted future frame.
#   simvp / pixel_tda / video3d_tda:
#     Pixel-space and spatiotemporal persistence baselines.
#
# Common latent modes:
#   z                 latent trajectory only
#   z_h0 / z_h1       append H0 or H1 persistence features
#   z_both            append both H0 and H1 persistence features
#   z_pca_*            PCA controls
#   z_kpca_*           kernel-PCA controls
#   z_fuse_*           learned feature-fusion variants
#   *_zero             replace topology features with zeros
#   *_shuffle          shuffle topology features
#   *_noise            replace topology features with noise
#   *_shift            temporally shift topology features
#
# Curvature-feature ablation modes:
#   z_curv_both        z plus curvature-persistence features
#   z_both_curv_both   z plus ordinary and curvature-persistence features
#
# Dataset reference:
#   bouncing_rings / bouncing_disks       irregular synthetic motion
#   orbiting_rings / orbiting_disks       periodic synthetic motion
#   moving_mnist                           translated/rotated MNIST sequences
#   lorenz96                               rasterized multivariate dynamics
#   electric_devices                       rasterized UCR time series
#   glioblastoma / hela                    CTC microscopy sequences
#
# Encoder/cache workflow:
#   Add --include-retrain-encoder when intentionally refreshing encoders. Otherwise,
#   compatible frozen checkpoints and feature caches are reused. Add
#   --recompute-latent-tda-features only when the encoder, window, bins, signature,
#   or VR metric changed and the cache cannot safely be reused.
#
# Hyperparameter and manifold-signature selection:
#   uv run python scripts/tune_latent_z_hparams.py --device cpu
#   uv run python scripts/select_manifold_signatures.py --device cpu
# Predictor learning rate, hidden size, and epochs are selected using z-only
# validation MSE and then reused across every scenario/mode for that dataset.

export PYTHONWARNINGS="ignore::SyntaxWarning"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$MPLCONFIGDIR"

args=(
  --scenario manifold_mixed_geo_latent_tda #latent_tda,geo_latent_tda,manifold_mixed_geo_latent_tda
  --dataset moving_mnist,lorenz96,electric_devices,glioblastoma,hela # bouncing_rings,bouncing_disks,orbiting_rings,orbiting_disks
  --seeds 0,1,2,3,4
  --modes z_both #z,z_both
  --latent-dim 16

  # Candidate signatures come from the training-only triangle diagnostic and
  # include Euclidean-softened alternatives selected later by validation.
  # --manifold-signature-policy triangle_shrinkage
  # --signature-shrinkages 0,0.5,0.75,1
  # --triangle-knn 4,8,12
  # --triangle-samples 10000
  # --triangle-max-points 512
  # Final runs load exactly one training/validation-selected signature per dataset.
  --manifold-signature-file results/signature_selection/selected_manifold_signatures.json

  # product_manifold: H/S/E geodesics; euclidean: ordinary cdist on the same z.
  --vr-distance euclidean #product_manifold # euclidean
  --device cpu
  --geo-ae-epochs 10
  --ae-epochs 10
  --geo-ae-lambda 0.1
  --learning-rate 1e-3
  --hparam-file results/hparam_search/latent_z_best_hparams.json
  --horizon 5
  --latent-tda-window 15
  --latent-tda-bins 16
  --output-dir results/mixed_curvature_cpu

  # Useful optional overrides:
  # --manifold-signature-policy manual
  # --manifold-signatures e16,h3_s4_e9
  # --manifold-signature-file results/signature_selection/selected_manifold_signatures.json
  # --include-retrain-encoder
  # --recompute-latent-tda-features
  # --hidden-dim 128
  # --predictor-learning-rate 1e-3
  # --predictor-epochs 40
  # --ae-frame-batch-size 256
  # --ae-max-frames-per-epoch 8192
  # --geo-ae-pair-batch-size 64
  # --latent-tda-max-train 200
  # --latent-tda-max-test 100
  # --num-train-clips 512
  # --num-test-clips 128
  # --profile-run
  # --profile-sizes 16,32,64,128

  # General experiment controls:
  # --device auto                    # auto | cpu | cuda | mps
  # --seeds 0,1,2                    # comma-separated repeat seeds
  # --horizon 5                      # forecast steps ahead
  # --latent-dim 16                  # common encoder output dimension
  # --decoder-type conv              # conv for decoded-image runs; mlp is legacy default
  # --reuse-predictor                # load compatible predictor checkpoints
  # --no-save                        # print results without writing CSV/config output
  # --output-dir results/ml_persistence

  # Predictor controls (normally supplied per dataset by --hparam-file):
  # --predictor-type lstm            # lstm | xlstm
  # --hidden-dim 128                 # LSTM internal hidden dimension
  # --predictor-learning-rate 1e-3   # predictor only; distinct from encoder LR
  # --predictor-epochs 40

  # Standard AE controls:
  # --learning-rate 1e-3             # fixed encoder/autoencoder learning rate
  # --ae-epochs 10
  # --ae-frame-batch-size 256
  # --ae-max-frames-per-epoch 8192
  # --include-retrain-encoder        # refresh encoder once per seed
  # --retrain-encoder                # alias of --include-retrain-encoder

  # Latent-persistence controls:
  # --latent-tda-window 15
  # --latent-tda-bins 16
  # --latent-tda-max-train 200
  # --latent-tda-max-test 100
  # --recompute-latent-tda-features  # rebuild cached latent/PH features

  # Euclidean GeoAE controls:
  # --geo-ae-lambda 0.1
  # --geo-ae-epochs 10
  # --geo-ae-pair-batch-size 64

  # TopoAE controls:
  # --topo-ae-lambda 0.1
  # --topo-ae-epochs 10
  # --topo-ae-pair-batch-size 64
  # --topo-ae-distance signature     # signature | wasserstein

  # Product-manifold controls:
  # --vr-distance product_manifold   # product_manifold | euclidean
  # --manifold-signature-policy triangle_shrinkage # triangle_shrinkage | manual
  # --signature-shrinkages 0,0.5,0.75,1
  # --manifold-signatures e16,h3_s4_e9
  # --triangle-knn 4,8,12
  # --triangle-samples 10000
  # --triangle-max-points 512

  # Real/frame-space persistence:
  # --real-tda-scale 15
  # --real-tda-bins 25

  # Pixel and SimVP baselines:
  # --simvp-input-frames 5
  # --pixel-tda-batch-size 32
  # --pixel-tda-fg-weight 10.0
  # --pixel-tda-fg-threshold 0.05

  # Streaming 3D persistence:
  # --video3d-tda-bins 16
  # --video3d-tda-scale 15
  # --video3d-tda-boundary-slices 2
  # --recompute-video3d-tda-features

  # Auxiliary topology prediction:
  # --aux-tda-lambda 1.0

  # VAE/BYOL representation baselines:
  # --vae-beta 0.001
  # --vae-epochs 10
  # --byol-epochs 10
  # --byol-noise-std 0.05

  # Frozen foundation-model baselines:
  # --vjepa-repo facebook/vjepa2-vitl-fpc64-256
  # --vjepa-batch-size 1
  # --vjepa-num-frames 16
  # --clip-repo openai/clip-vit-base-patch32
  # --clip-batch-size 64
  # --clip-image-size 224

  # Stability/fidelity diagnostics:
  # --stability-noise-levels 0,0.01,0.03,0.05,0.10

  # Retained research-ablation parameters:
  # --route-lambda 0.03
  # --route-knn 5
  # --route-temperature 0.1
  # --pwgeo-ae-lambda 0.1
  # --pwgeo-knn 5
  # --pwgeo-h0-weight 2.0
  # --pwgeo-h1-weight 1.0
  # --pwgeo-blend 0,0.1,0.25,0.5
)

printf 'Running topo.ml_runner with args:\n'
printf '  %q\n' "${args[@]}" "$@"
uv run python -m topo.ml_runner "${args[@]}" "$@"

# Euclidean-VR control (do not rerun the baseline scenarios or manifold z):
# bash run_ml_persistence.sh \
#   --scenario manifold_mixed_geo_latent_tda \
#   --modes z_both \
#   --vr-distance euclidean
