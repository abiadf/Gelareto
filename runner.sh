#!/usr/bin/env bash

# Default: run both 1D and 2D with runner defaults.
# uv run python -m topo.runner --case both --save-plots

# uv run python -m topo.runner \
#     --case both \
#     --n-points 500000 \
#     --rows 300 \
#     --cols 300 \
#     --chunks 10 \
#     --save-plots

uv run python -m topo.runner --case 2d --dataset-2d ring_patch --ring-strength 1.0 --patch-strength 3.0 --rows 300 --cols 300 --chunks 10 \
  --save-plots

# Add --max-plot-chunks N only if you want sensitivity curves beyond the active --chunks value.

# Save plots to images/runner_outputs/run_YYYYMMDD_HHMMSS/.
# uv run python -m topo.runner --case both --save-plots
# uv run python -m topo.runner --case both --save-plots --plot-format pdf

# 1d/2d
# uv run python -m topo.runner --case 1d --n-points 100000 --chunks 10
# uv run python -m topo.runner --case 2d --rows 100 --cols 100 --chunks 10

# --dataset-1d constant, smooth, multiscale, random_walk, peaks, legacy
# --dataset-2d constant, hills, mountainous, rings, ring_patch, texture, legacy
# For ring_patch, tune ring visibility with --ring-strength, e.g. 0.5, 1.0, 2.0.
# Tune background patchiness with --patch-strength, e.g. 1.0, 2.0, 4.0.

# Chunk-count sensitivity.
# uv run python -m topo.runner --case both --chunks 1

# Force device.
# uv run python -m topo.runner --case both --device cpu
# uv run python -m topo.runner --case both --device cuda

# Smaller plot sanity check.
# uv run python -m topo.runner --case both --n-points 1000 --rows 30 --cols 30 --chunks 5 --save-plots
