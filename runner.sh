#!/usr/bin/env bash

# Default: run both 1D and 2D with runner defaults.
uv run python -m topo.runner

# 1D only.
# uv run python -m topo.runner --case 1d --n-points 100000 --chunks 10

# 2D only.
# uv run python -m topo.runner --case 2d --rows 100 --cols 100 --chunks 10

# 1D datasets.
# uv run python -m topo.runner --case 1d --dataset-1d constant --n-points 100000 --chunks 10
# uv run python -m topo.runner --case 1d --dataset-1d smooth --n-points 100000 --chunks 10
# uv run python -m topo.runner --case 1d --dataset-1d multiscale --n-points 100000 --chunks 10
# uv run python -m topo.runner --case 1d --dataset-1d random_walk --n-points 100000 --chunks 10
# uv run python -m topo.runner --case 1d --dataset-1d peaks --n-points 100000 --chunks 10
# uv run python -m topo.runner --case 1d --dataset-1d legacy --n-points 100000 --chunks 10

# 2D datasets.
# uv run python -m topo.runner --case 2d --dataset-2d constant --rows 100 --cols 100 --chunks 10
# uv run python -m topo.runner --case 2d --dataset-2d hills --rows 100 --cols 100 --chunks 10
# uv run python -m topo.runner --case 2d --dataset-2d mountainous --rows 100 --cols 100 --chunks 10
# uv run python -m topo.runner --case 2d --dataset-2d rings --rows 100 --cols 100 --chunks 10
# uv run python -m topo.runner --case 2d --dataset-2d texture --rows 100 --cols 100 --chunks 10
# uv run python -m topo.runner --case 2d --dataset-2d legacy --rows 100 --cols 100 --chunks 10

# Chunk-count sensitivity.
# uv run python -m topo.runner --case both --chunks 1
# uv run python -m topo.runner --case both --chunks 2
# uv run python -m topo.runner --case both --chunks 5
# uv run python -m topo.runner --case both --chunks 10
# uv run python -m topo.runner --case both --chunks 20

# Force device.
# uv run python -m topo.runner --case both --device cpu
# uv run python -m topo.runner --case both --device cuda
