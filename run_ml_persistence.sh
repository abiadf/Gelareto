#!/usr/bin/env bash
set -euo pipefail

uv run python -m topo.ml_runner \
  --scenario aux_tda \
  --dataset moving_mnist \
  --seeds 0,1,2,3,4 \
  --modes none,aux_h0,aux_h1,aux_both \
  --aux-tda-lambda 1 \
  "$@"
