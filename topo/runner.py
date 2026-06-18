import gc
import cProfile
import pstats

from time import perf_counter
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from memory_profiler import memory_usage
from time import perf_counter

import gudhi
from gudhi import CubicalComplex
import cripser
import dionysus as dion

from topo.persistence_1d import find_extrema_in_timeseries, compute_1d_sublevel_persistence, compute_batch_sublevel_persistence, \
    StreamingSublevelPersistence, BatchStreamingSublevelPersistence
from topo.persistence_2d import compute_h0_h1_fast, run_streaming_persistence, prepare_multi_chunk_wrapper

from topo.utils import DataUtils, MemoryUtils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_the_pipeline(input_dim = 2):
    """input_dim = dim of input data, 1-3 """
    # select between 1d or 2d
    if input_dim == 1:
        y_vals = DataUtils.make_yvals_1d(num_timesteps, start_point, end_point)
    elif input_dim == 2:
        arr_2d = DataUtils.generate_patchy_matrix(80, 80, device=device, min_val=0.0, max_val=100, smooth_radius=2)
    elif input_dim == 3:
        # arr_3d = DataUtils.generate_patchy_matrix(20, 20, 20, device=device, min_val=0.0, max_val=100, smooth_radius=2)
        pass
    arr_2d_np = arr_2d.numpy()
    h0, h1 = compute_h0_h1_fast(arr_2d_np)
    return h0, h1



if __name__ == "__main__":
    run_the_pipeline(input_dim = 2)
    print("Done!")

