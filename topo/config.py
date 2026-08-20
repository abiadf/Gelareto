"""Config file of datasets and their params"""
DATASET_CONFIGS = {
    "diginetica": {
        "kind": "diginetica",
        "root": "datasets/2D/diginetica",
        "cache_path": "datasets/2D/diginetica/processed/diginetica.pt",
        "max_items": 5_000,
        "min_path_length": 6,
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "HORIZON": 1,
        "LATENT_TDA_WINDOW": 5,
        "LATENT_DIM": 16,
        # Selected on chronological validation sessions using the z-only mode.
        "HIDDEN_DIM": 64,
        "PREDICTOR_EPOCHS": 5,
        "learning_rate": 5e-3,
        "BATCH_SIZE": 256,
        "MAX_TRAIN_WINDOWS": 50_000,
        "GRAPH_GEO_LAMBDA": 0.1,
        "GRAPH_GEO_PAIR_BATCH": 64,
        # Validation winner among candidates guided by the h3_s10_e3 diagnosis.
        "MANIFOLD_SIGNATURE": "h2_s7_e7",
        "TDA_BINS": 8,
        "TDA_FUSION_EPOCHS": 3,
        "RUN_SEEDS": list(range(5)),
    },
    "retailrocket": {
        "kind": "retailrocket",
        "root": "datasets/2D/retailrocket",
        "cache_path": "datasets/2D/retailrocket/processed/retailrocket_v2.pt",
        "event_types": ["view", "addtocart", "transaction"],
        "session_gap_minutes": 30,
        "min_path_length": 6,
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "HORIZON": 1,
        "LATENT_TDA_WINDOW": 5,
        "LATENT_DIM": 16,
        "HIDDEN_DIM": 64,
        "PREDICTOR_EPOCHS": 8,
        "learning_rate": 3e-3,
        "BATCH_SIZE": 256,
        "MAX_TRAIN_WINDOWS": 50_000,
        "GRAPH_GEO_LAMBDA": 0.1,
        "GRAPH_GEO_PAIR_BATCH": 64,
        # Validation winner among the triangle-softened candidates.
        "MANIFOLD_SIGNATURE": "e16",
        "TDA_BINS": 8,
        "TDA_FUSION_EPOCHS": 3,
        "RUN_SEEDS": list(range(5)),
    },
    "wikispeedia": {
        "kind": "wikispeedia",
        # The downloaded directory is intentionally left at the user's existing
        # spelling; the public dataset and pipeline key use "Wikispeedia".
        "root": "datasets/2D/wikispedia/wikispeedia_paths-and-graph",
        "cache_path": "datasets/2D/wikispedia/processed/wikispeedia.pt",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        # Five context nodes plus the next-node target.
        "min_path_length": 6,
        "HORIZON": 1,
        "LATENT_TDA_WINDOW": 5,
        "LATENT_DIM": 16,
        "HIDDEN_DIM": 64,
        "PREDICTOR_EPOCHS": 8,
        "learning_rate": 3e-3,
        "BATCH_SIZE": 256,
        "MAX_TRAIN_WINDOWS": 50_000,
        "GRAPH_GEO_LAMBDA": 0.1,
        "GRAPH_GEO_PAIR_BATCH": 64,
        # Validation winner among triangle-softened candidates (test paths untouched).
        "MANIFOLD_SIGNATURE": "e16",
        "TDA_BINS": 8,
        "TDA_FUSION_EPOCHS": 3,
        "RUN_SEEDS": list(range(5)),
    },
    "hirros": {
        "kind": "hirros",
        "train_dir": "datasets/2D/hirros/hirros train",
        "test_dir": "datasets/2D/hirros/hirros test",
        "cache_dir": "datasets/2D/hirros/processed",
        "expected_frames": 29,
        "image_size": (64, 64),
        "learning_rate": 1e-3,
        "LATENT_DIM": 16,
        "HIDDEN_DIM": 128,
        "PREDICTOR_EPOCHS": 40,
        "AE_FRAME_BATCH_SIZE": 128,
        "AE_MAX_FRAMES_PER_EPOCH": None,
        "REAL_TDA_SCALE": 5,
        "REAL_TDA_BINS": 25,
        "HORIZON": 5,
        "RUN_SEEDS": list(range(3)),
        "LATENT_TDA_WINDOW": 15,
        "LATENT_TDA_BINS": 16,
        "LATENT_TDA_PREDICTOR_EPOCHS": 40,
    },
    "growing_tree": {
        "kind": "growing_tree",
        "num_train_clips": 512,
        "num_test_clips": 128,
        "cache_dir": "datasets/2D/growing_tree/processed",
        "clip_len": 64,
        "tree_depth": 5,
        "branching_factor": 4,
        "path_split_seed": 2026,
        "image_size": (64, 64),
        "train_seed_offset": 0,
        # Consecutive path IDs guarantee disjoint train/test root-to-leaf routes.
        "test_seed_offset": 512,
        "learning_rate": 1e-3,
        "LATENT_DIM": 16,
        "HIDDEN_DIM": 128,
        "PREDICTOR_EPOCHS": 40,
        "AE_FRAME_BATCH_SIZE": 256,
        "AE_MAX_FRAMES_PER_EPOCH": 8192,
        "REAL_TDA_SCALE": 5,
        "REAL_TDA_BINS": 25,
        "HORIZON": 5,
        "RUN_SEEDS": list(range(5)),
        "LATENT_TDA_WINDOW": 15,
        "LATENT_TDA_BINS": 16,
        "LATENT_TDA_PREDICTOR_EPOCHS": 40,
    },
    "moving_mnist": {
        "kind": "moving_mnist",
        "path": "datasets/2D/mnist_test_seq.npy",
        "train_slice": slice(0, 1000),
        "test_slice": slice(9500, 9_700),
        "image_size": (64, 64),
        "learning_rate": 3e-3,
        "LATENT_DIM": 48,
        "HIDDEN_DIM": 256,
        "PREDICTOR_EPOCHS": 40,
        "REAL_TDA_SCALE": 15,
        "REAL_TDA_BINS": 25,
        "HORIZON": 5,
        "RUN_SEEDS": list(range(3)),

        # params for TDA-on-latents: 
        "LATENT_TDA_WINDOW" : 20,
        "LATENT_TDA_BINS"   : 16,
        "LATENT_TDA_PREDICTOR_EPOCHS" : 40,

        # params for real-space TDA ablation:
        "REAL_TDA_MODES": ["none", "h0", "h0_shuffle", "h0_noise", "h0_shift"],
    },
    "davis_images": {
        "kind": "davis_images",
        "root": "datasets/2D/davis_images",
        "seq_len": 20,
        "stride": 5,
        "train_ratio": 0.7,
        "seed": 42,
        "max_train_windows": 400,
        "max_test_windows": 100,
        "image_size": (64, 64),
        "learning_rate": 1e-4,
        "LATENT_DIM": 16,
        "HIDDEN_DIM": 128,
        "PREDICTOR_EPOCHS": 10,
        "REAL_TDA_SCALE": 1,
        "REAL_TDA_BINS": 25,
        "HORIZON": 10,
        "RUN_SEEDS": list(range(3)),
        # params for TDA-on-latents: 
        "LATENT_TDA_WINDOW" : 20,
        "LATENT_TDA_BINS"   : 16,
        "LATENT_TDA_PREDICTOR_EPOCHS" : 10,

        # params for real-space TDA ablation:
        "REAL_TDA_MODES": ["none", "h0", "h0_shuffle", "h0_noise", "h0_shift"],
    },
    "celltracking_fluo": {
        "kind": "tif_folder",
        "folder": "datasets/2D/celltracking_fluo/01",
        "train_slice": slice(0, 45),
        "test_slice": slice(45, None),
        "image_size": (64, 64),
        "learning_rate": 3e-4,
        "LATENT_DIM": 128,
        "HIDDEN_DIM": 128,
        "PREDICTOR_EPOCHS": 10,
        "REAL_TDA_SCALE": 15,
        "REAL_TDA_BINS": 25,
        "HORIZON": 1,
        "RUN_SEEDS": list(range(5)),
        # params for TDA-on-latents: 
        "LATENT_TDA_WINDOW" : 20,
        "LATENT_TDA_BINS"   : 16,
        "LATENT_TDA_PREDICTOR_EPOCHS" : 10,

        # params for real-space TDA ablation:
        "REAL_TDA_MODES": ["none", "h0", "h0_shuffle", "h0_noise", "h0_shift"],
    },
    "bouncing_rings": {
        "kind": "lorenz_moving_shapes",
        "num_train_clips": 512,
        "num_test_clips": 128,
        "cache_dir": "datasets/2D/bouncing_rings/processed",
        "cache_dtype": "uint8",
        "clip_len": 50,
        "image_size": (64, 64),
        "base_radius": 4,
        "min_balls": 5,
        "max_balls": 8,
        "lorenz_dt": 0.018,
        "radius_pulse_amp": 0.45,
        "radius_pulse_freq": 0.35,
        "thickness_min": 1,
        "thickness_max": 3,
        "shape": "ring",
        "overlap_strength": 0.3,
        "normalize": "minmax",
        "train_seed_offset": 0,
        "test_seed_offset": 50000,
        "learning_rate": 5e-3,
        "LATENT_DIM": 64,
        "HIDDEN_DIM": 256,
        "PREDICTOR_EPOCHS": 40,
        "AE_FRAME_BATCH_SIZE": 256,
        "AE_MAX_FRAMES_PER_EPOCH": 8192,
        "REAL_TDA_SCALE": 5,
        "REAL_TDA_BINS": 25,
        "HORIZON": 5,
        "RUN_SEEDS": list(range(5)),

        # params for TDA-on-latents:
        "LATENT_TDA_WINDOW": 20,
        "LATENT_TDA_BINS": 16,
        "LATENT_TDA_PREDICTOR_EPOCHS": 40,

        # params for auxiliary topology prediction:
        "AUX_TDA_MODES": ["none", "aux_h0", "aux_h1", "aux_both"],
        "AUX_TDA_LAMBDA": 0.1,
        "AUX_TDA_PREDICTOR_EPOCHS": 4,

        # params for z+TDA -> future-frame prediction:
        "PIXEL_TDA_MODES": ["none", "h0", "h1", "both"],
        "PIXEL_TDA_PREDICTOR_EPOCHS": 4,
        "PIXEL_TDA_BATCH_SIZE": 32,
        "PIXEL_TDA_FG_WEIGHT": 10.0,
        "PIXEL_TDA_FG_THRESHOLD": 0.05,

        # params for xLSTM ablation:
        "XLSTM_MODES": ["none", "h0", "h0_zero", "h0_shuffle", "h0_noise", "h0_shift"],
        "XLSTM_BATCH_SIZE": 128,
        "XLSTM_MAX_TRAIN_WINDOWS": 4096,
        "XLSTM_MAX_TEST_WINDOWS": None,
        "XLSTM_STANDARDIZE_INPUTS": True,
        "XLSTM_STANDARDIZE_TARGETS": True,
        "XLSTM_RUN_LATENT_TDA": False,
        "XLSTM_RECOMPUTE_LATENT_TDA_FEATURES": False,
        "XLSTM_LATENT_TDA_MODES": ["z", "z_latent_h0", "z_latent_h1", "z_latent_both"],

        # params for real-space TDA ablation:
        "REAL_TDA_MODES": ["none", "h0", "h0_zero", "h0_shuffle", "h0_noise", "h0_shift"],
    },
    "glioblastoma": {
        "kind": "ctc_tif_clips",
        "root": "datasets/2D/glioblastoma",
        "train_sequence_dirs": [
            "datasets/2D/glioblastoma/PhC-C2DH-U373 train/01",
            "datasets/2D/glioblastoma/PhC-C2DH-U373 train/02",
        ],
        "test_sequence_dirs": [
            "datasets/2D/glioblastoma/PhC-C2DH-U373 test/01",
            "datasets/2D/glioblastoma/PhC-C2DH-U373 test/02",
        ],
        "learning_rate": 1e-2,
        "LATENT_DIM": 128,
        "HIDDEN_DIM": 256,
        "PREDICTOR_EPOCHS": 40,
        "REAL_TDA_SCALE": 15,
        "REAL_TDA_BINS": 25,

        # CTC time-lapse preprocessing: raw stack -> full-frame temporal clips.
        # Set full_frame=False to restore the old spatial patch tracks.
        "full_frame": True,
        "image_size": (64, 64),
        "patch_size": 256,
        "spatial_stride": 128,
        "win_len": 50,
        "temporal_stride": 1,
        "normalize": "minmax",
        "empty_patch_filter": False,
        "min_temporal_std": 0.03, #removes clips with low temporal var (e.g. empty background)
        "min_mean_intensity": 0.01,
        "max_train_clips": 256,
        "max_test_clips": 256,
        "preprocessed_train_tensor": "datasets/2D/glioblastoma/processed/train_full_frame_clips.pt",
        "preprocessed_test_tensor": "datasets/2D/glioblastoma/processed/test_full_frame_clips.pt",

        # Clip-tensor ablation runner params.
        "RUN_SEEDS": list(range(5)),
        "HORIZON": 5,
        "AE_EPOCHS": 2,
        "AE_MAX_FRAMES_PER_EPOCH": 4096,
        "MLP_EPOCHS": 4,
        "MLP_BATCH_SIZE": 512,
        "RUN_CELLTRACKING_LATENT_TDA": True,
        "INPUT_VARIANTS": {
            "no_tda": "z",
            "h0": "z_h0",
            "h0_shuffle": "z_h0_shuffle",
            "h0_noise": "z_h0_noise",
            "h0_shift": "z_h0_shift",
        },

        # params for TDA-on-latents: 
        "LATENT_TDA_WINDOW" : 20,
        "LATENT_TDA_BINS"   : 16,
        "LATENT_TDA_PREDICTOR_EPOCHS": 40,
    },
    "hela": {
        "kind": "ctc_tif_clips",
        "root": "datasets/2D/hela",
        "train_sequence_dirs": [
            "datasets/2D/hela/DIC-C2DH-HeLa train/01",
            "datasets/2D/hela/DIC-C2DH-HeLa train/02",
        ],
        "test_sequence_dirs": [
            "datasets/2D/hela/DIC-C2DH-HeLa test/01",
            "datasets/2D/hela/DIC-C2DH-HeLa test/02",
        ],
        "learning_rate": 1e-2,
        "LATENT_DIM": 128,
        "HIDDEN_DIM": 256,
        "PREDICTOR_EPOCHS": 40,
        "REAL_TDA_SCALE": 15,
        "REAL_TDA_BINS": 25,

        # CTC time-lapse preprocessing: raw stack -> full-frame temporal clips.
        # Set full_frame=False to restore the old spatial patch tracks.
        "full_frame": True,
        "image_size": (64, 64),
        "patch_size": 256,
        "spatial_stride": 128,
        "win_len": 50,
        "temporal_stride": 5,
        "normalize": "minmax",
        "empty_patch_filter": False,
        "min_temporal_std": 0.03,
        "min_mean_intensity": 0.01,
        "max_train_clips": 96,
        "max_test_clips": 48,
        "preprocessed_train_tensor": "datasets/2D/hela/processed/train_full_frame_clips.pt",
        "preprocessed_test_tensor": "datasets/2D/hela/processed/test_full_frame_clips.pt",

        # Clip-tensor ablation runner params.
        "RUN_SEEDS": list(range(3)),
        "HORIZON": 1,
        "AE_EPOCHS": 2,
        "AE_MAX_FRAMES_PER_EPOCH": 2048,
        "MLP_EPOCHS": 4,
        "MLP_BATCH_SIZE": 512,
        "RUN_CELLTRACKING_LATENT_TDA": True,
        "INPUT_VARIANTS": {
            "no_tda": "z",
            "h0": "z_h0",
            "h0_shuffle": "z_h0_shuffle",
            "h0_noise": "z_h0_noise",
            "h0_shift": "z_h0_shift",
        },

        # params for TDA-on-latents:
        "LATENT_TDA_WINDOW": 20,
        "LATENT_TDA_BINS": 16,
        "LATENT_TDA_PREDICTOR_EPOCHS": 40,
    },}

# Fluorescent HeLa nuclei from the Cell Tracking Challenge.  Keep this separate
# from ``hela`` (DIC-C2DH-HeLa): the modalities and cached tensors differ.  The
# current forecasting scenarios consume the raw image timelines; ``*_GT/TRA``
# lineage annotations are intentionally preserved for a later lineage task.
DATASET_CONFIGS["hela_fluo"] = {
    **DATASET_CONFIGS["hela"],
    "root": "datasets/2D/hela_fluo",
    "train_sequence_dirs": [
        "datasets/2D/hela_fluo/Fluo-N2DL-HeLa train/01",
        "datasets/2D/hela_fluo/Fluo-N2DL-HeLa train/02",
    ],
    "test_sequence_dirs": [
        "datasets/2D/hela_fluo/Fluo-N2DL-HeLa test/01",
        "datasets/2D/hela_fluo/Fluo-N2DL-HeLa test/02",
    ],
    "preprocessed_train_tensor": "datasets/2D/hela_fluo/processed/train_full_frame_clips.pt",
    "preprocessed_test_tensor": "datasets/2D/hela_fluo/processed/test_full_frame_clips.pt",
    # Use dense temporal windows without spatial duplication.  Spatially
    # transformed copies destabilized the GeoAE distance-preservation loss.
    # Test sequences remain untouched and retain stride 5.
    "train_temporal_stride": 1,
    "test_temporal_stride": 5,
    "train_augmentations": ["identity"],
    # Dense clips are ordered by source sequence: 43 from 01, then 43 from 02.
    # Selection scripts use these groups for leakage-free two-fold validation.
    "selection_sequence_clip_counts": [43, 43],
    "max_train_clips": None,
    "max_test_clips": None,
}

DATASET_CONFIGS["lorenz96"] = {
    "kind": "lorenz96_timeseries",
    "num_train_clips": 512,
    "num_test_clips": 128,
    "cache_dir": "datasets/timeseries/lorenz96/processed",
    "cache_dtype": "float32",
    "clip_len": 100,
    "n_vars": 40,
    "forcing": 8.0,
    "dt": 0.01,
    "sample_stride": 5,
    "transient_steps": 1000,
    "init_noise_std": 0.01,
    "value_scale": 10.0,
    "image_size": (64, 64),
    "normalize": "none",
    "train_seed_offset": 0,
    "test_seed_offset": 50000,
    "learning_rate": 1e-2,
    "LATENT_DIM": 64,
    "HIDDEN_DIM": 128,
    "PREDICTOR_EPOCHS": 40,
    "AE_FRAME_BATCH_SIZE": 256,
    "AE_MAX_FRAMES_PER_EPOCH": 8192,
    "REAL_TDA_SCALE": 5,
    "REAL_TDA_BINS": 25,
    "HORIZON": 5,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 20,
    "LATENT_TDA_BINS": 16,
    "LATENT_TDA_PREDICTOR_EPOCHS": 40,
    "PIXEL_TDA_MODES": ["none", "h0", "h1", "both"],
    "PIXEL_TDA_PREDICTOR_EPOCHS": 10,
    "PIXEL_TDA_BATCH_SIZE": 32,
    "PIXEL_TDA_FG_WEIGHT": 1.0,
    "PIXEL_TDA_FG_THRESHOLD": 0.05,
    "REAL_TDA_MODES": ["none", "h0", "h1", "both"],
}

DATASET_CONFIGS["noisy_frames"] = {
    "kind": "noisy_video",
    "num_train_clips": 512,
    "num_test_clips": 128,
    "cache_dir": "datasets/2D/noisy_frames/processed",
    "cache_dtype": "uint8",
    "clip_len": 50,
    "image_size": (64, 64),
    "noise_kind": "iid",
    "normalize": "none",
    "train_seed_offset": 0,
    "test_seed_offset": 50000,
    "learning_rate": 3e-4,
    "LATENT_DIM": 64,
    "HIDDEN_DIM": 96,
    "PREDICTOR_EPOCHS": 10,
    "AE_FRAME_BATCH_SIZE": 256,
    "AE_MAX_FRAMES_PER_EPOCH": 8192,
    "REAL_TDA_SCALE": 5,
    "REAL_TDA_BINS": 25,
    "HORIZON": 5,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 10,
    "LATENT_TDA_BINS": 16,
    "LATENT_TDA_PREDICTOR_EPOCHS": 10,
    "PIXEL_TDA_MODES": ["none", "h0", "h1", "both"],
    "PIXEL_TDA_PREDICTOR_EPOCHS": 10,
    "PIXEL_TDA_BATCH_SIZE": 32,
    "PIXEL_TDA_FG_WEIGHT": 1.0,
    "PIXEL_TDA_FG_THRESHOLD": 0.05,
    "REAL_TDA_MODES": ["none", "h0", "h1", "both"],
}

DATASET_CONFIGS["electric_devices"] = {
    "kind": "aeon_classification",
    "aeon_name": "ElectricDevices",
    "extract_path": "datasets/timeseries/aeon_data",
    "cache_dir": "datasets/timeseries/electric_devices/processed",
    "cache_dtype": "uint8",
    "image_size": (32, 32),
    "render_window": 16,
    "max_train_clips": 1024,
    "max_test_clips": 512,
    "subset_seed": 0,
    "learning_rate": 5e-3,
    "LATENT_DIM": 64,
    "HIDDEN_DIM": 256,
    "PREDICTOR_EPOCHS": 40,
    "AE_FRAME_BATCH_SIZE": 256,
    "AE_MAX_FRAMES_PER_EPOCH": 8192,
    "REAL_TDA_SCALE": 5,
    "REAL_TDA_BINS": 25,
    "HORIZON": 5,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 10,
    "LATENT_TDA_BINS": 16,
    "LATENT_TDA_PREDICTOR_EPOCHS": 40,
    "PIXEL_TDA_MODES": ["none", "h0", "h1", "both"],
    "PIXEL_TDA_PREDICTOR_EPOCHS": 10,
    "PIXEL_TDA_BATCH_SIZE": 32,
    "PIXEL_TDA_FG_WEIGHT": 1.0,
    "PIXEL_TDA_FG_THRESHOLD": 0.05,
    "REAL_TDA_MODES": ["none", "h0", "h1", "both"],
}

DATASET_CONFIGS["bouncing_disks"] = {
    **DATASET_CONFIGS["bouncing_rings"],
    "shape": "disk",
    "learning_rate": 5e-3,
    "cache_dir": "datasets/2D/bouncing_disks/processed",
}

_ORBITING_BASE = {
    "kind": "orbiting_shapes",
    "num_train_clips": 512,
    "num_test_clips": 128,
    "cache_dtype": "uint8",
    "clip_len": 50,
    "image_size": (64, 64),
    "base_radius": 4,
    "min_shapes": 2,
    "max_shapes": 4,
    "orbit_radius_min": 9,
    "orbit_radius_max": 21,
    "angular_speed_min": 0.18,
    "angular_speed_max": 0.42,
    "thickness_min": 1,
    "thickness_max": 3,
    "normalize": "minmax",
    "train_seed_offset": 0,
    "test_seed_offset": 50000,
    "learning_rate": 3e-3,
    "LATENT_DIM": 64,
    "HIDDEN_DIM": 256,
    "PREDICTOR_EPOCHS": 40,
    "AE_FRAME_BATCH_SIZE": 256,
    "AE_MAX_FRAMES_PER_EPOCH": 8192,
    "REAL_TDA_SCALE": 5,
    "REAL_TDA_BINS": 25,
    "HORIZON": 5,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 20,
    "LATENT_TDA_BINS": 16,
    "LATENT_TDA_PREDICTOR_EPOCHS": 40,
    "AUX_TDA_MODES": ["none", "aux_h0", "aux_h1", "aux_both"],
    "AUX_TDA_LAMBDA": 0.1,
    "AUX_TDA_PREDICTOR_EPOCHS": 4,
    "PIXEL_TDA_MODES": ["none", "h0", "h1", "both"],
    "PIXEL_TDA_PREDICTOR_EPOCHS": 4,
    "PIXEL_TDA_BATCH_SIZE": 32,
    "PIXEL_TDA_FG_WEIGHT": 10.0,
    "PIXEL_TDA_FG_THRESHOLD": 0.05,
    "REAL_TDA_MODES": ["none", "h0", "h1", "both"],
}

DATASET_CONFIGS["orbiting_rings"] = {
    **_ORBITING_BASE,
    "shape": "ring",
    "cache_dir": "datasets/2D/orbiting_rings/processed",
}

DATASET_CONFIGS["orbiting_disks"] = {
    **_ORBITING_BASE,
    "shape": "disk",
    "cache_dir": "datasets/2D/orbiting_disks/processed",
}

# Matched synthetic corpus for trajectory-level classification.  The source
# generators use identical object-count and appearance ranges so motion cannot
# be classified merely by counting objects.
_CLASSIFICATION_SOURCES = []
for _source_name in ("bouncing_rings", "bouncing_disks", "orbiting_rings", "orbiting_disks"):
    _source = dict(DATASET_CONFIGS[_source_name])
    _source["num_train_clips"] = 128
    _source["num_test_clips"] = 64
    _source["min_balls"] = 3
    _source["max_balls"] = 5
    _source["min_shapes"] = 3
    _source["max_shapes"] = 5
    _source["radius_pulse_amp"] = 0.0
    _source["cache_dir"] = f"datasets/2D/synthetic_motion_classification/{_source_name}/processed"
    _CLASSIFICATION_SOURCES.append(
        {
            "name": _source_name,
            "motion_label": 0 if _source_name.startswith("bouncing") else 1,
            "object_label": 0 if _source_name.endswith("disks") else 1,
            "config": _source,
        }
    )

DATASET_CONFIGS["synthetic_motion_classification"] = {
    "kind": "synthetic_motion_classification",
    "sources": _CLASSIFICATION_SOURCES,
    "learning_rate": 1e-3,
    "LATENT_DIM": 64,
    "HIDDEN_DIM": 64,
    "PREDICTOR_EPOCHS": 100,
    "AE_EPOCHS": 10,
    "AE_FRAME_BATCH_SIZE": 256,
    "AE_MAX_FRAMES_PER_EPOCH": 8192,
    "REAL_TDA_SCALE": 5,
    "REAL_TDA_BINS": 25,
    "HORIZON": 5,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 20,
    "LATENT_TDA_BINS": 16,
    "CLASSIFICATION_TASKS": ["motion", "object"],
    "CLASSIFICATION_MODES": ["z", "h0", "h1", "both", "z_h0", "z_h1", "z_both", "z_h0_shuffle", "z_h1_shuffle"],
}

DATASET_CONFIGS["character_trajectories"] = {
    "kind": "aeon_raw_classification",
    "aeon_name": "CharacterTrajectories",
    "extract_path": "datasets/timeseries/aeon_data",
    "cache_dir": "datasets/timeseries/CharacterTrajectories/processed",
    "resample_length": 100,
    "max_train_clips": None,
    "max_test_clips": None,
    "learning_rate": 1e-3,
    "LATENT_DIM": 3,
    "HIDDEN_DIM": 64,
    "PREDICTOR_EPOCHS": 100,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 20,
    "LATENT_TDA_BINS": 16,
    "CLASSIFICATION_TASKS": ["character"],
    "CLASSIFICATION_MODES": ["z", "h0", "h1", "both", "z_h0", "z_h1", "z_both", "z_h0_shuffle", "z_h1_shuffle"],
}

DATASET_CONFIGS["natops"] = {
    "kind": "aeon_raw_classification",
    "aeon_name": "NATOPS",
    "extract_path": "datasets/timeseries/aeon_data",
    "cache_dir": "datasets/timeseries/NATOPS/processed",
    "resample_length": 51,
    "max_train_clips": None,
    "max_test_clips": None,
    "learning_rate": 1e-3,
    "LATENT_DIM": 24,
    "HIDDEN_DIM": 64,
    "PREDICTOR_EPOCHS": 100,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 20,
    "LATENT_TDA_BINS": 16,
    "CLASSIFICATION_TASKS": ["gesture"],
    "CLASSIFICATION_MODES": ["z", "h0", "h1", "both", "z_h0", "z_h1", "z_both", "z_h0_shuffle", "z_h1_shuffle"],
}

# Prepared YouTube datasets are registered dynamically. Running
# ``scripts/prepare_youtube_sequences.py`` creates metadata and makes each video
# available on the next invocation as ``--dataset youtube_<name>``.
try:
    from topo.ml_tda_youtube import discover_youtube_dataset_configs

    DATASET_CONFIGS.update(discover_youtube_dataset_configs())
except ImportError:
    # Keep lightweight config inspection usable before optional video
    # dependencies have been installed.
    pass
