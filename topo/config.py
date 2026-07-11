"""Config file of datasets and their params"""
DATASET_CONFIGS = {
    "moving_mnist": {
        "kind": "moving_mnist",
        "path": "datasets/2D/mnist_test_seq.npy",
        "train_slice": slice(0, 1000),
        "test_slice": slice(9500, 9_700),
        "image_size": (64, 64),
        "learning_rate": 3e-4,
        "LATENT_DIM": 48,
        "HIDDEN_DIM": 64,
        "PREDICTOR_EPOCHS": 10,
        "REAL_TDA_SCALE": 15,
        "REAL_TDA_BINS": 25,
        "HORIZON": 5,
        "RUN_SEEDS": list(range(3)),

        # params for TDA-on-latents: 
        "LATENT_TDA_WINDOW" : 20,
        "LATENT_TDA_BINS"   : 16,
        "LATENT_TDA_PREDICTOR_EPOCHS" : 10,

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
        "clip_len": 30,
        "image_size": (96, 96),
        "base_radius": 6,
        "min_balls": 5,
        "max_balls": 8,
        "lorenz_dt": 0.018,
        "radius_pulse_amp": 0.45,
        "radius_pulse_freq": 0.35,
        "thickness_min": 2,
        "thickness_max": 5,
        "shape": "ring",
        "overlap_strength": 0.3,
        "normalize": "minmax",
        "train_seed_offset": 0,
        "test_seed_offset": 50000,
        "learning_rate": 3e-4,
        "LATENT_DIM": 64,
        "HIDDEN_DIM": 96,
        "PREDICTOR_EPOCHS": 4,
        "AE_FRAME_BATCH_SIZE": 256,
        "AE_MAX_FRAMES_PER_EPOCH": 8192,
        "REAL_TDA_SCALE": 5,
        "REAL_TDA_BINS": 25,
        "HORIZON": 5,
        "RUN_SEEDS": list(range(5)),

        # params for TDA-on-latents:
        "LATENT_TDA_WINDOW": 20,
        "LATENT_TDA_BINS": 16,
        "LATENT_TDA_PREDICTOR_EPOCHS": 8,

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
            "datasets/2D/glioblastoma/glioblastoma_train/01",
            "datasets/2D/glioblastoma/glioblastoma_train/02",
        ],
        "test_sequence_dirs": [
            "datasets/2D/glioblastoma/glioblastoma_test/02",
        ],
        "learning_rate": 3e-4,
        "LATENT_DIM": 128,
        "REAL_TDA_SCALE": 15,
        "REAL_TDA_BINS": 25,

        # CTC time-lapse preprocessing: raw stack -> spatial patch tracks -> temporal clips.
        "patch_size": 256,
        "spatial_stride": 128,
        "win_len": 20,
        "temporal_stride": 5,
        "normalize": "minmax",
        "empty_patch_filter": True,
        "min_temporal_std": 0.03, #removes clips with low temporal var (e.g. empty background)
        "min_mean_intensity": 0.01,
        "max_train_clips": 128,
        "max_test_clips": 64,
        "preprocessed_train_tensor": "datasets/2D/glioblastoma/processed/train_clips.pt",
        "preprocessed_test_tensor": "datasets/2D/glioblastoma/processed/test_clips.pt",

        # Clip-tensor ablation runner params.
        "RUN_SEEDS": list(range(3)),
        "HORIZON": 5,
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
        "LATENT_TDA_WINDOW" : 20,
        "LATENT_TDA_BINS"   : 16,
    },
    "hela": {
        "kind": "ctc_tif_clips",
        "root": "datasets/2D/hela",
        "train_sequence_dirs": [
            "datasets/2D/hela/HeLa_DIC-C2DH_train/01",
            "datasets/2D/hela/HeLa_DIC-C2DH_train/02",
        ],
        "test_sequence_dirs": [
            "datasets/2D/hela/HeLa_DIC-C2DH_test/01",
            "datasets/2D/hela/HeLa_DIC-C2DH_test/02",
        ],
        "learning_rate": 3e-4,
        "LATENT_DIM": 128,
        "REAL_TDA_SCALE": 15,
        "REAL_TDA_BINS": 25,

        # CTC time-lapse preprocessing: raw stack -> spatial patch tracks -> temporal clips.
        "patch_size": 256,
        "spatial_stride": 128,
        "win_len": 20,
        "temporal_stride": 5,
        "normalize": "minmax",
        "empty_patch_filter": True,
        "min_temporal_std": 0.03,
        "min_mean_intensity": 0.01,
        "max_train_clips": 96,
        "max_test_clips": 48,
        "preprocessed_train_tensor": "datasets/2D/hela/processed/train_clips.pt",
        "preprocessed_test_tensor": "datasets/2D/hela/processed/test_clips.pt",

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
    },}

DATASET_CONFIGS["bouncing_disks"] = {
    **DATASET_CONFIGS["bouncing_rings"],
    "shape": "disk",
    "cache_dir": "datasets/2D/bouncing_disks/processed",
}

_ORBITING_BASE = {
    "kind": "orbiting_shapes",
    "num_train_clips": 512,
    "num_test_clips": 128,
    "cache_dtype": "uint8",
    "clip_len": 30,
    "image_size": (96, 96),
    "base_radius": 6,
    "min_shapes": 2,
    "max_shapes": 4,
    "orbit_radius_min": 14,
    "orbit_radius_max": 32,
    "angular_speed_min": 0.18,
    "angular_speed_max": 0.42,
    "thickness_min": 2,
    "thickness_max": 4,
    "normalize": "minmax",
    "train_seed_offset": 0,
    "test_seed_offset": 50000,
    "learning_rate": 3e-4,
    "LATENT_DIM": 64,
    "HIDDEN_DIM": 96,
    "PREDICTOR_EPOCHS": 4,
    "AE_FRAME_BATCH_SIZE": 256,
    "AE_MAX_FRAMES_PER_EPOCH": 8192,
    "REAL_TDA_SCALE": 5,
    "REAL_TDA_BINS": 25,
    "HORIZON": 5,
    "RUN_SEEDS": list(range(5)),
    "LATENT_TDA_WINDOW": 20,
    "LATENT_TDA_BINS": 16,
    "LATENT_TDA_PREDICTOR_EPOCHS": 8,
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
