import os

setup_config = {

    "DATASETS_ROOT": "/home/shimon/research/datasets",
    "REPO_BASE": "/home/shimon/research/diffusion_inversions/FPI",
    "OUTPUT_ROOT": "/home/shimon/research/diffusion_inversions/results",
}

setup_config = {k: os.environ[k] if k in os.environ else v for k, v in setup_config.items()}

