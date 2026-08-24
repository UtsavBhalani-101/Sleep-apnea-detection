"""
Pipeline-wide configuration: paths, patient splits, signal + training constants.
Single source of truth — edit here to retarget the pipeline.
"""

import os
from pathlib import Path


def _resolve_data_dir() -> str:
    """
    Resolve dataset directory dynamically:
    1. Check environment variable 'DATA_DIR' or 'SLEEP_DATA_DIR'.
    2. Check Kaggle input directories (/kaggle/input/...).
    3. Check Google Colab input directories (/content/...).
    4. Check relative workspace datasets folder.
    5. Fallback to default local path.
    """
    # 1. Explicit Environment Variable
    env_dir = os.environ.get("DATA_DIR") or os.environ.get("SLEEP_DATA_DIR")
    if env_dir:
        return str(Path(env_dir).resolve())

    # 2. Kaggle environment: look for files directory under /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Common Kaggle dataset mount names
        candidates = list(kaggle_input.glob("**/files")) + list(kaggle_input.glob("*ucd*"))
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.edf")):
                return str(candidate)
        # Fallback to direct path under kaggle
        direct_kaggle = kaggle_input / "st-vincents-university-hospital-sleep-apnea" / "files"
        if direct_kaggle.exists():
            return str(direct_kaggle)

    # 3. Colab environment
    colab_dir = Path("/content/datasets/files")
    if colab_dir.exists():
        return str(colab_dir)

    # 4. Local workspace relative path
    repo_root = Path(__file__).resolve().parent.parent
    local_relative = (
        repo_root
        / "datasets"
        / "st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
        / "files"
    )
    if local_relative.exists():
        return str(local_relative)

    # 5. Default fallback path (Windows)
    return str(
        Path(
            r"d:\Sleep irregularity\datasets"
            r"\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
            r"\files"
        )
    )


DATA_DIR = _resolve_data_dir()

EDF_SUFFIX = ".edf"
RESP_EVT_SUFFIX = "_respevt.txt"

# Channels to keep when loading EDFs (order matters — index 0 is SpO2).
EDF_CHANNELS = ["SpO2", "Flow", "ribcage", "abdo"]

# ─────────────────────────────────────────────────────────────────────────────
# Patient splits
# ─────────────────────────────────────────────────────────────────────────────

ALL_PATIENTS = [
    "ucddb002", "ucddb003", "ucddb005", "ucddb006", "ucddb007",
    "ucddb008", "ucddb009", "ucddb010", "ucddb011", "ucddb012",
    "ucddb013", "ucddb014", "ucddb015", "ucddb017", "ucddb018",
    "ucddb019", "ucddb020", "ucddb021", "ucddb022", "ucddb023",
    "ucddb024", "ucddb025", "ucddb026", "ucddb027", "ucddb028",
]

TEST_PATIENTS = ["ucddb002", "ucddb003", "ucddb005", "ucddb006", "ucddb018"]
TRAIN_PATIENTS = [p for p in ALL_PATIENTS if p not in TEST_PATIENTS]
SANITY_CHECK_PATIENTS = ALL_PATIENTS[4:5]

# ─────────────────────────────────────────────────────────────────────────────
# Signal constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SFREQ = 10.0
WINDOW_SECONDS = 30
SAMPLES_PER_WIN = int(WINDOW_SECONDS * TARGET_SFREQ)

# Labeling rule: event must overlap window by at least this many seconds
OVERLAP_THRESHOLD_SECS = 10

# Z-score clip bound (artifact suppression)
CLIP_SIGMA = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# Sequence / training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN = 10

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARN_RATE = 1e-3
DROPOUT = 0.3

# Random seed for reproducibility
RANDOM_SEED = 42
