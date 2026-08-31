"""
Pipeline-wide configuration: paths, patient splits, signal + training constants.
Single source of truth — edit here to retarget the pipeline.
"""

import os
from pathlib import Path

# The expected dataset subfolder relative to your repo or Kaggle
DATA_REL_PATH = Path("datasets/st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0/files")

# Locations to search, in priority order:
CANDIDATE_PATHS = [
    os.environ.get("DATA_DIR"),                                     # 1. Env variable (if set)
    Path("/kaggle/input/datasets/antiti/ucddb-dataset/st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0/files"),  # 2. UCDDB dataset (actual path)
    Path("/kaggle/input/st-vincents-sleep-apnea/files"),            # 3. UCDDB dataset (standard path)
    Path("/kaggle/input/datasets/antiti/shhs-dataset/polysomnography"),  # 4. SHHS dataset base
    Path(__file__).resolve().parents[1] / DATA_REL_PATH,            # 5. Local repo folder
    Path(r"d:\Sleep irregularity") / DATA_REL_PATH,                 # 6. Windows absolute fallback
]

def get_data_dir() -> str:
    for candidate in CANDIDATE_PATHS:
        if candidate and Path(candidate).exists():
            return str(candidate)
    # If none exist, fail loudly with a helpful message:
    raise FileNotFoundError(
        "Could not find dataset directory. Set 'DATA_DIR' environment variable."
    )

DATA_DIR = get_data_dir()


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

# Regularization & training-loop knobs (added per Step-1 overfitting diagnosis).
WEIGHT_DECAY = 1e-3           # Adam weight decay
LSTM_DROPOUT = 0.5            # dropout applied after the BiLSTM (before classifier head)
EARLY_STOP_PATIENCE = 5       # stop if val loss doesn't improve for N epochs
LR_SCHEDULER_PATIENCE = 3     # plateau patience for ReduceLROnPlateau
LR_SCHEDULER_FACTOR = 0.5     # LR multiplier on plateau

# Random seed for reproducibility
RANDOM_SEED = 42
