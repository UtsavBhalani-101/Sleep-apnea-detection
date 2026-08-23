"""
Pipeline-wide configuration: paths, patient splits, signal + training constants.
Single source of truth — edit here to retarget the pipeline.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = (
    r"d:\Sleep irregularity\datasets"
    r"\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
    r"\files"
)

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
