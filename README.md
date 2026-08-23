# Sleep Apnea Detection Pipeline

**Obstructive Sleep Apnea (OSA) detection from polysomnography (PSG) signals** — a refactored, modular pipeline built on the **UCDDB** (St. Vincent's University Hospital / University College Dublin Sleep Apnea Database), **SHHS** and **MESA**.

## Project Goal

Build a **clinically deployable pre-screening / triage tool** for OSA — not a PSG replacement. The gap: ~90% of OSA cases are undiagnosed because overnight PSG is expensive, lab-bound, and scarce. A model that can flag "high probability of OSA — refer for PSG" from home/wearable sensor data fills a real clinical need.

**Final clinical target:** AHI estimation with Bland-Altman agreement (bias < ±3 events/hr, limits < ±15) and severity category accuracy > 70%.

---

## 📁 Repository Structure

```
Sleep irregularity/
├── pipeline/                      # Core modular pipeline (NEW)
│   ├── __init__.py               # Package exports
│   ├── config.py                 # All constants: paths, splits, hyperparams
│   ├── loader.py                 # EDF load, resample, respevt parse, gate check
│   ├── windows.py                # Windowing, labeling, z-score normalization
│   ├── dataset.py                # PyTorch Dataset + per-patient builder
│   ├── models.py                 # ApneaCNNLSTM (CNN + BiLSTM)
│   ├── train.py                  # train_one_epoch, validate, fit
│   ├── eval.py                   # evaluate, evaluate_threshold_sweep
│   ├── orchestrator.py           # Full-pipeline entry point
│   └── sanity.py                 # Single-patient smoke test
│
├── context/                       # Project documentation & decisions
│   ├── Project_goal.md           # North star, success criteria, constraints
│   ├── architecture_state.md     # Current pipeline step-by-step
│   ├── experiments_timeline.md   # Chronological EXP_001–EXP_010
│   ├── knowledge_base.md         # Thematic distilled findings
│   └── Confusion.md              # Open questions (apnea vs hypopnea)
│
├── datasets/                      # UCDDB data (not in repo)
│   └── st-vincents-.../files/     # .edf + _respevt.txt per patient
│
├── sleep-apnea-pipeline pycode.py # Original monolithic notebook export (kept for reference)
└── README.md                      # This file
```

---

## 🔬 Pipeline Overview (Steps 1–9)

| Step | Module | Function | Description |
|------|--------|----------|-------------|
| 1 | `loader.load_edf` | Load EDF + select 4 channels (`Flow`, `ribcage`, `abdo`, `SpO2`) |
| 2 | `loader.resample_to_10hz` | Anti-aliased decimation to 10 Hz via `resample_poly` |
| 3 | `loader.parse_respevt` | Parse `_respevt.txt` → list of `(onset_sec, duration_sec)` for APNEA/HYP |
| 4 | `windows.make_windows_and_labels` | Slice into 30s non-overlapping windows (300 samples); label=1 if event overlaps ≥ 10s |
| 5 | `windows.normalize` | Per-channel z-score on normal windows only; clip to ±5σ; SpO2 min-max scaled |
| 6 | `loader.run_gate_check` | Diagnostic print — **always run before training** |
| 7 | `models.ApneaCNNLSTM` | CNN encodes each window → BiLSTM across SEQ_LEN windows |
| 8 | `dataset` + `train` | `ApneaSequenceDataset`, WeightedRandomSampler, training loop |
| 9 | `eval` | Macro F1 (threshold sweep), confusion matrix, AHI-ready evaluation |

---

## 📊 Dataset — UCDDB

- **25 patients** (`ucddb002`–`ucddb028`; 004 and 016 missing)
- Each patient: `{patient}_lifecard.edf` + `{patient}_respevt.txt`
- **Channels used:** Flow (airflow), ribcage (thoracic effort), abdo (abdominal effort), SpO2
- **Patient split (fixed, not stratified):**
  - Train: `ucddb002`–`ucddb023` (20 patients)
  - Test: `ucddb024`–`ucddb028` (5 patients)
- *Stratified splitting planned before SHHS scaling — see `context/architecture_state.md`*

---

## ⚙️ Quick Start

### Prerequisites
```bash
pip install mne scipy scikit-learn torch pandas numpy matplotlib
```

### Configure data path
Edit `pipeline/config.py` — set `DATA_DIR` to your UCDDB `files/` directory:
```python
DATA_DIR = r"d:\Sleep irregularity\datasets\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0\files"
```

### Run sanity check (smoke test on 1 patient)
```bash
# Default: ucddb002, 15 epochs, batch=16
python -m pipeline.sanity

# Different patient
python -m pipeline.sanity --patient ucddb003

# Different epochs / batch size
python -m pipeline.sanity --epochs 5 --batch-size 32

# All flags
python -m pipeline.sanity --patient ucddb005 --epochs 20 --batch-size 8
```
**Expected:** Train loss drops below 0.10, accuracy > 90%. If not, pipeline is broken upstream.

### Run full pipeline
```bash
# Full 20-patient train / 5-patient test
python -m pipeline.orchestrator

# Sanity mode (same single patient for train + test)
python -m pipeline.orchestrator --sanity
```

---

## 🏃‍♂️ Running the Pipeline

### Sanity Check (`pipeline/sanity.py`)

Single-patient overfit test — confirms the entire pipeline (loader → dataset → model → train → eval) is wired correctly.

```bash
# Usage examples
python -m pipeline.sanity                              # ucddb002, 15 epochs, batch 16
python -m pipeline.sanity --patient ucddb003           # Different patient
python -m pipeline.sanity --epochs 5 --batch-size 32   # Faster smoke test
python -m pipeline.sanity --patient ucddb005 --epochs 20 --batch-size 8
```

**Outputs:** Gate check → training loss curve → final metrics + threshold sweep → **PASS/FAIL verdict**.

**Verdict criteria:** `train_loss < 0.10` AND `accuracy ≥ 0.90`. Failure means a bug in data pipeline, not the model.

### Orchestrator (`pipeline/orchestrator.py`)

End-to-end training + evaluation on the full patient split.

```bash
# Full pipeline (default: 20 train / 5 test patients, 30 epochs)
python -m pipeline.orchestrator

# Sanity mode (single patient for both train + test)
python -m pipeline.orchestrator --sanity
```

**Outputs:**
- Gate check on `ucddb002`
- Per-patient dataset build progress
- 30 epochs of training + validation loss
- Final evaluation: optimal threshold, accuracy, macro F1, confusion matrix
- Threshold sweep (sensitivity/precision/F1 at t=0.10–0.85)

---

## 🧪 Current Architecture

### Model: `ApneaCNNLSTM` (`pipeline/models.py`)
```
Input:  (batch, SEQ_LEN, 4, 300)   # 4 channels × 300 timesteps (30s @ 10Hz)
Output: (batch, SEQ_LEN)            # 1 logit per window in sequence

CNN Encoder (per window):
  Conv1d(4→32, k=7) + BN + ReLU + MaxPool(2)   → (32, 150)
  Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)  → (64, 75)
  Conv1d(64→128, k=3) + BN + ReLU + AdaptiveAvgPool1d(8) → (128, 8)
  Flatten + Linear(1024→128) + ReLU + Dropout  → (128,)

BiLSTM (across SEQ_LEN windows):
  LSTM(128→64, bidirectional=True)  → (SEQ_LEN, 128)
  Global mean pool + Classifier(128→32→1) → (SEQ_LEN, 1) logits
```

### Training Configuration (`pipeline/config.py`)
| Parameter | Value |
|-----------|-------|
| `SEQ_LEN` | 10 (5 minutes of context) |
| `BATCH_SIZE` | 64 |
| `NUM_EPOCHS` | 30 |
| `LEARN_RATE` | 1e-3 |
| `DROPOUT` | 0.3 |
| Loss | `BCEWithLogitsLoss` (no pos_weight — sampler handles imbalance) |
| Sampler | `WeightedRandomSampler` on sequence-level apnea ratio |

---

## 📈 Current Performance (UCDDB — EXP_004 Best)

| Metric | Value | Threshold |
|--------|-------|-----------|
| Macro F1 | 0.471 | t=0.65 |
| Sensitivity | 43.3% | t=0.50 |
| Precision | 16.1% | t=0.50 |
| Accuracy | 50.1% | t=0.50 |

**Key insight from experiments:** Precision has been stuck at **14–18% across all experiments** regardless of channel selection, class weighting, or patient filtering. Root cause: **no temporal context** (CNN sees each 30s window independently) + **hypopnea label ambiguity** (hypopnea windows visually resemble normal breathing). Next step: CNN+LSTM temporal modeling (EXP_010 in progress).

---

## 🧭 What's Next (Roadmap)

| Priority | Component | Status |
|----------|-----------|--------|
| 1 | **CNN + LSTM** (EXP_010) | In progress — temporal context to break precision floor |
| 2 | 3-class separation (Normal / Apnea / Hypopnea) | Queued — isolates label ambiguity |
| 3 | Name-based channel guard | Bug fix before SHHS — current count-based guard is a silent failure risk |
| 4 | SHHS pretraining (~6,000 subjects) | Planned — UCDDB becomes OOD holdout |
| 5 | AHI estimation post-processor | Rule-based temporal merge → events → AHI → Bland-Altman |
| 6 | AHI-stratified patient splitting | Required for SHHS scale |
| 7 | MESA cross-validation | Secondary dataset |
| 8 | Explainability (SHAP / attention) | Far future |

---

## 📚 Context Documents

| File | Purpose |
|------|---------|
| `context/Project_goal.md` | North star, clinical framing, success criteria, end-state constraints |
| `context/architecture_state.md` | Current pipeline step-by-step, model config, missing components table |
| `context/experiments_timeline.md` | Chronological EXP_001–EXP_010 with hypotheses, results, triggers |
| `context/knowledge_base.md` | Thematic distilled findings (label contamination, precision floor, channels, class imbalance, temporal context, dataset scale) |
| `context/Confusion.md` | Open questions on apnea vs hypopnea, 3-class experiment |

---

## ⚠️ Known Issues / Gotchas

1. **Count-based channel guard** in `build_dataset()` — skips patients with different channel *count* but not different channel *names*. Must fix to name-based matching before SHHS.
2. **Fixed patient split** is not AHI-stratified — test set severity distribution unknown.
3. **UCDDB only (25 patients)** — not sufficient for generalization. SHHS pretraining is the path forward.
4. **Hypopnea label contamination** — binary label merges APNEA and HYP; precision ceiling partially explained by this.
5. **No AHI estimation yet** — window-level F1 is a scaffolding metric; clinical target is AHI agreement.

---

## 📜 License

Research / educational use. UCDDB data license applies — see PhysioNet.

---

## 🤝 Contributing

This is an active research pipeline. See `context/experiments_timeline.md` for the current hypothesis queue. Before proposing changes, check `context/knowledge_base.md` "Rejected Ideas" table.

---

**Main entry points:**
- `python -m pipeline.sanity` — single-patient smoke test
- `python -m pipeline.orchestrator` — full pipeline