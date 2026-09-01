"""
Per-dataset specifications (UCDDB, SHHS, MESA, ...).

A ``DatasetSpec`` captures everything that changes between datasets so the
rest of the pipeline can stay generic:

  * where the raw EDFs live (``data_subdir`` under ``DATA_DIR``, or an
    absolute ``data_dir`` override)
  * the EDF filename suffix and how to derive the patient ID from it
  * the annotation filename suffix and parser (``"respevt"`` vs
    ``"nsrr-xml"``)
  * which channels to keep, in the canonical order
    (SpO2/SaO2, Flow/Airflow, ribcage/thoracic, abdo/abdominal)
  * which event types count as apnea/hypopnea for the binary label

Adding a new dataset = adding one ``DatasetSpec`` here. Nothing else needs
to change.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    edf_suffix: str
    annotation_suffix: str
    annotation_format: str            # "respevt" | "nsrr-xml"
    channels: tuple[str, ...]         # canonical order
    # Channels whose events count as the apnea/hypopnea positive class.
    # Matched (case-insensitive) against the annotation record's EventConcept.
    apnea_event_substrings: tuple[str, ...] = ()
    hypopnea_event_substrings: tuple[str, ...] = ()
    # Patient-ID regex applied to the EDF filename (no suffix).
    patient_id_regex: str = r".+"
    # Optional override for where the EDFs actually live. If None, the loader
    # joins ``DATA_DIR`` with the configured subdirectory per dataset.
    data_dir: str | None = None
    # Optional separate directory for annotation files (NSRR XMLs live in a
    # sibling ``annotations-events-nsrr/<subset>/`` folder rather than next
    # to the EDFs). If None, falls back to ``data_dir``.
    annotations_dir: str | None = None

    def edf_path(self, pid: str) -> str:
        root = Path(self.data_dir) if self.data_dir else Path(config.DATA_DIR)
        return str(root / f"{pid}{self.edf_suffix}")

    def annotation_path(self, pid: str) -> str:
        ann_root = (
            Path(self.annotations_dir)
            if self.annotations_dir
            else (Path(self.data_dir) if self.data_dir else Path(config.DATA_DIR))
        )
        return str(ann_root / f"{pid}{self.annotation_suffix}")

    def parse_patient_id(self, filename: str) -> str | None:
        """Strip the EDF suffix and pull the patient ID out of the stem."""
        if not filename.lower().endswith(self.edf_suffix.lower()):
            return None
        stem = filename[: -len(self.edf_suffix)]
        match = re.match(self.patient_id_regex, stem)
        return match.group(0) if match else stem


# ─────────────────────────────────────────────────────────────────────────────
# UCDDB — St. Vincent's / University College Dublin Sleep Apnea Database
# ─────────────────────────────────────────────────────────────────────────────
UCDDB = DatasetSpec(
    name="ucddb",
    edf_suffix=".edf",
    annotation_suffix="_respevt.txt",
    annotation_format="respevt",
    channels=("SpO2", "Flow", "ribcage", "abdo"),
    apnea_event_substrings=("APNEA",),
    hypopnea_event_substrings=("HYP",),
    patient_id_regex=r"ucddb\d+",
)


# ─────────────────────────────────────────────────────────────────────────────
# SHHS — Sleep Heart Health Study (NSRR XML annotations)
# ─────────────────────────────────────────────────────────────────────────────
# SHHS ships split across ``shhs1`` and ``shhs2`` subfolders; this generic
# spec works for either as long as ``data_dir`` is set to the right
# ``edfs/shhsN`` folder. The annotation dir mirrors at
# ``annotations-events-nsrr/shhsN``.
SHHS_BASE = "/kaggle/input/datasets/antiti/shhs-dataset/polysomnography"


def make_shhs_spec(
    subset: str = "shhs1",
    data_dir: str | None = None,
    annotations_dir: str | None = None,
) -> DatasetSpec:
    """
    Build a SHHS spec. ``subset`` is "shhs1" or "shhs2"; ``data_dir``
    defaults to ``<SHHS_BASE>/edfs/<subset>`` and ``annotations_dir`` defaults
    to ``<SHHS_BASE>/annotations-events-nsrr/<subset>``.
    """
    if data_dir is None:
        data_dir = os.path.join(SHHS_BASE, "edfs", subset)
    if annotations_dir is None:
        annotations_dir = os.path.join(SHHS_BASE, "annotations-events-nsrr", subset)
    return DatasetSpec(
        name=f"shhs-{subset}",
        edf_suffix=".edf",
        annotation_suffix="-nsrr.xml",
        annotation_format="nsrr-xml",
        # SpO2, airflow, thoracic effort, abdominal effort (canonical order).
        channels=("SaO2", "NEW AIR", "THOR RES", "ABDO RES"),
        apnea_event_substrings=("obstructive apnea", "central apnea", "mixed apnea"),
        hypopnea_event_substrings=("hypopnea",),
        patient_id_regex=r"shhs\d-\d+",
        data_dir=data_dir,
        annotations_dir=annotations_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MESA — Multi-Ethnic Study of Atherosclerosis (placeholder, fill in once
# you've inspected one EDF + annotation file.)
# ─────────────────────────────────────────────────────────────────────────────
def make_mesa_spec(
    data_dir: str | None = None,
    annotations_dir: str | None = None,
) -> DatasetSpec:
    if data_dir is None:
        data_dir = os.path.join("/kaggle/input/mesa/polysomnography", "edfs")
    if annotations_dir is None:
        annotations_dir = os.path.join(
            "/kaggle/input/mesa/polysomnography", "annotations-events-nsrr"
        )
    return DatasetSpec(
        name="mesa",
        edf_suffix=".edf",
        annotation_suffix="-nsrr.xml",
        annotation_format="nsrr-xml",
        # TODO: confirm against a real MESA EDF.
        channels=("SaO2", "AIRFLOW", "THOR RES", "ABDO RES"),
        apnea_event_substrings=("obstructive apnea", "central apnea", "mixed apnea"),
        hypopnea_event_substrings=("hypopnea",),
        patient_id_regex=r"mesa-\d+",
        data_dir=data_dir,
        annotations_dir=annotations_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, DatasetSpec] = {
    "ucddb": UCDDB,
    "shhs1": make_shhs_spec("shhs1"),
    "shhs2": make_shhs_spec("shhs2"),
    "mesa": make_mesa_spec(),
}


def get_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def register_spec(spec: DatasetSpec) -> None:
    """Allow callers (e.g. custom Kaggle setups) to inject a new spec."""
    _REGISTRY[spec.name.lower()] = spec


# ─────────────────────────────────────────────────────────────────────────────
# SHHS / NSRR XML parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_nsrr_xml_events(xml_path: str) -> list[tuple[str, float, float]]:
    """
    Parse an NSRR-format XML annotation file.

    Returns a list of ``(event_concept, onset_sec, duration_sec)`` tuples
    (one per ``<ScoredEvent>`` with a parseable Start/Duration). The caller
    filters these against the dataset's apnea/hypopnea substrings.

    Returns ``[]`` if the file is missing.
    """
    if not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    events: list[tuple[str, float, float]] = []

    for scored in root.iter("ScoredEvent"):
        concept_el = scored.find("EventConcept")
        start_el = scored.find("Start")
        dur_el = scored.find("Duration")
        if concept_el is None or start_el is None or dur_el is None:
            continue
        concept = (concept_el.text or "").strip()
        try:
            onset = float(start_el.text or "nan")
            duration = float(dur_el.text or "nan")
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        events.append((concept, onset, duration))
    return events


def filter_apnea_events(
    raw_events: list[tuple[str, float, float]],
    spec: DatasetSpec,
) -> list[tuple[float, float]]:
    """Project (concept, onset, dur) tuples onto the dataset's apnea set."""
    apnea_substrings = tuple(s.lower() for s in spec.apnea_event_substrings)
    hypopnea_substrings = tuple(s.lower() for s in spec.hypopnea_event_substrings)
    out: list[tuple[float, float]] = []
    for concept, onset, duration in raw_events:
        c = concept.lower()
        if any(sub in c for sub in apnea_substrings) or any(
            sub in c for sub in hypopnea_substrings
        ):
            out.append((onset, duration))
    return out