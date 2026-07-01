"""Wspólny builder manifestu dla wszystkich minerów Poker44.

Centralizuje: własne repo_url, PRZYPINANY commit (env POKER44_MODEL_REPO_COMMIT),
artifact_sha256 liczony z faktycznie ładowanego pliku modelu oraz uczciwą atestację.
Dzięki temu commit modelu N nie rusza commitów pozostałych (są przypinane per proces).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)

DEFAULT_REPO_URL = "https://github.com/Adamkoss111/poker44_ml"


def _sha256_file(path: Path) -> str:
    """SHA256 artefaktu (.joblib) — realny, różny per model wyróżnik tożsamości."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return ""


def _commit_from_map(repo_root: Path, model_path: Path) -> str:
    """Odczytuje commit przypięty do TEGO modelu z published/commit_map.json.
    Klucz = nazwa pliku modelu (np. 'model2.joblib'). Zapisywana przez publish_all.sh."""
    try:
        mp = Path(repo_root) / "published" / "commit_map.json"
        if not mp.exists():
            return ""
        data = json.loads(mp.read_text())
        return str(data.get(Path(model_path).name, "")).strip()
    except Exception:
        return ""


def resolve_repo_commit(repo_root: Path, model_path: Path | None = None) -> str:
    """Commit PRZYPINANY per model. Kolejność: env -> mapa commitów -> HEAD (fallback).
    Dzięki mapie każdy z 24 minerów startowanych jednym launcherem znajdzie SWÓJ commit."""
    env = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
    if env:
        return env
    if model_path is not None:
        mapped = _commit_from_map(repo_root, model_path)
        if mapped:
            return mapped
    return _git_head(repo_root)


def build_manifest(
    *,
    repo_root: Path,
    model_path: Path,
    model_name: str,
    model_version: str | int,
    implementation_files: Iterable[Path],
    framework: str = "xgboost",
) -> Tuple[dict, dict, str]:
    """Zwraca (manifest, compliance, digest). model_name/version/repo_* można nadpisać env."""
    artifact_sha256 = _sha256_file(model_path)
    repo_commit = resolve_repo_commit(repo_root, model_path)
    # Filtrujemy do istniejących plików — build_local_model_manifest wywala się na braku pliku.
    existing_impl = [Path(p) for p in implementation_files if Path(p).exists()]
    manifest = build_local_model_manifest(
        repo_root=repo_root,
        implementation_files=existing_impl,
        defaults={
            "model_name": os.getenv("POKER44_MODEL_NAME", model_name),
            "model_version": os.getenv("POKER44_MODEL_VERSION", str(model_version)),
            "framework": framework,
            "license": "MIT",
            "repo_url": os.getenv("POKER44_MODEL_REPO_URL", DEFAULT_REPO_URL),
            "repo_commit": repo_commit,
            "artifact_sha256": artifact_sha256,
            "notes": "XGBoost + quantile domain adaptation (artefakt per model).",
            "open_source": True,
            "inference_mode": "local",
            "training_data_statement": (
                "Trenowany na wydanym benchmarku Poker44 (API, z etykietami) oraz na "
                "syntetycznych danych human/bot. Nie uzywa prywatnych etykiet ewaluacyjnych walidatora."
            ),
            "training_data_sources": ["poker44-benchmark", "synthetic"],
            "private_data_attestation": (
                "Nie trenuje na prywatnych etykietach ewaluacyjnych walidatora. Nieetykietowane "
                "rece produkcyjne sluza wylacznie do dopasowania nienadzorowanego QuantileTransformer "
                "(adaptacja rozkladu cech)."
            ),
        },
    )
    return manifest, evaluate_manifest_compliance(manifest), manifest_digest(manifest)
