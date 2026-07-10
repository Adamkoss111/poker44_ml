#!/usr/bin/env python3
"""publish_all.py (my_solution2) — publikacja artefaktów minerów z ISOLOWANYM commitem.

Adaptacja scripts/publish_all.py do nowej strategii (na razie 1 miner: miner5,
docelowo do 3: aspik3/4/5). Skaluje się bez zmian — dopisz kolejne wpisy do MODELS.

Zasada (jak w oryginale):
  - dla KAŻDEGO modelu: wyczyść published/, skopiuj TYLKO ten model, zrób commit
    -> commit tego minera ma w drzewie wyłącznie jego model (repo_commit rozłączny).
  - zapamiętaj commit każdego modelu w published/commit_map.json (klucz = nazwa
    pliku w published/, np. 'model5.joblib'); miner5 odczyta z niej SWÓJ repo_commit
    przez manifest_utils.resolve_repo_commit.
  - NA KONIEC: wszystkie modele trafiają do published/ (runtime) + commit_map.json.

Źródła: artefakty z train_fusion.py leżą w my_solution2/fusion_models/model/<tag>.joblib.

MAPOWANIE (edytuj tutaj): published_name -> ścieżka artefaktu źródłowego.
  published_name MUSI zgadzać się z POKER44_MODEL_FILE minera (miner5 -> model5.joblib).

Push jest RĘCZNY (nie robimy git push za Ciebie). Skrypt zatrzymuje się po
commitach i wypisuje komendę do wypchnięcia. Użyj --push tylko jeśli świadomie
chcesz wypchnąć automatycznie.

Użycie:
    python my_solution2/publish_all.py            # commituje lokalnie, pokazuje 'git push'
    python my_solution2/publish_all.py --push     # commituje i pcha (świadomie)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "my_solution2"
SRC_DIR = SOLUTION / "fusion_models" / "model"
DST = ROOT / "published"
MAP = DST / "commit_map.json"

# published_name (== POKER44_MODEL_FILE minera) -> artefakt źródłowy w SRC_DIR
# Na razie jeden miner. ZMIEŃ 'S1_core.joblib' na artefakt, który faktycznie
# ma jechać jako miner5 (patrz my_solution2/fusion_models/model/).
MODELS = {
    "model5.joblib": SRC_DIR / "S1_core.joblib",
}


def git(*args, capture=False):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=capture, text=True)


def _clear_published():
    """Czysta karta pod jeden commit: usuń modele + mapę z published/."""
    if DST.exists():
        for f in DST.glob("*.joblib"):
            f.unlink()
    if MAP.exists():
        MAP.unlink()


def main():
    do_push = "--push" in sys.argv[1:]

    missing = {name: src for name, src in MODELS.items() if not Path(src).exists()}
    if missing:
        lines = "\n".join(f"  {n} -> {s}" for n, s in missing.items())
        sys.exit(f"Brak artefaktów źródłowych:\n{lines}\n"
                 f"Sprawdź MODELS w {Path(__file__).name} i wytrenuj/wskaż właściwy .joblib.")
    DST.mkdir(exist_ok=True)

    commit_map = {}
    for name, src in MODELS.items():
        _clear_published()                       # commit będzie miał TYLKO ten model
        shutil.copyfile(src, DST / name)
        git("add", "-A", str(DST.relative_to(ROOT)))
        git("commit", "-q", "-m", f"model {name}")
        commit_map[name] = git("rev-parse", "HEAD", capture=True).stdout.strip()
        print(f"  commit '{name}' <- {Path(src).name} -> {commit_map[name][:12]}  "
              f"(w drzewie tylko ten model)")

    # KONIEC: wszystkie modele do published/ (runtime) + mapa commitów
    _clear_published()
    for name, src in MODELS.items():
        shutil.copyfile(src, DST / name)
    MAP.write_text(json.dumps(commit_map, indent=2, sort_keys=True) + "\n")
    git("add", "-A", str(DST.relative_to(ROOT)))
    git("commit", "-q", "-m", "all models for runtime + commit map")

    print(f"\n{len(commit_map)} model(i) opublikowanych lokalnie. Mapa: {MAP.relative_to(ROOT)}")
    if do_push:
        print("Push (--push)...")
        git("push")
        print("GOTOWE (wypchnięte).")
    else:
        branch = git("rev-parse", "--abbrev-ref", "HEAD", capture=True).stdout.strip()
        print("Commity gotowe lokalnie. Aby wypchnąć, uruchom RĘCZNIE:")
        print(f"    git push origin {branch}")


if __name__ == "__main__":
    main()
