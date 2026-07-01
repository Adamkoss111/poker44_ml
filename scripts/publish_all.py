#!/usr/bin/env python3
"""publish_all.py — każdy model dostaje OSOBNY commit zawierający TYLKO ten model.

Zasada:
  - źródło: models/  (gitignored, wszystkie 24)
  - pętla: kopiuj modelN do published/, ale NAJPIERW usuń poprzednie ->
           commit ma w drzewie TYLKO modelN (commit dla miner8 = tylko model8)
  - zapamiętaj commit każdego modelu (commit_map.json)
  - NA KONIEC: skopiuj WSZYSTKIE modele do published/ (żeby minery mogły je
    zaciągnąć/załadować w runtime) + zapisz mapę i zrób JEDEN push.

repo_commit minera N wskazuje na commit, gdzie w drzewie jest tylko modelN.
HEAD (stan końcowy) ma wszystkie 24 — do runtime.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models"
DST = ROOT / "published"
MAP = DST / "commit_map.json"


def git(*args, capture=False):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=capture, text=True)


def natural_key(p: Path):
    m = re.match(r"(stack|model)(\d+)", p.stem)
    if not m:
        return (2, 0, p.name)
    return (0 if m.group(1) == "model" else 1, int(m.group(2)), p.name)


def _clear_published():
    """Usuwa wszystkie modele + mapę z published/ (czysta karta pod jeden commit)."""
    for f in DST.glob("*.joblib"):
        f.unlink()
    if MAP.exists():
        MAP.unlink()


def main():
    if not SRC.is_dir():
        sys.exit(f"Brak {SRC}/ — wrzuc tam modele (model1..model12, stack1..stack12).")
    files = sorted(SRC.glob("*.joblib"), key=natural_key)
    if not files:
        sys.exit(f"Brak *.joblib w {SRC}/.")
    DST.mkdir(exist_ok=True)

    commit_map = {}
    for src in files:
        name = src.name
        _clear_published()                 # usun poprzednie -> commit bedzie mial TYLKO ten
        shutil.copyfile(src, DST / name)
        git("add", "-A", str(DST.relative_to(ROOT)))
        git("commit", "-q", "-m", f"model {name}")
        commit_map[name] = git("rev-parse", "HEAD", capture=True).stdout.strip()
        print(f"  commit '{name}' -> {commit_map[name][:12]}  (w drzewie tylko ten model)")

    # KONIEC: wszystkie modele do published/ (runtime) + mapa commitow
    _clear_published()
    for src in files:
        shutil.copyfile(src, DST / src.name)
    MAP.write_text(json.dumps(commit_map, indent=2, sort_keys=True) + "\n")
    git("add", "-A", str(DST.relative_to(ROOT)))
    git("commit", "-q", "-m", "all models for runtime + commit map")

    print(f"\n{len(commit_map)} modeli opublikowanych. Mapa: {MAP.relative_to(ROOT)}")
    print("Push...")
    git("push")
    print("GOTOWE. published/ ma wszystkie modele (runtime), a repo_commit kazdego "
          "minera wskazuje commit z TYLKO jego modelem.")


if __name__ == "__main__":
    main()
