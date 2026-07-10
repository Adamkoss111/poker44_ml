"""
download_benchmark.py — automatyczne pobieranie danych benchmarku Poker44.

API (z repo vegasnc, potwierdzone): https://api.poker44.net/api/v1/benchmark
  GET /releases?limit=N          -> lista release'ów (dat)
  GET /chunks?sourceDate=D&limit=N[&split=S][&cursor=C]  -> rekordy, paginacja kursorem

Zapisuje per data do api_data/data_YYYY_MM_DD.json w formacie zgodnym z Twoim
istniejącym kodem liczenia cech:
    {"data": {"chunks": [ {chunks, groundTruth, split, sourceDate, ...}, ... ]}}
czyli data['data']['chunks'] -> lista payloadów jak dotychczas.

PRZYROSTOWO: daty, których plik już istnieje, są pomijane (chyba że --force).

Użycie:
    python download_benchmark.py                 # wszystkie nowe release'y
    python download_benchmark.py --days 5        # tylko 5 najnowszych dat
    python download_benchmark.py --force         # nadpisz istniejące
    python download_benchmark.py --features      # od razu policz cechy -> parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.poker44.net/api/v1/benchmark"
OUT_DIR = Path("api_data")
TIMEOUT = 30
MAX_RETRIES = 3
PAGE_LIMIT = 100


def _get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Nie udało się pobrać {url} (params={params}): {last_err}")


def list_release_dates(session: requests.Session, limit: int = 60) -> list[str]:
    payload = _get_json(session, f"{BASE_URL}/releases", {"limit": limit})
    data = payload.get("data", payload)
    releases = data.get("releases", [])
    dates = []
    for rel in releases:
        d = rel.get("sourceDate") or rel.get("source_date") or rel.get("date")
        if d:
            dates.append(str(d))
    return sorted(set(dates))


def download_date(session: requests.Session, source_date: str,
                  split: str | None = None) -> list[dict]:
    """Pobiera WSZYSTKIE rekordy danej daty (paginacja kursorem)."""
    records: list[dict] = []
    cursor = None
    page = 0
    while True:
        params: dict = {"sourceDate": source_date}
        if split:
            params["split"] = split
        if cursor:
            params["cursor"] = cursor
        payload = _get_json(session, f"{BASE_URL}/chunks", params)
        data = payload.get("data", payload)
        page_records = data.get("chunks", data if isinstance(data, list) else [])
        records.extend(page_records)
        page += 1
        cursor = (data.get("nextCursor") or data.get("cursor")
                  or data.get("next_cursor"))
        if not cursor:
            break
    print(f"  {source_date}: {len(records)} rekordów ({page} stron)")
    return records


def save_date_file(source_date: str, records: list[dict], out_dir: Path) -> Path:
    """Zapis w formacie zgodnym z istniejącym kodem: data['data']['chunks']."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"data_{source_date.replace('-', '_')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({"data": {"chunks": records}}, f)
    return fname


def build_features(files: list[Path], out_parquet: str = "df.parquet") -> None:
    """Opcjonalnie: policz cechy (pakiet features/) dla pobranych plików."""
    import pandas as pd
    from features import compute_features

    rows = []
    for path in sorted(files):
        data = json.load(open(path, encoding="utf-8"))
        file_date = "-".join(path.stem.split("_")[1:])
        for mc in data["data"]["chunks"]:
            chunks = mc.get("chunks") or []
            labels = mc.get("groundTruth") or mc.get("ground_truth") or []
            if len(chunks) != len(labels):
                print(f"  UWAGA {path.name}: {len(chunks)} chunks vs {len(labels)} labels — pomijam rekord")
                continue
            date = mc.get("windowStart") or mc.get("sourceDate") or file_date
            for label, chunk in zip(labels, chunks):
                feats = compute_features(chunk)
                if not feats:
                    continue
                feats["label"] = int(label)
                feats["data"] = str(date)
                rows.append(feats)
        print(f"  cechy: {path.name} OK (łącznie {len(rows)} wierszy)")
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(out_parquet)
        print(f"Zapisano {out_parquet}: {df.shape[0]} wierszy x {df.shape[1]} kolumn "
              f"| balans={df['label'].mean():.2f}")
    except ImportError:
        alt = str(Path(out_parquet).with_suffix(".pkl"))
        df.to_pickle(alt)
        print(f"(brak pyarrow/fastparquet) Zapisano {alt}: {df.shape[0]} x {df.shape[1]} "
              f"| balans={df['label'].mean():.2f}")


def main():
    ap = argparse.ArgumentParser(description="Pobieranie benchmarku Poker44")
    ap.add_argument("--days", type=int, default=None,
                    help="ile najnowszych dat pobrać (domyślnie: wszystkie dostępne)")
    ap.add_argument("--split", default=None, choices=[None, "train", "validation"],
                    help="opcjonalny filtr splitu API")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true", help="nadpisz istniejące pliki")
    ap.add_argument("--features", action="store_true",
                    help="po pobraniu policz cechy do df.parquet (wymaga pakietu features/)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    session = requests.Session()

    print("Pobieram listę release'ów...")
    dates = list_release_dates(session)
    if not dates:
        sys.exit("Brak release'ów w API — sprawdź endpoint/sieć.")
    if args.days:
        dates = dates[-args.days:]
    print(f"{len(dates)} dat: {dates[0]} .. {dates[-1]}")

    saved_files: list[Path] = []
    for d in dates:
        fname = out_dir / f"data_{d.replace('-', '_')}.json"
        if fname.exists() and not args.force:
            print(f"  {d}: już pobrane, pomijam ({fname.name})")
            saved_files.append(fname)
            continue
        records = download_date(session, d, split=args.split)
        if records:
            saved_files.append(save_date_file(d, records, out_dir))

    print(f"\nGotowe: {len(saved_files)} plików w {out_dir}/")

    if args.features:
        print("\nLiczę cechy...")
        build_features(saved_files)


if __name__ == "__main__":
    main()
