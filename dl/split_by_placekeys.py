#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split all_4cities_vbd3w.csv into per-city train/test sets using test placekeys.

Usage examples:
  # Using a directory of text files (one placekey per line). City name comes from filename.
  python split_by_placekeys.py \
    --input all_4cities_vbd3w.csv \
    --placekeys-dir test_placekeys/ \
    --outdir out_splits/

  # Using a single mapping CSV with columns: city,placekey
  python split_by_placekeys.py \
    --input all_4cities_vbd3w.csv \
    --mapping-csv test_placekeys_mapping.csv \
    --outdir out_splits/

Notes:
- Streams the input CSV (no pandas) to handle large files.
- Writes {city}_test.csv and {city}_train.csv in --outdir.
- Matches strictly on columns: 'city' and 'placekey'.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

def load_test_placekeys_from_dir(dir_path: Path) -> Dict[str, Set[str]]:
    """
    Load per-city test placekeys from a directory containing text files.
    Each file: one placekey per line (headerless). City name = filename stem (case preserved).
    Accepts .txt or .csv; if a line contains a comma and includes a header 'placekey',
    it will try to parse that column; otherwise treats each nonempty line as a placekey string.
    """
    by_city: Dict[str, Set[str]] = {}
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Placekeys directory not found: {dir_path}")

    files = [p for p in dir_path.iterdir() if p.suffix.lower() in (".txt", ".csv")]
    if not files:
        raise FileNotFoundError(f"No .txt or .csv files found in {dir_path}")

    for f in files:
        city = f.stem.strip()  # filename (without extension) is the city name
        city_set: Set[str] = set()
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            # Peek the first line
            first = fh.readline()
            if not first:
                continue
            # Reset file pointer
            fh.seek(0)

            # Heuristic: CSV with a header containing 'placekey'
            if ("," in first or "\t" in first) and ("placekey" in first.lower()):
                dialect = csv.Sniffer().sniff(first, delimiters=",\t;|")
                reader = csv.DictReader(fh, dialect=dialect)
                # Find a column named 'placekey' (case-insensitive)
                pk_col = None
                for c in reader.fieldnames or []:
                    if c.lower().strip() == "placekey":
                        pk_col = c
                        break
                if not pk_col:
                    # Fallback: treat lines as raw placekeys
                    fh.seek(0)
                    for line in fh:
                        s = line.strip()
                        if s and s.lower() != "placekey":
                            city_set.add(s)
                else:
                    for row in reader:
                        val = (row.get(pk_col) or "").strip()
                        if val:
                            city_set.add(val)
            else:
                # Treat as simple one placekey per line (ignore empty lines and a literal 'placekey' header)
                for line in fh:
                    s = line.strip()
                    if s and s.lower() != "placekey":
                        city_set.add(s)

        if city_set:
            by_city[city] = city_set

    if not by_city:
        raise ValueError(f"No placekeys loaded from directory: {dir_path}")
    return by_city


def load_test_placekeys_from_mapping_csv(mapping_csv: Path) -> Dict[str, Set[str]]:
    """
    Load a single CSV with columns: city, placekey (case-insensitive header match).
    """
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    by_city: Dict[str, Set[str]] = {}
    with mapping_csv.open("r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        reader = csv.DictReader(fh, dialect=dialect)

        # Find columns
        city_col = None
        pk_col = None
        for c in reader.fieldnames or []:
            cl = c.lower().strip()
            if cl == "city":
                city_col = c
            elif cl == "placekey":
                pk_col = c
        if not city_col or not pk_col:
            raise ValueError("Mapping CSV must contain headers 'city' and 'placekey'.")

        for row in reader:
            city = (row.get(city_col) or "").strip()
            pk = (row.get(pk_col) or "").strip()
            if city and pk:
                by_city.setdefault(city, set()).add(pk)

    if not by_city:
        raise ValueError(f"No (city, placekey) pairs found in {mapping_csv}")
    return by_city


def open_writers_for_cities(
    outdir: Path,
    cities: Set[str],
    header: list
) -> Dict[Tuple[str, str], csv.DictWriter]:
    """
    Create csv writers for each city's train/test files.
    Returns a dict keyed by (city, split) -> writer
    where split is 'train' or 'test'.
    """
    writers: Dict[Tuple[str, str], csv.DictWriter] = {}
    outdir.mkdir(parents=True, exist_ok=True)
    for city in sorted(cities):
        for split in ("train", "test"):
            safe_city = city.replace("/", "_")
            out_path = outdir / f"{safe_city}_{split}.csv"
            fh = open(out_path, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            writers[(city, split)] = (w, fh)  # store writer and handle
    return writers


def close_writers(writers: Dict[Tuple[str, str], csv.DictWriter]) -> None:
    for key, pair in writers.items():
        # pair is (writer, filehandle)
        _, fh = pair
        try:
            fh.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Split per-city train/test from a big CSV using test placekeys")
    ap.add_argument("--input", required=True, help="Path to all_4cities_vbd3w.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for per-city {train,test}.csv")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--placekeys-dir", help="Directory containing per-city placekey text files")
    group.add_argument("--mapping-csv", help="Single CSV with columns: city,placekey")

    ap.add_argument("--city-col", default="city_norm", help="City column name in the big CSV (default: city)")
    ap.add_argument("--placekey-col", default="placekey", help="Placekey column name (default: placekey)")
    ap.add_argument("--dialect", choices=["auto","comma","tab","pipe","semi"], default="auto",
                    help="CSV dialect for the big CSV (default: auto)")
    args = ap.parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)

    if not input_csv.exists():
        sys.exit(f"[ERROR] Input CSV not found: {input_csv}")

    # Load test placekeys per city
    if args.placekeys_dir:
        by_city = load_test_placekeys_from_dir(Path(args.placekeys_dir))
    else:
        by_city = load_test_placekeys_from_mapping_csv(Path(args.mapping_csv))

    all_cities = set(by_city.keys())
    print(f"[INFO] Loaded test placekeys for {len(all_cities)} cities:")
    for c in sorted(all_cities):
        print(f"  - {c}: {len(by_city[c])} test placekeys")

    # Prepare to read big CSV
    # Dialect detection
    if args.dialect == "auto":
        with input_csv.open("r", encoding="utf-8", errors="ignore") as fh:
            sample = fh.read(8192)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except Exception:
                dialect = csv.getdialect("excel")
    else:
        if args.dialect == "comma":
            class _D(csv.Dialect):
                delimiter=","
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        elif args.dialect == "tab":
            class _D(csv.Dialect):
                delimiter="\t"
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        elif args.dialect == "pipe":
            class _D(csv.Dialect):
                delimiter="|"
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        elif args.dialect == "semi":
            class _D(csv.Dialect):
                delimiter=";"
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        dialect = _D  # type: ignore

    # First pass: get header, validate columns
    with input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        header = reader.fieldnames
        if not header:
            sys.exit("[ERROR] Could not read header from input CSV.")

        # Normalize lookup columns
        city_col = None
        pk_col = None
        for c in header:
            cl = c.lower().strip()
            if cl == args.city_col.lower():
                city_col = c
            if cl == args.placekey_col.lower():
                pk_col = c

        if not city_col or not pk_col:
            sys.exit(f"[ERROR] Required columns not found. "
                     f"Looked for city='{args.city_col}', placekey='{args.placekey_col}' in header: {header}")

    # Open per-city train/test writers
    writers_raw = open_writers_for_cities(outdir, all_cities, header)
    # Convert to a simpler mapping: (city, split) -> writer only
    writers = {(k[0], k[1]): v[0] for k, v in writers_raw.items()}

    # Counters
    counts = {city: {"train": 0, "test": 0} for city in all_cities}
    total_rows = 0
    total_matched_cities = 0

    # Stream rows and route
    with input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        for row in reader:
            total_rows += 1
            city_val = (row.get(city_col) or "").strip()
            pk_val = (row.get(pk_col) or "").strip()
            if not city_val or not pk_val:
                continue

            # Only handle cities we have test placekeys for
            if city_val in by_city:
                total_matched_cities += 1
                if pk_val in by_city[city_val]:
                    writers[(city_val, "test")].writerow(row)
                    counts[city_val]["test"] += 1
                else:
                    writers[(city_val, "train")].writerow(row)
                    counts[city_val]["train"] += 1
            else:
                # City not in test list set — ignore (or you could collect separately if desired)
                pass

    # Close files
    close_writers(writers_raw)
    
    #!/usr/bin/env python3
"""
Split all_4cities_vbd3w.csv into per-city train/test sets using test placekeys.

Usage examples:
  # Using a directory of text files (one placekey per line). City name comes from filename.
  python split_by_placekeys.py \
    --input all_4cities_vbd3w.csv \
    --placekeys-dir test_placekeys/ \
    --outdir out_splits/

  # Using a single mapping CSV with columns: city,placekey
  python split_by_placekeys.py \
    --input all_4cities_vbd3w.csv \
    --mapping-csv test_placekeys_mapping.csv \
    --outdir out_splits/

Notes:
- Streams the input CSV (no pandas) to handle large files.
- Writes {city}_test.csv and {city}_train.csv in --outdir.
- Matches strictly on columns: 'city' and 'placekey'.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

def load_test_placekeys_from_dir(dir_path: Path) -> Dict[str, Set[str]]:
    """
    Load per-city test placekeys from a directory containing text files.
    Each file: one placekey per line (headerless). City name = filename stem (case preserved).
    Accepts .txt or .csv; if a line contains a comma and includes a header 'placekey',
    it will try to parse that column; otherwise treats each nonempty line as a placekey string.
    """
    by_city: Dict[str, Set[str]] = {}
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Placekeys directory not found: {dir_path}")

    files = [p for p in dir_path.iterdir() if p.suffix.lower() in (".txt", ".csv")]
    if not files:
        raise FileNotFoundError(f"No .txt or .csv files found in {dir_path}")

    for f in files:
        city = f.stem.strip()  # filename (without extension) is the city name
        city_set: Set[str] = set()
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            # Peek the first line
            first = fh.readline()
            if not first:
                continue
            # Reset file pointer
            fh.seek(0)

            # Heuristic: CSV with a header containing 'placekey'
            if ("," in first or "\t" in first) and ("placekey" in first.lower()):
                dialect = csv.Sniffer().sniff(first, delimiters=",\t;|")
                reader = csv.DictReader(fh, dialect=dialect)
                # Find a column named 'placekey' (case-insensitive)
                pk_col = None
                for c in reader.fieldnames or []:
                    if c.lower().strip() == "placekey":
                        pk_col = c
                        break
                if not pk_col:
                    # Fallback: treat lines as raw placekeys
                    fh.seek(0)
                    for line in fh:
                        s = line.strip()
                        if s and s.lower() != "placekey":
                            city_set.add(s)
                else:
                    for row in reader:
                        val = (row.get(pk_col) or "").strip()
                        if val:
                            city_set.add(val)
            else:
                # Treat as simple one placekey per line (ignore empty lines and a literal 'placekey' header)
                for line in fh:
                    s = line.strip()
                    if s and s.lower() != "placekey":
                        city_set.add(s)

        if city_set:
            by_city[city] = city_set

    if not by_city:
        raise ValueError(f"No placekeys loaded from directory: {dir_path}")
    return by_city


def load_test_placekeys_from_mapping_csv(mapping_csv: Path) -> Dict[str, Set[str]]:
    """
    Load a single CSV with columns: city, placekey (case-insensitive header match).
    """
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    by_city: Dict[str, Set[str]] = {}
    with mapping_csv.open("r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        reader = csv.DictReader(fh, dialect=dialect)

        # Find columns
        city_col = None
        pk_col = None
        for c in reader.fieldnames or []:
            cl = c.lower().strip()
            if cl == "city":
                city_col = c
            elif cl == "placekey":
                pk_col = c
        if not city_col or not pk_col:
            raise ValueError("Mapping CSV must contain headers 'city' and 'placekey'.")

        for row in reader:
            city = (row.get(city_col) or "").strip()
            pk = (row.get(pk_col) or "").strip()
            if city and pk:
                by_city.setdefault(city, set()).add(pk)

    if not by_city:
        raise ValueError(f"No (city, placekey) pairs found in {mapping_csv}")
    return by_city


def open_writers_for_cities(
    outdir: Path,
    cities: Set[str],
    header: list
) -> Dict[Tuple[str, str], csv.DictWriter]:
    """
    Create csv writers for each city's train/test files.
    Returns a dict keyed by (city, split) -> writer
    where split is 'train' or 'test'.
    """
    writers: Dict[Tuple[str, str], csv.DictWriter] = {}
    outdir.mkdir(parents=True, exist_ok=True)
    for city in sorted(cities):
        for split in ("train", "test"):
            safe_city = city.replace("/", "_")
            out_path = outdir / f"{safe_city}_{split}.csv"
            fh = open(out_path, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            writers[(city, split)] = (w, fh)  # store writer and handle
    return writers


def close_writers(writers: Dict[Tuple[str, str], csv.DictWriter]) -> None:
    for key, pair in writers.items():
        # pair is (writer, filehandle)
        _, fh = pair
        try:
            fh.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Split per-city train/test from a big CSV using test placekeys")
    ap.add_argument("--input", required=True, help="Path to all_4cities_vbd3w.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for per-city {train,test}.csv")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--placekeys-dir", help="Directory containing per-city placekey text files")
    group.add_argument("--mapping-csv", help="Single CSV with columns: city,placekey")

    ap.add_argument("--city-col", default="city", help="City column name in the big CSV (default: city)")
    ap.add_argument("--placekey-col", default="placekey", help="Placekey column name (default: placekey)")
    ap.add_argument("--dialect", choices=["auto","comma","tab","pipe","semi"], default="auto",
                    help="CSV dialect for the big CSV (default: auto)")
    args = ap.parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)

    if not input_csv.exists():
        sys.exit(f"[ERROR] Input CSV not found: {input_csv}")

    # Load test placekeys per city
    if args.placekeys_dir:
        by_city = load_test_placekeys_from_dir(Path(args.placekeys_dir))
    else:
        by_city = load_test_placekeys_from_mapping_csv(Path(args.mapping_csv))

    all_cities = set(by_city.keys())
    print(f"[INFO] Loaded test placekeys for {len(all_cities)} cities:")
    for c in sorted(all_cities):
        print(f"  - {c}: {len(by_city[c])} test placekeys")

    # Prepare to read big CSV
    # Dialect detection
    if args.dialect == "auto":
        with input_csv.open("r", encoding="utf-8", errors="ignore") as fh:
            sample = fh.read(8192)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except Exception:
                dialect = csv.getdialect("excel")
    else:
        if args.dialect == "comma":
            class _D(csv.Dialect):
                delimiter=","
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        elif args.dialect == "tab":
            class _D(csv.Dialect):
                delimiter="\t"
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        elif args.dialect == "pipe":
            class _D(csv.Dialect):
                delimiter="|"
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        elif args.dialect == "semi":
            class _D(csv.Dialect):
                delimiter=";"
                quotechar='"'
                doublequote=True
                lineterminator="\n"
                quoting=csv.QUOTE_MINIMAL
        dialect = _D  # type: ignore

    # First pass: get header, validate columns
    with input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        header = reader.fieldnames
        if not header:
            sys.exit("[ERROR] Could not read header from input CSV.")

        # Normalize lookup columns
        city_col = None
        pk_col = None
        for c in header:
            cl = c.lower().strip()
            if cl == args.city_col.lower():
                city_col = c
            if cl == args.placekey_col.lower():
                pk_col = c

        if not city_col or not pk_col:
            sys.exit(f"[ERROR] Required columns not found. "
                     f"Looked for city='{args.city_col}', placekey='{args.placekey_col}' in header: {header}")

    # Open per-city train/test writers
    writers_raw = open_writers_for_cities(outdir, all_cities, header)
    # Convert to a simpler mapping: (city, split) -> writer only
    writers = {(k[0], k[1]): v[0] for k, v in writers_raw.items()}

    # Counters
    counts = {city: {"train": 0, "test": 0} for city in all_cities}
    total_rows = 0
    total_matched_cities = 0

    # Stream rows and route
    with input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        for row in reader:
            total_rows += 1
            city_val = (row.get(city_col) or "").strip()
            pk_val = (row.get(pk_col) or "").strip()
            if not city_val or not pk_val:
                continue

            # Only handle cities we have test placekeys for
            if city_val in by_city:
                total_matched_cities += 1
                if pk_val in by_city[city_val]:
                    writers[(city_val, "test")].writerow(row)
                    counts[city_val]["test"] += 1
                else:
                    writers[(city_val, "train")].writerow(row)
                    counts[city_val]["train"] += 1
            else:
                # City not in test list set — ignore (or you could collect separately if desired)
                pass

    # Close files
    close_writers(writers_raw)

    # Summary
    print("\n[SUMMARY]")
    print(f"Input rows read: {total_rows}")
    print(f"Rows with city in provided test sets: {total_matched_cities}")
    for c in sorted(all_cities):
        tr = counts[c]["train"]
        te = counts[c]["test"]
        print(f"  {c}: train={tr:,}  test={te:,}  (test_placekeys={len(by_city[c]):,})")
    print(f"\nDone. Outputs written to: {outdir.resolve()}")
    

if __name__ == "__main__":
    main()
