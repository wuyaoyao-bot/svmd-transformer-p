"""Extract the nine manuscript locations from the 2002–2023 annual NPZ archive.

The time column is a nominal row-based calendar for calendar-sized years.
Years with a row-count discrepancy retain every source row and a blank time.
Run this beside code_gim_1h_pilot_v1.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from code_gim_1h_pilot_v1 import SITES, stamp


YEARS = range(2002, 2024)
MEMBER = "sequence_array.npy"
BLOCK_ROWS = 256
SITE_COUNT = len(SITES)


def grid_mapping():
    latitudes = [87.5 - 2.5 * index for index in range(71)]
    longitudes = [-180.0 + 5.0 * index for index in range(73)]
    mapping = {}
    for site, (latitude, longitude) in SITES.items():
        i = min(range(len(latitudes)), key=lambda index: abs(latitudes[index] - latitude))
        j = min(range(len(longitudes)), key=lambda index: abs(longitudes[index] - longitude))
        mapping[site] = dict(
            latitude=latitude, longitude=longitude, grid_lat=latitudes[i], grid_lon=longitudes[j],
            flattened_column_0based=i * len(longitudes) + j,
        )
    return mapping


def split_for_year(year):
    if year <= 2017:
        return "train"
    if year <= 2020:
        return "validation"
    return "test"


def read_array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NumPy array header version {version}")
    if (
        len(shape) != 2 or shape[0] <= 0 or shape[1] != 71 * 73
        or fortran_order or dtype.kind not in "fiu"
    ):
        raise ValueError(f"Expected a numeric C-order N×5183 array, found {shape} {dtype}")
    return shape, dtype


def extract_year(path, year, columns, writer):
    first = datetime(year, 1, 1, tzinfo=timezone.utc)
    expected = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) - first).days * 24
    split = split_for_year(year)
    digest = hashlib.sha256()
    clipped_count = 0
    missing_count = 0
    with zipfile.ZipFile(path) as archive, archive.open(MEMBER) as stream:
        shape, dtype = read_array_header(stream)
        rows, grid_cells = shape
        calendar_size = rows == expected
        time_basis = "nominal_row_time" if calendar_size else "unresolved_row_count"
        for start in range(0, rows, BLOCK_ROWS):
            count = min(BLOCK_ROWS, rows - start)
            byte_count = count * grid_cells * dtype.itemsize
            payload = stream.read(byte_count)
            if len(payload) != byte_count:
                raise ValueError(f"Array ends early in {path.name} at row {start}")
            digest.update(payload)
            block = np.frombuffer(payload, dtype=dtype).reshape(count, grid_cells)
            selected = block[:, columns].astype(np.float64)
            finite = np.isfinite(selected)
            negative = finite & (selected < 0)
            clipped_count += int(negative.sum())
            missing_count += int((~finite).sum())
            selected[negative] = 0.0
            for offset, values in enumerate(selected.tolist()):
                source_row = start + offset
                nominal = stamp(first + timedelta(hours=source_row)) if calendar_size else ""
                clean = [value if math.isfinite(value) else "" for value in values]
                writer.writerow([year, source_row, split, nominal, time_basis, *clean])
        if stream.read(1):
            raise ValueError(f"Array has unexpected trailing bytes in {path.name}")
    return dict(
        year=year, split=split, source_file=path.name, rows=rows,
        expected_hourly_rows=expected, row_difference=rows - expected, time_basis=time_basis,
        negative_values_set_to_zero=clipped_count, nonfinite_values_written_blank=missing_count,
        source_dtype=dtype.str, source_array_payload_sha256=digest.hexdigest(),
    )


def prepare(npz_dir, out):
    paths = {year: npz_dir / f"sequence_array_{year}.npz" for year in YEARS}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing annual NPZ files in {npz_dir} ({', '.join(missing)})")
    if out.exists():
        raise ValueError(f"Output directory already exists. Choose a new --out directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    mapping = grid_mapping()
    columns = [item["flattened_column_0based"] for item in mapping.values()]
    summary = []
    with tempfile.TemporaryDirectory(prefix="gim_archive_build_", dir=out.parent) as temporary:
        staged = Path(temporary) / "archive"
        staged.mkdir()
        with (staged / "vtec_9locations_2002_2023.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["year", "source_row_0based", "split", "nominal_utc", "time_basis", *SITES])
            for index, (year, path) in enumerate(paths.items(), 1):
                try:
                    item = extract_year(path, year, columns, writer)
                except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
                    raise ValueError(f"Could not extract {path.name} ({exc})") from exc
                summary.append(item)
                print(
                    f"[{index:02d}/{len(YEARS)}] {year} | rows={item['rows']} "
                    f"| expected={item['expected_hourly_rows']} | {item['time_basis']}",
                    flush=True,
                )

        with (staged / "year_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        config = dict(
            source_directory=str(npz_dir),
            years=[YEARS.start, YEARS.stop - 1],
            split_years=dict(train=[2002, 2017], validation=[2018, 2020], test=[2021, 2023]),
            grid_mapping=mapping,
            value_units="TECU as stored in the legacy processed archive",
            preprocessing="Finite negative values become zero. Nonfinite values are blank.",
            time_policy=(
                "A nominal calendar assigns January 1 00:00 UTC plus the source row number in hours "
                "when the annual count matches the calendar. Original map epochs remain unverified. "
                "Years with a count discrepancy have blank nominal_utc values."
            ),
            total_rows=sum(item["rows"] for item in summary),
            row_count_discrepancy_years=[item["year"] for item in summary if item["row_difference"]],
        )
        (staged / "archive_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        staged.rename(out)

    counts = {
        split: sum(item["rows"] for item in summary if item["split"] == split)
        for split in ("train", "validation", "test")
    }
    flagged = config["row_count_discrepancy_years"]
    print(f"ARCHIVE PASS | years={len(YEARS)} | locations={SITE_COUNT} | rows={config['total_rows']}")
    print(f"Raw rows | train={counts['train']} validation={counts['validation']} test={counts['test']}")
    print(f"Unresolved time axis years: {', '.join(map(str, flagged)) or 'none by row count'}")
    print(f"Results: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.npz_dir.resolve(), args.out.resolve())
    except (OSError, ValueError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
