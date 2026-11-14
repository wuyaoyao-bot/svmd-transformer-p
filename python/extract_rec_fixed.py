"""Export the regional VTEC series and global mean from annual grid arrays."""

from __future__ import annotations

import argparse
import calendar
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = {
    "EU": ((-10, 40), (35, 60)),
    "NA": ((-130, -60), (25, 55)),
    "EA": ((70, 140), (15, 55)),
    "SA": ((10, 50), (-40, -10)),
    "LL": ((15, 50), (-20, 20)),
}


def grid_coordinates():
    lats = np.arange(87.5, -87.6, -2.5)
    lons = np.arange(-180, 180.1, 5)
    return np.repeat(lats, len(lons)), np.tile(lons, len(lats))


def annual_timestamps(path, size):
    match = re.search(r"(19|20)\d{2}", path.name)
    if match is None:
        raise ValueError(f"Cannot determine the year from {path.name}")
    year = int(match.group())
    expected = 8784 if calendar.isleap(year) else 8760
    if size != expected:
        warnings.warn(f"{path.name} has {size} archive rows for {expected} calendar hours.", stacklevel=2)
    return pd.date_range(f"{year}-01-01 00:00:00", periods=size, freq="h")


def mean_vtec(grid, mask, latitude):
    selected = grid[:, mask]
    weights = np.cos(np.deg2rad(latitude[mask]))
    numerator = np.nansum(selected * weights, axis=1)
    denominator = (~np.isnan(selected) * weights).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def template_index(template):
    frame = pd.read_csv(template)
    by_lower = {str(name).lower(): name for name in frame.columns}
    candidates = ("datetime", "date_time", "time", "timestamp", "epoch", "date", "utc")
    time_column = next((by_lower[key] for key in candidates if key in by_lower), None)
    if time_column is None:
        first_column = frame.columns[0]
        try:
            pd.to_datetime(frame[first_column])
        except (ValueError, TypeError, OverflowError):
            pass
        else:
            time_column = first_column
    fields = [name for name in ("VTEC_True", "Math_Base", "R_Target") if name in frame.columns]
    if time_column is None:
        warnings.warn("The template has no recognized time column. Its row labels start at 2002-01-01.", stacklevel=2)
        result = frame.drop(columns=fields)
        result.index = pd.date_range("2002-01-01 00:00:00", periods=len(frame), freq="h")
    else:
        frame[time_column] = pd.to_datetime(frame[time_column])
        result = frame.drop(columns=fields).set_index(time_column)
    result.index.name = "Datetime"
    return result


def regional_series(npz_dir):
    latitude, longitude = grid_coordinates()
    masks = {}
    for region, (lon, lat) in REGIONS.items():
        masks[region] = ((longitude >= lon[0]) & (longitude <= lon[1])
                         & (latitude >= lat[0]) & (latitude <= lat[1]))
    # The date line is stored in both the first and last longitude columns.
    masks["GEC"] = longitude < 180
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No annual NPZ files in {npz_dir}")
    blocks = {region: [] for region in masks}
    timestamps = []
    for index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as archive:
            key = "sequence_array" if "sequence_array" in archive.files else archive.files[0]
            grid = archive[key]
        if grid.ndim != 2 or grid.shape[1] != len(latitude):
            raise ValueError(f"{path.name} must have 5,183 grid columns")
        if index == 0:
            annual = np.nanmean(grid, axis=0)
            if not np.nanmean(annual[np.abs(latitude) <= 20]) > np.nanmean(annual[np.abs(latitude) >= 60]):
                warnings.warn("Check the assumed latitude and longitude order in the annual grid.", stacklevel=2)
        timestamps.append(annual_timestamps(path, len(grid)).values)
        for region, mask in masks.items():
            blocks[region].append(mean_vtec(grid, mask, latitude))
    return pd.DataFrame(
        {region: np.concatenate(values) for region, values in blocks.items()},
        index=pd.DatetimeIndex(np.concatenate(timestamps), name="Datetime"),
    )


def export(npz_dir, template, out_dir, interpolate):
    series = regional_series(npz_dir)
    aligned = series.join(template_index(template), how="inner")
    auxiliary = [name for name in aligned.columns if name not in series]
    out_dir.mkdir(parents=True, exist_ok=True)
    for region in series:
        output = aligned[[region, *auxiliary]].rename(columns={region: "VTEC_True"})
        if interpolate and output["VTEC_True"].isna().any():
            output["VTEC_True"] = output["VTEC_True"].interpolate(method="linear", limit_direction="both")
        path = out_dir / f"{region}_REC_Raw.csv"
        output.to_csv(path, mode="x")
        print(f"{path.name}: {len(output):,} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--template-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep-nan", action="store_true", help="Keep missing values instead of interpolating.")
    args = parser.parse_args()
    export(args.npz_dir, args.template_csv, args.out_dir, not args.keep_nan)


if __name__ == "__main__":
    main()
