"""Export the South America (SAM) VTEC series from annual grid arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extract_rec_fixed import annual_timestamps, grid_coordinates


def export(npz_dir, template, out_dir):
    latitude, longitude = grid_coordinates()
    mask = (longitude >= -85) & (longitude <= -35) & (latitude >= -55) & (latitude <= 15)
    weights = np.cos(np.deg2rad(latitude[mask]))
    times = []
    means = []
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No annual NPZ files in {npz_dir}")
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            grid = archive["sequence_array"]
        if grid.ndim != 2 or grid.shape[1] != len(latitude):
            raise ValueError(f"{path.name} must have 5,183 grid columns")
        values = grid[:, mask]
        numerator = np.nansum(values * weights, axis=1)
        denominator = np.nansum(np.where(np.isnan(values), 0, weights), axis=1)
        means.append(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0))
        times.append(annual_timestamps(path, len(grid)).values)

    series = pd.DataFrame({"VTEC_True": np.concatenate(means)},
                          index=pd.DatetimeIndex(np.concatenate(times)))
    index_data = pd.read_csv(template)
    index_data["Time"] = pd.to_datetime(index_data["Time"])
    auxiliary = index_data.drop(columns=["VTEC_True", "Math_Base", "R_Target"], errors="ignore")
    aligned = series.join(auxiliary.set_index("Time"), how="inner")
    aligned["VTEC_True"] = aligned["VTEC_True"].interpolate(method="linear", limit_direction="both")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "SAM_REC_Raw.csv"
    aligned.to_csv(path, mode="x")
    print(f"{path.name}: {len(aligned):,} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--template-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.npz_dir, args.template_csv, args.out_dir)


if __name__ == "__main__":
    main()
