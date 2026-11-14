"""Compare direct lead times on matched regional test hours using saved SVMD features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import code_gim_review_regional_table5_v1 as regional


pilot, baseline, pmsc = regional.pilot, regional.baseline, regional.pmsc
MODELS = regional.MODELS
FIXED_CAP, DST_SENSITIVITY = pmsc.FIXED_CAP, pmsc.DST_SENSITIVITY


def verify_parent(parent, sources, dst_hashes):
    path = parent / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    modules = (pilot, baseline, regional.svmd, pmsc, regional.storms, regional.storms.fourier)
    expected = dict(
        script_sha256=pilot.sha256_file(Path(regional.__file__)),
        code_sha256={Path(m.__file__).name: pilot.sha256_file(Path(m.__file__)) for m in modules},
        dst_sha256=dst_hashes, regions=list(regional.REGIONS), split_years=baseline.SPLIT_YEARS,
        excluded_years=[regional.OMITTED_YEAR], history_hours=baseline.HISTORY,
        lead_hours=1, cross_year_windows=False, train_validation_stride=baseline.TRAIN_STRIDE,
        models=list(MODELS), arms=[list(arm) for arm in regional.ARMS], seed=pilot.SEED,
        base_cap=regional.BASE_CAP, fixed_cap=FIXED_CAP, dst_sensitivity=DST_SENSITIVITY,
        phys_weight=pmsc.PHYS_WEIGHT,
    )
    changed = [name for name, value in expected.items() if config.get(name) != value]
    changed.extend(region for region, source in sources.items()
                   if config["input_series"][region]["sha256"] != source["sha256"])
    if changed or not (parent / "table5_rmse.csv").is_file():
        raise ValueError(f"Need the matching completed regional 1 h run. Differences: {changed}")
    return pilot.sha256_file(path)


def index_saved_samples(values, dst_years, column, data, test_keys):
    keys = {"test": test_keys}
    history, stride = baseline.HISTORY, baseline.TRAIN_STRIDE
    for split in ("train", "validation"):
        blocks = []
        first, last = baseline.SPLIT_YEARS[split]
        for year in range(first, last + 1):
            if year not in dst_years:
                continue
            vtec = values[year][:, column]
            if len(vtec) <= history:
                continue
            vtec_windows = np.lib.stride_tricks.sliding_window_view(vtec, history + 1)[::stride]
            dst_windows = np.lib.stride_tricks.sliding_window_view(dst_years[year], history + 1)[::stride, :-1]
            keep = np.isfinite(vtec_windows).all(axis=1) & np.isfinite(dst_windows).all(axis=1)
            hours = np.arange(history, len(vtec), stride)[keep]
            blocks.append(np.column_stack((np.full(len(hours), year), hours)))
        keys[split] = np.concatenate(blocks, axis=0)

    for split, index in keys.items():
        _, truth = shifted_targets(values, column, index, 1)
        if not np.array_equal(truth, data[split]["truth"]):
            raise ValueError(f"Archived feature order disagrees with {split} targets")
    return keys


def shifted_targets(values, column, keys, lead):
    targets = keys.astype(np.int64, copy=True)
    targets[:, 1] += lead - 1
    truth = np.full(len(targets), np.nan)
    for year in np.unique(targets[:, 0]):
        vtec = values[int(year)][:, column]
        keep = (targets[:, 0] == year) & (targets[:, 1] < len(vtec))
        truth[keep] = vtec[targets[keep, 1]]
    return targets, truth


def matched_test(values, column, keys, leads):
    positions = {}
    for lead in leads:
        targets, truth = shifted_targets(values, column, keys, lead)
        positions[lead] = {tuple(targets[i]): i for i in np.flatnonzero(np.isfinite(truth))}
    common = sorted(set.intersection(*(set(index) for index in positions.values())))
    if {year for year, _ in common} != set(baseline.TEST_YEARS):
        raise ValueError(f"Missing matched target hours in a test year for {regional.REGIONS[column]}")
    return np.asarray(common), {lead: np.asarray([index[key] for key in common])
                                for lead, index in positions.items()}


def attach_features(parent, region, data):
    size = sum(len(rows["truth"]) for rows in data.values())
    residual, base = pmsc.read_features(pmsc.feature_paths(parent, region, size), size)
    start = 0
    for rows in data.values():
        stop = start + len(rows["truth"])
        rows.update(residual=residual[start:stop], base=base[start:stop])
        start = stop


def lead_samples(values, column, data, keys, lead, test_positions):
    result = {}
    for split, rows in data.items():
        _, truth = shifted_targets(values, column, keys[split], lead)
        take = test_positions if split == "test" else np.flatnonzero(np.isfinite(truth))
        if not len(take):
            raise ValueError(f"No {split} samples for {regional.REGIONS[column]} at {lead} h")
        selected = {name: rows[name][take] for name in ("raw", "origin", "dst", "residual")}
        selected.update(truth=truth[take], base=selected["origin"] + lead * (rows["base"][take] - selected["origin"]))
        result[split] = selected
    return result


def identity(test, keys):
    return dict(keys=keys, truth=test["truth"], origin=test["origin"], dst_t0=test["dst"][:, -1])


def read_forecasts(path, expected):
    return regional.storms.fourier.aligned_forecasts(path, (*expected, *MODELS), expected, MODELS)


def fit_lead(torch, device, data, keys, lead, folder):
    expected = identity(data["test"], keys)
    if folder.is_dir():
        print(f"{folder.parent.name} | lead={lead} h | reusing completed forecasts", flush=True)
        return read_forecasts(folder / "predictions.npz", expected)
    counts = " | ".join(f"{split}={len(rows['truth'])}" for split, rows in data.items())
    print(f"{folder.parent.name} | lead={lead} h | {counts}", flush=True)
    folder.parent.mkdir(parents=True, exist_ok=True)
    result = dict(expected)
    with tempfile.TemporaryDirectory(prefix="fit_lead_", dir=folder.parent) as temporary:
        staged = Path(temporary) / "completed"
        staged.mkdir()
        raw = {split: (rows["raw"], rows["truth"]) for split, rows in data.items()}
        reference = baseline.fit_site(torch, device, raw, keys, staged / "baselines")
        result.update({name: reference[name] for name in baseline.MODELS})
        previous = (pmsc.BASE_CAP, pmsc.FIXED_CAP, pmsc.DST_SENSITIVITY)
        try:
            pmsc.BASE_CAP = lead * regional.BASE_CAP
            pmsc.FIXED_CAP = lead * FIXED_CAP
            pmsc.DST_SENSITIVITY = lead * DST_SENSITIVITY
            for name, branch, constraint in regional.ARMS:
                result[name], state = pmsc.fit_arm(torch, device, name, data, branch, constraint)
                state.update(lead_hours=lead, base_cap_tecu=pmsc.BASE_CAP, fixed_cap_tecu=pmsc.FIXED_CAP,
                             dst_sensitivity=pmsc.DST_SENSITIVITY, phys_weight=pmsc.PHYS_WEIGHT)
                torch.save(state, staged / f"{name}.pt")
        finally:
            pmsc.BASE_CAP, pmsc.FIXED_CAP, pmsc.DST_SENSITIVITY = previous
        np.savez_compressed(staged / "predictions.npz", **result)
        staged.rename(folder)
    print(f"{folder.parent.name} | lead={lead} h | PASS", flush=True)
    return result


def annual_rows(results):
    years = tuple(baseline.TEST_YEARS)
    for (region, lead), data in results.items():
        for period in (*years, f"{years[0]}-{years[-1]}"):
            keep = data["keys"][:, 0] == period if isinstance(period, int) else slice(None)
            truth = data["truth"][keep]
            hours = sum(int((regional.YEARS[y + 1] - regional.YEARS[y]) / pilot.HOUR)
                        for y in ((period,) if isinstance(period, int) else years))
            yield dict(region=region, period=period, lead_hours=lead, scored_hours=len(truth), calendar_hours=hours,
                       **{f"{name}_rmse_tecu": baseline.error_metrics(data[name][keep] - truth)[0] for name in MODELS})


def prediction_rows(results):
    for (region, lead), data in results.items():
        for i, (year, hour) in enumerate(data["keys"]):
            target = regional.YEARS[int(year)] + int(hour) * pilot.HOUR
            yield dict(region=region, lead_hours=lead,
                       nominal_origin_utc=pilot.stamp(target - lead * pilot.HOUR), nominal_target_utc=pilot.stamp(target),
                       dst_t0_nt=data["dst_t0"][i], origin_tecu=data["origin"][i], actual_tecu=data["truth"][i],
                       **{name: data[name][i] for name in MODELS})


def run(args):
    import torch

    values, sources = regional.load_series(args.project_root)
    dst_years, dst_hashes = pmsc.read_dst_years(args.archive_dir, values, args.omni_dir)
    parent_hash = verify_parent(args.parent, sources, dst_hashes)
    saved_paths = {region: args.parent / "sites" / region / "predictions.npz" for region in regional.REGIONS}
    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    args.out.mkdir(parents=True, exist_ok=True)
    with baseline.run_lock(args.out):
        baseline.establish_config(args.out, dict(
            parent_sha256=parent_hash, parent_forecast_sha256={r: pilot.sha256_file(p) for r, p in saved_paths.items()},
            script_sha256=pilot.sha256_file(Path(__file__)), leads_hours=list(args.leads), models=list(MODELS),
            test_matching="same target hours across leads within each region and year",
            training_target="direct t0+h with 72 hourly inputs through t0", device=device_name,
            base_extrapolation="VTEC(t0) + h * (one-hour SVMD base - VTEC(t0))",
            fixed_cap_per_hour_tecu=FIXED_CAP, adaptive_base_per_hour_tecu=regional.BASE_CAP,
            dst_sensitivity_per_hour=DST_SENSITIVITY, phys_weight=pmsc.PHYS_WEIGHT,
            cap_rule="multiply the hourly threshold at origin by lead hours",
            torch_version=str(torch.__version__), numpy_version=np.__version__,
        ))
        results = {}
        for column, region in enumerate(regional.REGIONS):
            data, test_keys = regional.samples_for_region(values, dst_years, column)
            keys = index_saved_samples(values, dst_years, column, data, test_keys)
            common, positions = matched_test(values, column, test_keys, args.leads)
            saved = read_forecasts(saved_paths[region], identity(data["test"], test_keys))
            if any(lead > 1 for lead in args.leads):
                attach_features(args.parent, region, data)
            for lead in args.leads:
                take = positions[lead]
                if lead == 1:
                    result = {name: array[take] for name, array in saved.items()}
                else:
                    samples = lead_samples(values, column, data, keys, lead, take)
                    folder = args.out / "sites" / region / f"lead_{lead:02d}h"
                    result = fit_lead(torch, device, samples, common, lead, folder)
                results[region, lead] = result
            print(f"{region} | common test target hours={len(common)} | all leads ready", flush=True)
        table = list(annual_rows(results))
        pilot.write_csv(args.out / "annual_test_rmse.csv", list(table[0]), table)
        pilot.write_csv(args.out / "predictions.csv", list(next(prediction_rows(results))), prediction_rows(results))
    print(f"REGIONAL HORIZONS PASS | {len(table)} rows | {args.out / 'annual_test_rmse.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home() / "Desktop" / "SVMD+lstm")
    parser.add_argument("--archive-dir", type=Path, default=Path("results/archive_2002_2023_v1"))
    parser.add_argument("--omni-dir", type=Path)
    parser.add_argument("--parent", type=Path, default=Path("results/review_regional_table5_v1"))
    parser.add_argument("--out", type=Path, default=Path("results/review_regional_horizons_v1"))
    parser.add_argument("--leads", nargs="+", type=int, default=[1, 3, 6])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    args.omni_dir = args.omni_dir or args.project_root / "画图脚本" / "F10.7 data"
    args.leads = sorted(set(args.leads))
    try:
        if args.leads[0] < 1:
            raise ValueError("Lead times must be positive hours")
        if args.out.resolve() in {path.resolve() for path in (args.parent, args.project_root, args.archive_dir, args.omni_dir)}:
            raise ValueError("Choose a separate output directory for this horizon comparison")
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, EOFError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
