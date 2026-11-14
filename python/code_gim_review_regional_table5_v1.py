"""Rebuild Table 5 from the archived regional and global mean VTEC series."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import code_gim_main_pmsc_2002_2023_v1 as pmsc
import code_gim_review_storm_events_2021_2023_v1 as storms


pilot, baseline, svmd = pmsc.pilot, pmsc.baseline, pmsc.svmd
REGIONS = ("EU", "NA", "EA", "SAM", "LL", "GEC")
OMITTED_YEAR = 2020
BASE_CAP = 4.0
ARMS = (
    ("transformer_dst", "raw", None),
    ("svmd_unconstrained", "svmd", None),
    ("fixed_12", "svmd", "fixed"),
    ("adaptive_4", "svmd", "adaptive"),
)
MODELS = (*baseline.MODELS, *(arm[0] for arm in ARMS))
YEARS = {year: datetime(year, 1, 1, tzinfo=timezone.utc) for year in range(2002, 2025)}


def load_series(root):
    values = {
        year: np.full((int((YEARS[year + 1] - YEARS[year]) / pilot.HOUR), len(REGIONS)), np.nan)
        for year in baseline.YEARS
    }
    sources = {}
    for column, region in enumerate(REGIONS):
        path = root / f"{region}_REC_Raw.csv"
        previous = None
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = [name.strip() for name in next(reader, [])]
            if not header or header[0] not in ("Datetime", "", "Unnamed: 0") or "VTEC_True" not in header:
                raise ValueError(f"Expected the timestamp and VTEC_True columns in {path}")
            value_index = header.index("VTEC_True")
            for row in reader:
                try:
                    epoch = datetime.fromisoformat(row[0].strip())
                    epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch.astimezone(timezone.utc)
                    vtec = float(row[value_index]) if row[value_index].strip() else np.nan
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid regional input at {path}, line {reader.line_num}") from exc
                if (epoch.year not in values or epoch.minute or epoch.second or epoch.microsecond
                        or (previous is not None and epoch <= previous)):
                    raise ValueError(f"Expected increasing hourly archive timestamps at {path}, line {reader.line_num}")
                previous = epoch
                if epoch.year != OMITTED_YEAR:
                    hour = int((epoch - YEARS[epoch.year]) / pilot.HOUR)
                    values[epoch.year][hour, column] = vtec
        coverage = {str(y): int(np.isfinite(a[:, column]).sum()) for y, a in values.items() if y != OMITTED_YEAR}
        sources[region] = dict(path=str(path.resolve()), sha256=pilot.sha256_file(path), finite_hours=coverage)
        print(f"{region} | finite raw hours={sum(coverage.values())} | 2020 omitted", flush=True)
    return values, sources


def samples_for_region(values, dst_years, index):
    samples, keys = baseline.build_samples(values, index)
    data = {}
    for split, (x, truth) in samples.items():
        history = pmsc.aligned_dst(values, index, dst_years, split)
        if history.shape != x.shape:
            raise ValueError(f"VTEC and Dst history lengths disagree for {REGIONS[index]} {split}")
        keep = np.isfinite(history).all(axis=1)
        if not keep.any():
            raise ValueError(f"No common VTEC and Dst windows for {REGIONS[index]} {split}")
        data[split] = dict(raw=x[keep], truth=truth[keep], origin=x[keep, -1], dst=history[keep])
        if split == "test":
            keys = keys[keep]
    counts = " | ".join(f"{s}={len(d['truth'])}" for s, d in data.items())
    print(f"{REGIONS[index]} | {counts}", flush=True)
    return data, keys


def probe(pool, args, values, dst_years):
    tasks, sizes = [], []
    for index, region in enumerate(REGIONS):
        data, _ = samples_for_region(values, dst_years, index)
        for split, rows in data.items():
            x = rows["raw"]
            selected = np.linspace(0, len(x) - 1, min(svmd.PROBE_PER_SPLIT, len(x)), dtype=int)
            tasks.append((len(sizes), x[selected], f"{region} {split}"))
            sizes.append(len(x))
    count, warnings, seconds = 0, 0, 0.0
    started = time.perf_counter()
    for index, residual, _, flags, elapsed in svmd.completed_chunks(pool, tasks, len(tasks)):
        count += len(residual)
        warnings += int(flags.sum())
        seconds += sizes[index] * elapsed / len(residual)
    print(f"REGIONAL TIMING PASS | probe_windows={count} | warnings={warnings} | "
          f"wall_seconds={time.perf_counter() - started:.1f} | planned_windows={sum(sizes)} | "
          f"estimated_decomposition_hours={seconds / args.workers / 3600:.2f}", flush=True)


def fit_region(torch, device, pool, out, region, data, keys):
    test = data["test"]
    identity = dict(keys=keys, truth=test["truth"], origin=test["origin"], dst_t0=test["dst"][:, -1])
    folder = out / "sites" / region
    if folder.is_dir():
        fields = (*identity, *MODELS, "svmd_numeric_warning")
        saved = storms.fourier.aligned_forecasts(folder / "predictions.npz", fields, identity, MODELS)
        flags = saved["svmd_numeric_warning"]
        if flags.shape != test["truth"].shape or not np.isin(flags, (0, 1)).all():
            raise ValueError(f"Invalid SVMD warning flags in {folder}")
        print(f"{region} | reusing completed forecasts", flush=True)
        return saved

    windows = np.concatenate([rows["raw"] for rows in data.values()], axis=0)
    residual, base, flags = svmd.collect_features(pool, windows, region, out / "features" / region)
    start = 0
    for rows in data.values():
        stop = start + len(rows["raw"])
        rows.update(residual=residual[start:stop], base=base[start:stop])
        start = stop
    result = dict(identity, svmd_numeric_warning=flags[-len(test["truth"]):])
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"fit_{region}_", dir=folder.parent) as work:
        staged = Path(work) / "completed"
        staged.mkdir()
        raw_samples = {split: (rows["raw"], rows["truth"]) for split, rows in data.items()}
        references = baseline.fit_site(torch, device, raw_samples, keys, staged / "baselines")
        result.update({name: references[name] for name in baseline.MODELS})
        previous_cap = pmsc.BASE_CAP
        try:
            pmsc.BASE_CAP = BASE_CAP
            for name, branch, constraint in ARMS:
                result[name], state = pmsc.fit_arm(torch, device, name, data, branch, constraint)
                state.update(base_threshold_tecu=BASE_CAP if constraint == "adaptive" else None,
                             fixed_threshold_tecu=pmsc.FIXED_CAP if constraint == "fixed" else None,
                             dst_sensitivity=pmsc.DST_SENSITIVITY if constraint == "adaptive" else 0,
                             phys_weight=pmsc.PHYS_WEIGHT)
                torch.save(state, staged / f"{name}.pt")
        finally:
            pmsc.BASE_CAP = previous_cap
        np.savez_compressed(staged / "predictions.npz", **result)
        staged.rename(folder)
    print(f"{region} | REGIONAL PASS | SVMD warnings={int(flags.sum())}", flush=True)
    return result


def label_forecasts(results, events, membership):
    first_hours = {y: int((YEARS[y] - storms.START) / pilot.HOUR) for y in baseline.TEST_YEARS}
    event_ids = np.asarray([event["event_id"] for event in events])
    levels = np.asarray([event["storm_level"] for event in events])
    for region, data in results.items():
        target_hours = np.array([first_hours[int(y)] + int(hour) for y, hour in data["keys"]])
        indices = membership[target_hours - 1]
        active = indices >= 0
        if not np.array_equal(active, data["dst_t0"] <= storms.DISTURBED_DST):
            raise ValueError(f"Forecasts and storm catalogue disagree for {region}")
        data["target_hours"] = target_hours
        data["event_index"] = indices
        data["event_id"] = np.where(active, event_ids[indices], "")
        data["storm_level"] = np.where(active, levels[indices], "quiet")


def prediction_rows(results):
    for region, data in results.items():
        for i, hour in enumerate(data["target_hours"]):
            target = storms.START + int(hour) * pilot.HOUR
            yield dict(
                region=region, nominal_origin_utc=pilot.stamp(target - pilot.HOUR), nominal_target_utc=pilot.stamp(target),
                event_id=data["event_id"][i], storm_level=data["storm_level"][i],
                dst_t0_nt=data["dst_t0"][i], origin_tecu=data["origin"][i], actual_tecu=data["truth"][i],
                svmd_numeric_warning=int(data["svmd_numeric_warning"][i]),
                **{name: data[name][i] for name in MODELS},
            )


def score(data, keep):
    count = int(keep.sum())
    truth = data["truth"][keep]
    rmse = {f"{name}_rmse_tecu": baseline.error_metrics(data[name][keep] - truth)[0] if count else None
            for name in MODELS}
    reduction = rmse["svmd_unconstrained_rmse_tecu"] - rmse["adaptive_4_rmse_tecu"] if count else None
    return dict(n_pairs=count, **rmse, adaptive_4_rmse_reduction_vs_unconstrained_tecu=reduction)


def write_results(out, results, events, membership):
    label_forecasts(results, events, membership)
    fields = list(next(prediction_rows(results)))
    pilot.write_csv(out / "predictions.csv", fields, prediction_rows(results))
    table, by_event = [], []
    for region, data in results.items():
        for level in ("quiet", *storms.LEVELS):
            keep = data["storm_level"] == level
            n_events = len(np.unique(data["event_index"][keep])) if level != "quiet" else 0
            table.append(dict(region=region, storm_level=level, n_events=n_events, **score(data, keep)))
        for index in np.unique(data["event_index"][data["event_index"] >= 0]):
            event = events[index]
            by_event.append(dict(region=region, **event, **score(data, data["event_index"] == index)))
    pilot.write_csv(out / "table5_rmse.csv", list(table[0]), table)
    if by_event:
        pilot.write_csv(out / "event_rmse.csv", list(by_event[0]), by_event)
    print(f"REGIONAL TABLE 5 PASS | regions={len(results)} | {out / 'table5_rmse.csv'}", flush=True)


def run(args):
    values, sources = load_series(args.project_root)
    dst_years, dst_hashes = pmsc.read_dst_years(args.archive_dir, values, args.omni_dir)
    dst = np.concatenate([dst_years[y] for y in baseline.TEST_YEARS])
    events, membership = storms.catalogue(dst)
    if args.check_only:
        for index in range(len(REGIONS)):
            samples_for_region(values, dst_years, index)
        print(f"REGIONAL INPUT PASS | regions={len(REGIONS)} | Dst events={len(events)}", flush=True)
        return

    solver_path = args.svmd_file.resolve(strict=True)
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    context = mp.get_context("spawn")
    if args.timing_only:
        with context.Pool(args.workers, svmd.initialize_worker, (str(solver_path),)) as pool:
            probe(pool, args, values, dst_years)
        return

    import torch

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(f"REGIONAL RUN | device={device} | workers={args.workers}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with baseline.run_lock(args.out):
        baseline.establish_config(args.out, dict(
            input_series=sources, dst_sha256=dst_hashes, solver_sha256=pilot.sha256_file(solver_path),
            code_sha256={Path(m.__file__).name: pilot.sha256_file(Path(m.__file__))
                         for m in (pilot, baseline, svmd, pmsc, storms, storms.fourier)},
            script_sha256=pilot.sha256_file(Path(__file__)), time_basis="nominal archive CSV timestamps",
            regions=list(REGIONS), excluded_years=[OMITTED_YEAR], split_years=baseline.SPLIT_YEARS,
            history_hours=baseline.HISTORY, lead_hours=1, cross_year_windows=False,
            train_validation_stride=baseline.TRAIN_STRIDE, seed=pilot.SEED,
            models=list(MODELS), arms=[list(arm) for arm in ARMS], base_cap=BASE_CAP, fixed_cap=pmsc.FIXED_CAP,
            dst_sensitivity=pmsc.DST_SENSITIVITY, phys_weight=pmsc.PHYS_WEIGHT, device=str(device),
            torch_version=str(torch.__version__), numpy_version=np.__version__,
            storm_origin_threshold_nt=storms.DISTURBED_DST, storm_merge_hours=storms.MERGE_HOURS,
            storm_moderate_min_dst_nt=storms.MODERATE_DST, storm_intense_min_dst_nt=storms.INTENSE_DST,
        ))
        results = {}
        with context.Pool(args.workers, svmd.initialize_worker, (str(solver_path),)) as pool:
            for index, region in enumerate(REGIONS):
                data, keys = samples_for_region(values, dst_years, index)
                results[region] = fit_region(torch, device, pool, args.out, region, data, keys)
        write_results(args.out, results, events, membership)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home() / "Desktop" / "SVMD+lstm")
    parser.add_argument("--archive-dir", type=Path, default=Path("results/archive_2002_2023_v1"))
    parser.add_argument("--omni-dir", type=Path)
    parser.add_argument("--svmd-file", type=Path, default=Path("PyITD_svmd_zero_energy_v1.py"))
    parser.add_argument("--out", type=Path, default=Path("results/review_regional_table5_v1"))
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    args.omni_dir = args.omni_dir or args.project_root / "画图脚本" / "F10.7 data"
    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        if args.out.resolve() in {p.resolve() for p in (args.project_root, args.archive_dir, args.omni_dir)}:
            raise ValueError("Choose a separate output directory for the regional forecasts")
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, EOFError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
