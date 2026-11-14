"""Compare causal SVMD–Transformer on the frozen 2002–2023 baseline samples.

Keep the pilot and baseline scripts beside this file. Use --timing-only for
a brief decomposition benchmark before running the complete experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import code_gim_1h_pilot_v1 as pilot
import code_gim_main_baselines_2002_2023_v1 as baseline


SPLITS = ("train", "validation", "test")
HYBRID = "svmd_transformer"
MODELS = (*baseline.MODELS, HYBRID)
CHUNK_SIZE = 256
PROBE_PER_SPLIT = 8
WAIT_SECONDS = 600
_worker_svmd = None


def verify_baseline(archive_dir, baseline_dir):
    config = json.loads((baseline_dir / "run_config.json").read_text(encoding="utf-8"))
    expected = dict(
        split_years=baseline.SPLIT_YEARS, sites=list(pilot.SITES), models=list(baseline.MODELS),
        history_rows=baseline.HISTORY, lead_rows=1, train_validation_stride=baseline.TRAIN_STRIDE,
        test_stride=1, cross_year_windows=False, seed=pilot.SEED, max_epochs=baseline.MAX_EPOCHS,
        patience=baseline.PATIENCE, batch_size=baseline.TRAIN_BATCH, learning_rate=baseline.LEARNING_RATE,
    )
    changed = [key for key, value in expected.items() if config.get(key) != value]
    if changed:
        raise ValueError(f"Baseline configuration differs for {', '.join(changed)}")
    if pilot.sha256_file(Path(baseline.__file__)) != config["script_sha256"]:
        raise ValueError("The baseline script differs from the version used to produce the saved forecasts")
    if set(config["inputs"]) != {"archive_config.json", "year_summary.csv", "vtec_9locations_2002_2023.csv"}:
        raise ValueError("Expected the three archive files recorded by the baseline run")
    for name, digest in config["inputs"].items():
        if pilot.sha256_file(archive_dir / name) != digest:
            raise ValueError(f"Archive input differs from the saved baseline: {name}")
    if not (baseline_dir / "metrics.csv").is_file():
        raise ValueError(f"Complete the baseline run first: {baseline_dir}")
    return config


def initialize_worker(path):
    global _worker_svmd
    _worker_svmd = pilot.load_svmd(Path(path))


def decompose_chunk(task):
    start, windows, site = task
    flagged = np.zeros(len(windows), dtype=np.uint8)
    variable = iter(np.flatnonzero(np.ptp(windows, axis=1) != 0))

    def tracked(window, **kwargs):
        index = next(variable)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            result = _worker_svmd(window, **kwargs)
        flagged[index] = any(issubclass(item.category, RuntimeWarning) for item in captured)
        return result

    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        residual, extrapolated = pilot.svmd_inputs(np, tracked, windows, f"{site} chunk {start}")
    if not np.isfinite(residual).all() or not np.isfinite(extrapolated).all():
        raise ValueError(f"Non-finite SVMD features for {site} chunk {start}")
    return start, residual, extrapolated, flagged, time.perf_counter() - started


def completed_chunks(pool, tasks, total):
    pending = pool.imap_unordered(decompose_chunk, tasks, chunksize=1)
    for _ in range(total):
        try:
            yield pending.next(timeout=WAIT_SECONDS)
        except mp.TimeoutError as exc:
            raise RuntimeError(f"No SVMD block finished in {WAIT_SECONDS} seconds") from exc


def timing_probe(pool, workers, values, baseline_dir):
    tasks = []
    sizes = []
    for index, site in enumerate(pilot.SITES):
        samples, keys = baseline.build_samples(values, index)
        test_x, test_y = samples["test"]
        baseline.load_site_cache(baseline_dir / "sites" / site, keys, test_y, test_x[:, -1])
        for split in SPLITS:
            x = samples[split][0]
            selected = np.linspace(0, len(x) - 1, min(PROBE_PER_SPLIT, len(x)), dtype=int)
            tasks.append((len(sizes), x[selected], f"{site} {split}"))
            sizes.append(len(x))
    count, flagged, worker_seconds = 0, 0, 0.0
    started = time.perf_counter()
    for index, residual, _, flags, seconds in completed_chunks(pool, tasks, len(tasks)):
        count += len(residual)
        flagged += int(flags.sum())
        worker_seconds += sizes[index] * seconds / len(residual)
    estimate_hours = worker_seconds / workers / 3600
    print(f"Timing probe | windows={count} | workers={workers} | wall_seconds={time.perf_counter() - started:.1f}")
    print(f"SVMD warnings | probe_windows={flagged}/{count}")
    print(f"TIMING PASS | planned_windows={sum(sizes)} | estimated_decomposition_hours={estimate_hours:.2f}")


def save_chunk(path, residual, extrapolated, flags):
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            np.savez_compressed(stream, residual=residual, extrapolated=extrapolated, flags=flags)
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def collect_features(pool, windows, site, folder):
    folder.mkdir(parents=True, exist_ok=True)
    size = len(windows)
    residual = np.empty_like(windows)
    extrapolated = np.empty(size, dtype=np.float64)
    flagged = np.empty(size, dtype=np.uint8)

    def put(start, r, e, f):
        stop = min(size, start + CHUNK_SIZE)
        if (
            r.shape != (stop - start, baseline.HISTORY) or e.shape != (stop - start,)
            or f.shape != (stop - start,) or not np.isfinite(r).all()
            or not np.isfinite(e).all() or not np.isin(f, (0, 1)).all()
        ):
            raise ValueError(f"Inconsistent SVMD block for {site} at row {start}")
        residual[start:stop], extrapolated[start:stop], flagged[start:stop] = r, e, f

    missing = []
    cached = 0
    for start in range(0, size, CHUNK_SIZE):
        path = folder / f"{start:06d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as feature:
                put(start, feature["residual"], feature["extrapolated"], feature["flags"])
            cached += min(CHUNK_SIZE, size - start)
        else:
            missing.append(start)
    print(f"{site} | SVMD windows={size} | resumed={cached}", flush=True)
    tasks = ((start, windows[start:start + CHUNK_SIZE], site) for start in missing)
    done, computed = cached, 0
    started = time.perf_counter()
    for start, r, e, f, _ in completed_chunks(pool, tasks, len(missing)):
        put(start, r, e, f)
        save_chunk(folder / f"{start:06d}.npz", r, e, f)
        done += len(r)
        computed += len(r)
        if computed == len(r) or computed % (8 * CHUNK_SIZE) < len(r) or done == size:
            elapsed = time.perf_counter() - started
            remaining = (size - done) * elapsed / computed / 60
            print(f"{site} | SVMD {done}/{size} | est. {remaining:.1f} min left at this location", flush=True)
    return residual, extrapolated, flagged


def fit_site(torch, device, pool, samples, saved, site, out):
    target_path = out / "sites" / site
    truth = saved["actual"]
    if target_path.exists():
        with np.load(target_path / "predictions.npz", allow_pickle=False) as cached:
            keys, actual, hybrid, flagged = (cached[key] for key in ("keys", "actual", HYBRID, "flags"))
        if (
            not np.array_equal(keys, saved["keys"]) or not np.array_equal(actual, truth)
            or hybrid.shape != truth.shape or flagged.shape != truth.shape
            or not np.isfinite(hybrid).all() or not np.isin(flagged, (0, 1)).all()
            or not (target_path / f"{HYBRID}.pt").is_file()
        ):
            raise ValueError(f"Saved SVMD predictions do not match frozen test samples: {target_path}")
        print(f"{site} | reusing completed SVMD forecast", flush=True)
        return dict(saved, svmd_transformer=hybrid, flags=flagged)

    x = np.concatenate([samples[split][0] for split in SPLITS])
    residual, extrapolated, flagged = collect_features(pool, x, site, out / "features" / site)
    scale = max(float(samples["train"][0].std()), 0.1)
    tensors, warning_counts = {}, {}
    start = 0
    for split in SPLITS:
        y = samples[split][1]
        stop = start + len(y)
        tensors[split] = (
            torch.as_tensor((residual[start:stop] / scale)[:, :, None], dtype=torch.float32),
            torch.as_tensor((y - extrapolated[start:stop]) / scale, dtype=torch.float32),
        )
        warning_counts[split] = int(flagged[start:stop].sum())
        start = stop
    print(f"{site} | SVMD numerical-warning windows {warning_counts}", flush=True)
    change, state, epoch, mse = baseline.fit_model(torch, HYBRID, tensors, device)
    hybrid = extrapolated[-len(truth):] + change.astype(np.float64) * scale
    test_flags = flagged[-len(truth):]
    with tempfile.TemporaryDirectory(prefix="svmd_fit_", dir=target_path.parent) as temporary:
        staged = Path(temporary) / "completed"
        staged.mkdir()
        torch.save(
            dict(state_dict=state, train_scale=scale, best_epoch=epoch, validation_mse=mse,
                 history=baseline.HISTORY, warning_counts=warning_counts),
            staged / f"{HYBRID}.pt",
        )
        np.savez_compressed(staged / "predictions.npz", keys=saved["keys"], actual=truth,
                            svmd_transformer=hybrid, flags=test_flags)
        staged.rename(target_path)
    print(f"{site} | SVMD SITE PASS", flush=True)
    return dict(saved, svmd_transformer=hybrid, flags=test_flags)


def scores(results):
    rows = baseline.build_metrics(results)
    for period in ("all", *baseline.TEST_YEARS):
        parts, location_scores = [], []
        for site, data in results.items():
            errors = data[HYBRID] - data["actual"]
            if period != "all":
                errors = errors[data["keys"][:, 0] == period]
            if not len(errors):
                continue
            rmse, mae = baseline.error_metrics(errors)
            parts.append(errors)
            location_scores.append((rmse, mae))
            rows.append(dict(period=period, location=site, model=HYBRID, n_pairs=len(errors),
                             rmse_tecu=rmse, mae_tecu=mae))
        if parts:
            errors = np.concatenate(parts)
            pooled = baseline.error_metrics(errors)
            macro = np.mean(location_scores, axis=0)
            for label, (rmse, mae) in (("pooled", pooled), ("location_macro", macro)):
                rows.append(dict(period=period, location=label, model=HYBRID, n_pairs=len(errors),
                                 rmse_tecu=float(rmse), mae_tecu=float(mae)))
    return rows


def prediction_rows(results):
    by_site = {
        site: iter(zip(data[HYBRID], data["flags"], strict=True)) for site, data in results.items()
    }
    for row in baseline.prediction_rows(results):
        hybrid, flagged = next(by_site[row["site"]])
        yield dict(row, svmd_transformer=float(hybrid), svmd_numeric_warning=int(flagged))


def run_full(args, pool, values, config, svmd_path):
    import torch

    if str(torch.__version__) != config["torch_version"] or np.__version__ != config["numpy_version"]:
        raise ValueError("Use the same PyTorch and NumPy environment as the saved baselines")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("The saved baseline used CUDA, which is currently unavailable")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    args.out.mkdir(parents=True, exist_ok=True)
    with baseline.run_lock(args.out):
        lock_config = dict(
            baseline_config_sha256=pilot.sha256_file(args.baseline_dir / "run_config.json"),
            baseline_prediction_sha256={
                site: pilot.sha256_file(args.baseline_dir / "sites" / site / "predictions.npz")
                for site in pilot.SITES
            },
            script_sha256=pilot.sha256_file(Path(__file__)),
            pilot_script_sha256=pilot.sha256_file(Path(pilot.__file__)),
            svmd_file_sha256=pilot.sha256_file(svmd_path),
            chunk_size=CHUNK_SIZE, test_stride=1, history_rows=baseline.HISTORY, lead_rows=1,
        )
        baseline.establish_config(args.out, lock_config)
        (args.out / "sites").mkdir(exist_ok=True)
        results = {}
        for index, site in enumerate(pilot.SITES):
            samples, keys = baseline.build_samples(values, index)
            test_x, test_y = samples["test"]
            saved = baseline.load_site_cache(
                args.baseline_dir / "sites" / site, keys, test_y, test_x[:, -1]
            )
            results[site] = fit_site(torch, device, pool, samples, saved, site, args.out)

        metrics = scores(results)
        pilot.write_csv(args.out / "metrics.csv", list(metrics[0]), metrics)
        pilot.write_csv(
            args.out / "predictions.csv",
            ["site", "year", "target_source_row_0based", "nominal_target_utc", "actual_tecu",
             *MODELS, "svmd_numeric_warning"],
            prediction_rows(results),
        )
        for row in metrics:
            if row["period"] == "all" and row["location"] in ("pooled", "location_macro"):
                print(f"{row['location']:14} {row['model']:17} N={row['n_pairs']:7} "
                      f"RMSE={row['rmse_tecu']:.4f} TECU")
        flagged = sum(int(data["flags"].sum()) for data in results.values())
        total = sum(len(data["actual"]) for data in results.values())
        print(f"SVMD numerical-warning test windows | {flagged}/{total}")
        print(f"MAIN SVMD PASS | {args.out / 'metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.home() / "Desktop" / "SVMD+lstm")
    parser.add_argument("--svmd-file", type=Path)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        if args.out.resolve() in (args.archive_dir.resolve(), args.baseline_dir.resolve()):
            raise ValueError("Choose a separate --out directory for the SVMD comparison")
        config = verify_baseline(args.archive_dir, args.baseline_dir)
        values = baseline.load_archive(args.archive_dir)
        svmd_path = args.svmd_file or args.project_root / "_third_party_svmd" / "PyITD_svmd_patched_axisfix_powerfix.py"
        pilot.load_svmd(svmd_path)
        for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[variable] = "1"
        with mp.get_context("spawn").Pool(
            args.workers, initializer=initialize_worker, initargs=(str(svmd_path.resolve()),)
        ) as pool:
            if args.timing_only:
                timing_probe(pool, args.workers, values, args.baseline_dir)
            else:
                run_full(args, pool, values, config, svmd_path)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, EOFError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
