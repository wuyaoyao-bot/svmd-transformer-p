"""Run the Fourier comparison and training-period VTEC jump statistics.

Place beside the existing long-archive scripts. The full Fourier comparison
reuses the completed PMSC predictions and trains the two Fourier hybrids.
The reference PMSC threshold is the one used in the historical main run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import code_gim_main_pmsc_2002_2023_v1 as pmsc


pilot, baseline, svmd = pmsc.pilot, pmsc.baseline, pmsc.svmd
PERIODS = (24.0, 12.0)
THRESHOLDS = (4, 8, 12)
FOURIER = "fourier"
HYBRIDS = (("fourier_transformer_dst", None), ("pmsc_fourier_transformer", "adaptive"))
NEW_MODELS = (FOURIER, *(name for name, _ in HYBRIDS))
MODEL_NAMES = ("persistence", *pmsc.MODELS, *NEW_MODELS)
ARCHIVE_FILES = ("archive_config.json", "year_summary.csv", "vtec_9locations_2002_2023.csv")


def harmonic_features(windows):
    """Fit intercept, daily, and semidiurnal terms to each past-only window."""
    hours = np.arange(1 - windows.shape[1], 2, dtype=np.float64)
    columns = [np.ones_like(hours)]
    for period in PERIODS:
        phase = 2 * np.pi * hours / period
        columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack(columns)
    past = design[:-1]
    coefficient = windows @ np.linalg.pinv(past).T
    background = coefficient @ past.T
    pure = coefficient @ design[-1]
    # Use the same origin anchor and three-interval slope as the SVMD branch.
    base = windows[:, -1] + (background[:, -1] - background[:, -4]) / 3
    return pure, windows - background, base


def metric_rows(results, names):
    for site, data in results.items():
        dst = data.get("dst_t0")
        cap = pmsc.dynamic_cap(dst) if dst is not None else None
        for year in ("all", *baseline.TEST_YEARS):
            period = np.ones(len(data["truth"]), dtype=bool) if year == "all" else data["keys"][:, 0] == year
            regimes = [("all", period)]
            if dst is not None:
                regimes.extend((("quiet", period & (dst > -30)), ("storm", period & (dst <= -50))))
            for regime, mask in regimes:
                n = int(mask.sum())
                if n == 0:
                    continue
                for name in names:
                    forecast = data[name][mask]
                    rmse, mae = baseline.error_metrics(forecast - data["truth"][mask])
                    row = dict(site=site, year=year, regime=regime, model=name,
                               n_pairs=n, rmse_tecu=rmse, mae_tecu=mae)
                    if cap is not None:
                        excess = np.maximum(np.abs(forecast - data["origin"][mask]) - cap[mask], 0)
                        row.update(jump_exceed_pct=float(100 * (excess > 0).mean()),
                                   jump_excess_mean_tecu=float(excess.mean()))
                    yield row


def fourier_only(values, out):
    results = {}
    for index, site in enumerate(pilot.SITES):
        samples, keys = baseline.build_samples(values, index)
        windows, truth = samples["test"]
        pure, _, _ = harmonic_features(windows)
        results[site] = dict(keys=keys, truth=truth, persistence=windows[:, -1], fourier=pure)
    metrics = list(metric_rows(results, ("persistence", FOURIER)))
    pilot.write_csv(out / "metrics.csv", list(metrics[0]), metrics)
    for row in metrics:
        if row["year"] == "all" and row["model"] == FOURIER:
            print(f"{row['site']} | Fourier RMSE={row['rmse_tecu']:.4f} | N={row['n_pairs']}")
    print(f"FOURIER BASELINE PASS | {out / 'metrics.csv'}")


def quiet_jump_rows(values, dst_years):
    for index, site in enumerate(pilot.SITES):
        jumps = []
        finite_pairs = 0
        aligned_pairs = 0
        for year, vtec in values.items():
            series = vtec[:, index]
            finite = np.isfinite(series[:-1]) & np.isfinite(series[1:])
            dst_at_origin = dst_years[year][:-1]
            aligned = finite & np.isfinite(dst_at_origin)
            quiet = aligned & (dst_at_origin > -30)
            finite_pairs += int(finite.sum())
            aligned_pairs += int(aligned.sum())
            jumps.append(np.abs(series[1:][quiet] - series[:-1][quiet]))
        changes = np.concatenate(jumps)
        if len(changes) == 0:
            raise ValueError(f"No quiet training pairs for {site}")
        quantiles = np.percentile(changes, [50, 90, 95, 99])
        yield dict(
            site=site, first_year=min(values), last_year=max(values), quiet_origin_dst_gt_nt=-30,
            n_finite_pairs=finite_pairs, n_with_origin_dst=aligned_pairs, n_quiet_pairs=len(changes),
            p50_tecu=float(quantiles[0]), p90_tecu=float(quantiles[1]),
            p95_tecu=float(quantiles[2]), p99_tecu=float(quantiles[3]),
            **{f"pct_gt_{cap}_tecu": float(100 * (changes > cap).mean()) for cap in THRESHOLDS},
        )


def training_thresholds(args, values, out):
    first, last = baseline.SPLIT_YEARS["train"]
    training = {year: values[year] for year in range(first, last + 1)}
    dst, hashes = pmsc.read_dst_years(args.archive_dir, training, args.omni_dir)
    rows = list(quiet_jump_rows(training, dst))
    baseline.establish_config(out, dict(
        train_years=[first, last], time_basis="nominal archive UTC", origin_dst_gt_nt=-30,
        within_year_adjacent_pairs=True, thresholds=list(THRESHOLDS), dst_sha256=hashes,
        archive_sha256={name: pilot.sha256_file(args.archive_dir / name) for name in ARCHIVE_FILES},
        script_sha256=pilot.sha256_file(Path(__file__)),
    ))
    path = out / "training_quiet_jumps.csv"
    pilot.write_csv(path, list(rows[0]), rows)
    for row in rows:
        print(f"{row['site']} | quiet N={row['n_quiet_pairs']} | P95={row['p95_tecu']:.3f} TECU "
              f"| >4={row['pct_gt_4_tecu']:.2f}% | >8={row['pct_gt_8_tecu']:.2f}% "
              f"| >12={row['pct_gt_12_tecu']:.2f}%", flush=True)
    print(f"REVIEW ITEM 2 PASS | {path}", flush=True)


def verify_pmsc(args, base_config, dst_hashes, torch):
    path = args.pmsc_dir / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(
        source_config_sha256=pilot.sha256_file(args.svmd_dir / "run_config.json"),
        script_sha256=pilot.sha256_file(Path(pmsc.__file__)), dst_sha256=dst_hashes,
        nominal_utc=True, excluded_years=[2020], history=baseline.HISTORY,
        arms=[list(arm) for arm in pmsc.ARMS], base_cap=pmsc.BASE_CAP,
        dst_sensitivity=pmsc.DST_SENSITIVITY, fixed_cap=pmsc.FIXED_CAP, phys_weight=pmsc.PHYS_WEIGHT,
        torch_version=str(torch.__version__), numpy_version=np.__version__, device=base_config["device"],
    )
    changed = [key for key, value in expected.items() if config.get(key) != value]
    if changed or not (args.pmsc_dir / "metrics.csv").is_file():
        raise ValueError(f"Need the completed matching PMSC run. Mismatched fields: {changed}")
    return pilot.sha256_file(path)


def aligned_forecasts(path, fields, identity, models):
    with np.load(path, allow_pickle=False) as stream:
        result = {key: stream[key] for key in fields}
    if any(not np.array_equal(result[key], value) for key, value in identity.items()):
        raise ValueError(f"Saved forecasts use different test rows: {path}")
    shape = identity["truth"].shape
    for name in models:
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise ValueError(f"Invalid {name} forecast in {path}")
    return result


def fit_site(torch, device, data, keys, frozen_path, folder):
    test = data["test"]
    identity = dict(keys=keys, truth=test["truth"], origin=test["origin"], dst_t0=test["dst"][:, -1])
    frozen = aligned_forecasts(frozen_path, (*identity, *pmsc.MODELS), identity, pmsc.MODELS)
    if folder.is_dir():
        result = aligned_forecasts(folder / "predictions.npz", (*identity, *NEW_MODELS), identity, NEW_MODELS)
        if any(not (folder / f"{name}.pt").is_file() for name, _ in HYBRIDS):
            raise ValueError(f"Missing fitted Fourier checkpoint: {folder}")
        print(f"{folder.name} | reusing Fourier forecasts", flush=True)
        return dict(frozen, **result, persistence=test["origin"])

    result = dict(identity)
    for split, rows in data.items():
        pure, rows["residual"], rows["base"] = harmonic_features(rows["raw"])
        if split == "test":
            result[FOURIER] = pure
    with tempfile.TemporaryDirectory(prefix="fit_fourier_", dir=folder.parent) as temporary:
        staged = Path(temporary) / "completed"
        staged.mkdir()
        for name, constraint in HYBRIDS:
            forecast, state = pmsc.fit_arm(torch, device, name, data, "svmd", constraint)
            result[name] = forecast
            torch.save(state, staged / f"{name}.pt")
        np.savez_compressed(staged / "predictions.npz", **result)
        staged.rename(folder)
    print(f"{folder.name} | FOURIER SITE PASS", flush=True)
    return dict(frozen, **result, persistence=test["origin"])


def prediction_rows(results):
    for site, data in results.items():
        for i, (year, row) in enumerate(data["keys"]):
            yield dict(
                site=site, year=int(year), target_source_row_0based=int(row),
                dst_t0_nt=float(data["dst_t0"][i]), origin_tecu=float(data["origin"][i]),
                actual_tecu=float(data["truth"][i]), **{name: float(data[name][i]) for name in MODEL_NAMES},
            )


def fourier_comparison(args, values, out):
    import torch

    base_config = svmd.verify_baseline(args.archive_dir, args.baseline_dir)
    svmd_config = pmsc.verify_source_run(args.baseline_dir, args.svmd_dir)
    dst, hashes = pmsc.read_dst_years(args.archive_dir, values, args.omni_dir)
    parent_hash = verify_pmsc(args, base_config, hashes, torch)
    device = torch.device(base_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("The original PMSC run used CUDA, which is currently unavailable")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    frozen = {site: args.pmsc_dir / "sites" / site / "predictions.npz" for site in pilot.SITES}
    baseline.establish_config(out, dict(
        pmsc_run_sha256=parent_hash, script_sha256=pilot.sha256_file(Path(__file__)),
        periods_archive_steps=list(PERIODS), slope_intervals=3, models=list(MODEL_NAMES),
        frozen_forecasts_sha256={site: pilot.sha256_file(path) for site, path in frozen.items()},
    ))
    sites_dir = out / "sites"
    sites_dir.mkdir(exist_ok=True)
    results = {}
    for index, site in enumerate(pilot.SITES):
        data, keys, _ = pmsc.site_data(args, values, dst, svmd_config, index, site)
        results[site] = fit_site(torch, device, data, keys, frozen[site], sites_dir / site)
    rows = list(metric_rows(results, MODEL_NAMES))
    path = out / "metrics.csv"
    pilot.write_csv(path, list(rows[0]), rows)
    fields = ["site", "year", "target_source_row_0based", "dst_t0_nt", "origin_tecu", "actual_tecu", *MODEL_NAMES]
    pilot.write_csv(out / "predictions.csv", fields, prediction_rows(results))
    for site in pilot.SITES:
        for regime in ("all", "storm"):
            scores = [r for r in rows if r["site"] == site and r["year"] == "all" and r["regime"] == regime]
            if scores:
                selected = [r for r in scores if r["model"] in (*NEW_MODELS, "svmd_transformer_dst", "pmsc_svmd_transformer")]
                result = " | ".join(f"{r['model']}={r['rmse_tecu']:.4f}" for r in selected)
                print(f"{site} | {regime} N={scores[0]['n_pairs']} | {result}", flush=True)
    print(f"REVIEW ITEM 1 PASS | {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fourier", "thresholds", "fourier-only"))
    parser.add_argument("--archive-dir", type=Path, default=Path("results/archive_2002_2023_v1"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/main_2002_2023_baselines_v1"))
    parser.add_argument("--svmd-dir", type=Path, default=Path("results/main_2002_2023_svmd_zero_energy_v1"))
    parser.add_argument("--pmsc-dir", type=Path, default=Path("results/main_2002_2023_pmsc_v1"))
    parser.add_argument("--omni-dir", type=Path,
                        default=Path.home() / "Desktop" / "SVMD+lstm" / "画图脚本" / "F10.7 data")
    parser.add_argument("--out", type=Path, default=Path("results/review_fourier_thresholds_v1"))
    parser.set_defaults(check_only=False)
    args = parser.parse_args()
    try:
        out = args.out / args.step
        sources = (args.archive_dir, args.baseline_dir, args.svmd_dir, args.pmsc_dir, args.omni_dir)
        if out.resolve() in {path.resolve() for path in sources}:
            raise ValueError("Choose a separate output directory for the reviewer experiments")
        values = baseline.load_archive(args.archive_dir)
        out.mkdir(parents=True, exist_ok=True)
        with baseline.run_lock(out):
            if args.step == "fourier":
                fourier_comparison(args, values, out)
            elif args.step == "thresholds":
                training_thresholds(args, values, out)
            else:
                fourier_only(values, out)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, EOFError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
