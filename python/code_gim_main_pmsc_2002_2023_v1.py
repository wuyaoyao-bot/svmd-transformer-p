"""Compare the archived PMSC loss on matched long-archive samples.

Uses the existing long-archive scripts and completed SVMD feature blocks.
The time basis is nominal archive row time. The unresolved 2020 year is
excluded from every new validation arm. No SVMD decomposition is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import code_gim_1h_pilot_v1 as pilot
import code_gim_main_baselines_2002_2023_v1 as baseline
import code_gim_main_svmd_2002_2023_v1 as svmd


ARMS = (
    ("transformer_dst", "raw", None),
    ("svmd_transformer_dst", "svmd", None),
    ("svmd_transformer_fixed12", "svmd", "fixed"),
    ("pmsc_svmd_transformer", "svmd", "adaptive"),
)
REFERENCE = "frozen_transformer_no_dst"
MODELS = (REFERENCE, *(name for name, _, _ in ARMS))
BASE_CAP, DST_SENSITIVITY, FIXED_CAP, PHYS_WEIGHT = 8.0, 0.05, 12.0, 0.05


def dynamic_cap(dst_t0):
    return BASE_CAP + DST_SENSITIVITY * np.abs(dst_t0)


def read_dst_years(archive, values, omni_dir):
    archive_config = json.loads((archive / "archive_config.json").read_text(encoding="utf-8"))
    excluded = set(archive_config["row_count_discrepancy_years"])
    if excluded != {2020}:
        raise ValueError(f"Unexpected unresolved archive years: {sorted(excluded)}")
    dst_years, hashes = {}, {}
    for year, vtec in values.items():
        if year in excluded:
            continue
        path = omni_dir / f"omni2_{year}.dat"
        first = datetime(year, 1, 1, tzinfo=timezone.utc)
        last = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        size = int((last - first).total_seconds() / 3600)
        if len(vtec) != size:
            raise ValueError(f"Nominal UTC row count disagrees for {year}")
        dst = np.full(size, np.nan)
        for epoch, value in pilot.read_dst(path).items():
            if epoch.year != year:
                raise ValueError(f"Dst record outside {year}: {epoch} in {path}")
            dst[int((epoch - first).total_seconds() / 3600)] = value
        count = int(np.isfinite(dst).sum())
        if not count:
            raise ValueError(f"No usable Dst records in {path}")
        if count < size:
            print(f"OMNI Dst {year} | present={count}/{size}", flush=True)
        dst_years[year] = dst
        hashes[path.name] = pilot.sha256_file(path)
    print("Dst input | nominal archive UTC | omitted year=2020", flush=True)
    return dst_years, hashes


def verify_source_run(baseline_dir, svmd_dir):
    config = json.loads((svmd_dir / "run_config.json").read_text(encoding="utf-8"))
    expected = dict(
        baseline_config_sha256=pilot.sha256_file(baseline_dir / "run_config.json"),
        script_sha256=pilot.sha256_file(Path(svmd.__file__)),
        pilot_script_sha256=pilot.sha256_file(Path(pilot.__file__)),
        chunk_size=svmd.CHUNK_SIZE, history_rows=baseline.HISTORY, lead_rows=1, test_stride=1,
    )
    changed = [key for key, value in expected.items() if config.get(key) != value]
    if changed or not (svmd_dir / "metrics.csv").is_file():
        raise ValueError(f"Need the completed, matching SVMD run. Mismatched fields: {changed}")
    return config


def aligned_dst(values, site_index, dst_years, split):
    first, last = baseline.SPLIT_YEARS[split]
    stride = 1 if split == "test" else baseline.TRAIN_STRIDE
    blocks = []
    for year in range(first, last + 1):
        vtec = values[year][:, site_index]
        if len(vtec) <= baseline.HISTORY:
            continue
        vtec_windows = np.lib.stride_tricks.sliding_window_view(vtec, baseline.HISTORY + 1)[::stride]
        valid_vtec = np.isfinite(vtec_windows).all(axis=1)
        if year in dst_years:
            # Exclude the target row. The last input value is Dst at t0.
            dst_windows = np.lib.stride_tricks.sliding_window_view(dst_years[year], baseline.HISTORY + 1)
            blocks.append(dst_windows[::stride, :-1][valid_vtec])
        else:
            blocks.append(np.full((int(valid_vtec.sum()), baseline.HISTORY), np.nan))
    return np.concatenate(blocks, axis=0)


def feature_paths(svmd_dir, site, size):
    for start in range(0, size, svmd.CHUNK_SIZE):
        path = svmd_dir / "features" / site / f"{start:06d}.npz"
        if not path.is_file():
            raise ValueError(f"Missing completed SVMD feature block: {path}")
        yield start, min(start + svmd.CHUNK_SIZE, size), path


def read_features(blocks, size):
    residual = np.empty((size, baseline.HISTORY), dtype=np.float64)
    base = np.empty(size, dtype=np.float64)
    for start, stop, path in blocks:
        with np.load(path, allow_pickle=False) as contents:
            history, extrapolated = contents["residual"], contents["extrapolated"]
        if (
            history.shape != (stop - start, baseline.HISTORY)
            or extrapolated.shape != (stop - start,)
            or not np.isfinite(history).all()
            or not np.isfinite(extrapolated).all()
        ):
            raise ValueError(f"Invalid SVMD feature block: {path}")
        residual[start:stop], base[start:stop] = history, extrapolated
    return residual, base


def site_data(args, values, dst_years, config, site_index, site):
    samples, keys = baseline.build_samples(values, site_index)
    test_x, truth = samples["test"]
    source = args.baseline_dir / "sites" / site
    digest = pilot.sha256_file(source / "predictions.npz")
    if digest != config["baseline_prediction_sha256"][site]:
        raise ValueError(f"Baseline forecasts changed after SVMD feature creation: {site}")
    saved = baseline.load_site_cache(source, keys, truth, test_x[:, -1])
    size = sum(len(y) for _, y in samples.values())
    blocks = list(feature_paths(args.svmd_dir, site, size))
    histories, masks = {}, {}
    for split in svmd.SPLITS:
        dst = aligned_dst(values, site_index, dst_years, split)
        if len(dst) != len(samples[split][1]):
            raise ValueError(f"Dst and archive sample order disagree for {site} {split}")
        keep = np.isfinite(dst).all(axis=1)
        if not keep.any():
            raise ValueError(f"No complete Dst history in {site} {split}")
        histories[split], masks[split] = dst[keep], keep
    coverage = " | ".join(f"{s}={int(k.sum())}/{len(k)}" for s, k in masks.items())
    print(f"{site} | {coverage} | cached SVMD blocks={len(blocks)}", flush=True)
    if args.check_only:
        return None

    residual, base = read_features(blocks, size)
    result, start = {}, 0
    for split in svmd.SPLITS:
        x, y = samples[split]
        stop = start + len(y)
        keep = masks[split]
        result[split] = dict(
            raw=x[keep], residual=residual[start:stop][keep], base=base[start:stop][keep],
            origin=x[keep, -1], truth=y[keep], dst=histories[split],
        )
        start = stop
    mask = masks["test"]
    return result, keys[mask], saved["transformer"][mask]


def prepare_tensors(torch, data, branch, constraint):
    feature_name, base_name = ("raw", "origin") if branch == "raw" else ("residual", "base")
    train = data["train"]
    residual = train["truth"] - train[base_name]
    scale = dict(
        feature_mean=float(train[feature_name].mean()),
        feature_std=max(float(train[feature_name].std()), 0.1),
        dst_mean=float(train["dst"].mean()), dst_std=max(float(train["dst"].std()), 0.1),
        residual_mean=float(residual.mean()), residual_std=max(float(residual.std()), 0.1),
    )
    tensors = {}
    for split, rows in data.items():
        features = np.stack((
            (rows[feature_name] - scale["feature_mean"]) / scale["feature_std"],
            (rows["dst"] - scale["dst_mean"]) / scale["dst_std"],
        ), axis=-1)
        mean, std = scale["residual_mean"], scale["residual_std"]
        labels = (rows["truth"] - rows[base_name] - mean) / std
        anchor = (rows[base_name] + mean - rows["origin"]) / std
        cap = None
        if constraint == "fixed":
            cap = np.full_like(anchor, FIXED_CAP / std)
        elif constraint == "adaptive":
            cap = dynamic_cap(rows["dst"][:, -1]) / std
        tensors[split] = tuple(
            None if array is None else torch.as_tensor(array, dtype=torch.float32)
            for array in (features, labels, anchor, cap)
        )
    return tensors, scale, base_name


def training_loss(torch, estimate, labels, anchor, cap):
    data_loss = (estimate - labels).square().mean()
    if cap is None:
        return data_loss
    # estimate + anchor is the reconstructed one-step VTEC jump in scaled units.
    return data_loss + PHYS_WEIGHT * torch.relu((estimate + anchor).abs() - cap).mean()


def fit_arm(torch, device, name, data, branch, constraint):
    tensors, scale, base_name = prepare_tensors(torch, data, branch, constraint)
    torch.manual_seed(pilot.SEED)
    generator = torch.Generator().manual_seed(pilot.SEED)
    model = baseline.make_model(torch, "transformer")
    model.project = torch.nn.Linear(2, model.project.out_features)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=baseline.LEARNING_RATE)
    x, labels, anchor, cap = tensors["train"]
    val_x, val_labels, _, _ = tensors["validation"]
    best_mse, best_epoch, state = math.inf, 0, None
    for epoch in range(1, baseline.MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for index in order.split(baseline.TRAIN_BATCH):
            optimizer.zero_grad()
            estimate = model(x[index].to(device))
            loss = training_loss(torch, estimate, labels[index].to(device), anchor[index].to(device),
                                 None if cap is None else cap[index].to(device))
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss for {name}")
            loss.backward()
            optimizer.step()
        estimate = baseline.predict_batches(torch, model, val_x, device)
        mse = float((estimate - val_labels).square().mean())
        print(f"  {name} epoch={epoch} | validation RMSE={math.sqrt(mse)*scale['residual_std']:.5f} TECU", flush=True)
        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= baseline.PATIENCE:
            break
    if state is None:
        raise ValueError(f"No finite validation score for {name}")
    model.load_state_dict(state)
    output = baseline.predict_batches(torch, model, tensors["test"][0], device).numpy().astype(np.float64)
    prediction = data["test"][base_name] + scale["residual_mean"] + scale["residual_std"] * output
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite test forecast for {name}")
    return prediction, dict(state_dict=state, scale=scale, branch=branch, constraint=constraint,
                            best_epoch=best_epoch, validation_mse=best_mse)


def fit_site(torch, device, data, keys, reference, folder):
    test = data["test"]
    expected = dict(keys=keys, truth=test["truth"], origin=test["origin"], dst_t0=test["dst"][:, -1],
                    **{REFERENCE: reference})
    if folder.is_dir():
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            cached = {key: saved[key] for key in (*expected, *(a[0] for a in ARMS))}
        if any(not np.array_equal(cached[key], value) for key, value in expected.items()):
            raise ValueError(f"Stored PMSC forecasts disagree with current inputs: {folder}")
        for name, _, _ in ARMS:
            if (
                cached[name].shape != test["truth"].shape or not np.isfinite(cached[name]).all()
                or not (folder / f"{name}.pt").is_file()
            ):
                raise ValueError(f"Invalid stored PMSC model output for {folder.name} {name}")
        print(f"{folder.name} | reusing completed PMSC forecasts", flush=True)
        return cached
    with tempfile.TemporaryDirectory(prefix="fit_pmsc_", dir=folder.parent) as temporary:
        staged = Path(temporary) / "completed"
        staged.mkdir()
        for name, branch, constraint in ARMS:
            forecast, state = fit_arm(torch, device, name, data, branch, constraint)
            expected[name] = forecast
            torch.save(state, staged / f"{name}.pt")
        np.savez_compressed(staged / "predictions.npz", **expected)
        staged.rename(folder)
    print(f"{folder.name} | PMSC SITE PASS", flush=True)
    return expected


def metrics_rows(results):
    for site, data in results.items():
        dst = data["dst_t0"]
        cap = dynamic_cap(dst)
        excess_truth = np.abs(data["truth"] - data["origin"]) > cap
        for year in ("all", *baseline.TEST_YEARS):
            period = np.ones(len(dst), dtype=bool) if year == "all" else data["keys"][:, 0] == year
            for regime, mask in (("all", period), ("quiet", period & (dst > -30)),
                                 ("storm", period & (dst <= -50))):
                n = int(mask.sum())
                if n == 0:
                    continue
                for name in MODELS:
                    rmse, mae = baseline.error_metrics((data[name] - data["truth"])[mask])
                    excess = np.maximum(np.abs(data[name][mask] - data["origin"][mask]) - cap[mask], 0)
                    yield dict(
                        site=site, year=year, regime=regime, model=name, n_pairs=n,
                        rmse_tecu=rmse, mae_tecu=mae,
                        jump_exceed_pct=float(100 * (excess > 0).mean()),
                        jump_excess_mean_tecu=float(excess.mean()),
                        truth_jump_exceed_pct=float(100 * excess_truth[mask].mean()),
                    )


def prediction_rows(results):
    for site, data in results.items():
        for index, (year, target_row) in enumerate(data["keys"]):
            dst = float(data["dst_t0"][index])
            yield dict(
                site=site, year=int(year), target_source_row_0based=int(target_row),
                dst_t0_nt=dst, reference_cap_tecu=float(dynamic_cap(dst)),
                origin_tecu=float(data["origin"][index]), actual_tecu=float(data["truth"][index]),
                **{name: float(data[name][index]) for name in MODELS},
            )


def run(args):
    base_config = svmd.verify_baseline(args.archive_dir, args.baseline_dir)
    source = verify_source_run(args.baseline_dir, args.svmd_dir)
    values = baseline.load_archive(args.archive_dir)
    dst, dst_hashes = read_dst_years(args.archive_dir, values, args.omni_dir)
    if args.check_only:
        for index, site in enumerate(pilot.SITES):
            site_data(args, values, dst, source, index, site)
        print("PMSC INPUT PASS | all sites | cached features and Dst coverage checked")
        return

    import torch

    device = torch.device(base_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("The baseline used CUDA, which is currently unavailable")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    args.out.mkdir(parents=True, exist_ok=True)
    with baseline.run_lock(args.out):
        baseline.establish_config(args.out, dict(
            source_config_sha256=pilot.sha256_file(args.svmd_dir / "run_config.json"),
            script_sha256=pilot.sha256_file(Path(__file__)), dst_sha256=dst_hashes,
            nominal_utc=True, excluded_years=[2020], history=baseline.HISTORY,
            arms=[list(arm) for arm in ARMS], base_cap=BASE_CAP, dst_sensitivity=DST_SENSITIVITY,
            fixed_cap=FIXED_CAP, phys_weight=PHYS_WEIGHT, torch_version=str(torch.__version__),
            numpy_version=np.__version__, device=base_config["device"],
        ))
        by_site = args.out / "sites"
        by_site.mkdir(exist_ok=True)
        results = {}
        for index, site in enumerate(pilot.SITES):
            data, keys, reference = site_data(args, values, dst, source, index, site)
            results[site] = fit_site(torch, device, data, keys, reference, by_site / site)
        metrics = list(metrics_rows(results))
        pilot.write_csv(args.out / "metrics.csv", list(metrics[0]), metrics)
        fields = ["site", "year", "target_source_row_0based", "dst_t0_nt", "reference_cap_tecu",
                  "origin_tecu", "actual_tecu", *MODELS]
        pilot.write_csv(args.out / "predictions.csv", fields, prediction_rows(results))
        for site in pilot.SITES:
            rows = [r for r in metrics if r["site"] == site and r["year"] == "all" and r["regime"] == "all"]
            print(f"{site} | " + " | ".join(f"{r['model']}={r['rmse_tecu']:.4f}" for r in rows))
        print(f"PMSC MAIN PASS | {args.out / 'metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("results/archive_2002_2023_v1"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/main_2002_2023_baselines_v1"))
    parser.add_argument("--svmd-dir", type=Path, default=Path("results/main_2002_2023_svmd_zero_energy_v1"))
    parser.add_argument("--omni-dir", type=Path,
                        default=Path.home() / "Desktop" / "SVMD+lstm" / "画图脚本" / "F10.7 data")
    parser.add_argument("--out", type=Path, default=Path("results/main_2002_2023_pmsc_v1"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.out.resolve() in {p.resolve() for p in (args.archive_dir, args.baseline_dir, args.svmd_dir)}:
            raise ValueError("Choose a separate output directory for PMSC")
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, EOFError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
