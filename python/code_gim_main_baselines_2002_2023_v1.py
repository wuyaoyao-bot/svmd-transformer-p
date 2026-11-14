"""Fit the long-archive persistence, LSTM and Transformer baselines.

Place this beside code_gim_1h_pilot_v1.py. The target is the next stored
VTEC row within a year. The 2020 validation rows use sequence order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from code_gim_1h_pilot_v1 import SEED, SITES, sha256_file, stamp, write_csv


SPLIT_YEARS = dict(train=[2002, 2017], validation=[2018, 2020], test=[2021, 2023])
YEARS = range(2002, 2024)
TEST_YEARS = range(2021, 2024)
HISTORY = 72
TRAIN_STRIDE = 3
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.001
TRAIN_BATCH = 64
EVAL_BATCH = 256
MODELS = ("persistence", "lstm", "transformer")


def load_archive(root):
    config = json.loads((root / "archive_config.json").read_text(encoding="utf-8"))
    if config["years"] != [2002, 2023] or config["split_years"] != SPLIT_YEARS:
        raise ValueError("Archive configuration does not match the 2002–2023 split")
    if set(config["grid_mapping"]) != set(SITES):
        raise ValueError("Archive locations do not match the pilot locations")
    with (root / "year_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    summary = {int(row["year"]): row for row in summaries}
    if set(summary) != set(YEARS) or len(summaries) != len(YEARS):
        raise ValueError("Expected one summary record for every year from 2002 to 2023")
    for year in TEST_YEARS:
        item = summary[year]
        if item["time_basis"] != "nominal_row_time" or int(item["rows"]) != 8760:
            raise ValueError(f"Expected the original nominal hourly test rows for {year}")
    values = {year: np.empty((int(summary[year]["rows"]), len(SITES))) for year in YEARS}
    next_row = dict.fromkeys(YEARS, 0)
    series_path = root / "vtec_9locations_2002_2023.csv"
    with series_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            year, source_row = int(row["year"]), int(row["source_row_0based"])
            if source_row != next_row[year] or source_row >= len(values[year]):
                raise ValueError(f"Unexpected source row {source_row} in year {year}")
            if row["split"] != summary[year]["split"] or row["time_basis"] != summary[year]["time_basis"]:
                raise ValueError(f"Archive metadata disagree for year {year}")
            values[year][source_row] = [float(row[site]) if row[site] else np.nan for site in SITES]
            next_row[year] += 1
    if any(len(values[year]) != next_row[year] for year in YEARS) or sum(next_row.values()) != config["total_rows"]:
        raise ValueError("Archive row counts disagree with its summary")
    return values


def build_samples(values, site_index):
    samples = {}
    test_keys = []
    for split, (first, last) in SPLIT_YEARS.items():
        blocks = []
        stride = 1 if split == "test" else TRAIN_STRIDE
        for year in range(first, last + 1):
            sequence = values[year][:, site_index]
            targets = np.arange(HISTORY, len(sequence), stride)
            if not len(targets):
                continue
            windows = np.lib.stride_tricks.sliding_window_view(sequence, HISTORY + 1)[::stride]
            keep = np.isfinite(windows).all(axis=1)
            blocks.append(windows[keep])
            if split == "test":
                test_keys.extend((year, int(target)) for target in targets[keep])
        if not blocks or not any(len(block) for block in blocks):
            raise ValueError(f"No valid {split} windows")
        packed = np.concatenate(blocks, axis=0)
        samples[split] = (packed[:, :-1], packed[:, -1])
    return samples, np.asarray(test_keys, dtype=np.int32)


def make_model(torch, name):
    nn = torch.nn

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.LSTM(1, 24, batch_first=True)
            self.output = nn.Linear(24, 1)

        def forward(self, values):
            states, _ = self.recurrent(values)
            return self.output(states[:, -1, :]).squeeze(-1)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(1, 24)
            self.position = nn.Parameter(torch.zeros(1, HISTORY, 24))
            layer = nn.TransformerEncoderLayer(
                d_model=24, nhead=4, dim_feedforward=48, dropout=0.1, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.output = nn.Linear(24, 1)

        def forward(self, values):
            states = self.encoder(self.project(values) + self.position)
            return self.output(states[:, -1, :]).squeeze(-1)

    return LSTM() if name == "lstm" else Transformer()


def predict_batches(torch, model, features, device):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch.to(device)).cpu() for batch in features.split(EVAL_BATCH)])


def fit_model(torch, name, tensors, device):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = make_model(torch, name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_x, train_y = tensors["train"]
    val_x, val_y = tensors["validation"]
    best_mse, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for indices in order.split(TRAIN_BATCH):
            optimizer.zero_grad()
            forecast = model(train_x[indices].to(device))
            loss = torch.nn.functional.mse_loss(forecast, train_y[indices].to(device))
            if not torch.isfinite(loss):
                raise ValueError(f"{name} produced a non-finite training loss")
            loss.backward()
            optimizer.step()
        val_forecast = predict_batches(torch, model, val_x, device)
        mse = float(torch.nn.functional.mse_loss(val_forecast, val_y))
        print(f"  {name} epoch={epoch}/{MAX_EPOCHS} | validation MSE={mse:.6f}", flush=True)
        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is None:
        raise ValueError(f"{name} produced no finite validation score")
    model.load_state_dict(best_state)
    change = predict_batches(torch, model, tensors["test"][0], device).numpy()
    if not np.isfinite(change).all():
        raise ValueError(f"{name} produced non-finite test predictions")
    return change, best_state, best_epoch, best_mse


def load_site_cache(path, keys, truth, persistence):
    with np.load(path / "predictions.npz", allow_pickle=False) as source:
        cached = {key: source[key] for key in ("keys", "actual", *MODELS)}
    if (
        not np.array_equal(cached["keys"], keys)
        or not np.array_equal(cached["actual"], truth)
        or not np.array_equal(cached["persistence"], persistence)
        or any(cached[name].shape != truth.shape or not np.isfinite(cached[name]).all() for name in MODELS)
        or not all((path / f"{name}.pt").is_file() for name in MODELS[1:])
    ):
        raise ValueError(f"Completed site cache is inconsistent with the experiment: {path}")
    return cached


def fit_site(torch, device, samples, keys, path):
    test_x, truth = samples["test"]
    persistence = test_x[:, -1]
    if path.exists():
        cached = load_site_cache(path, keys, truth, persistence)
        print(f"{path.name} | reusing completed predictions", flush=True)
        return cached
    mean = float(samples["train"][0].mean())
    scale = max(float(samples["train"][0].std()), 0.1)
    tensors = {
        split: (
            torch.as_tensor(((x - mean) / scale)[:, :, None], dtype=torch.float32),
            torch.as_tensor((y - x[:, -1]) / scale, dtype=torch.float32),
        )
        for split, (x, y) in samples.items()
    }
    counts = {split: len(y) for split, (_, y) in samples.items()}
    print(f"{path.name} | train={counts['train']} validation={counts['validation']} test={counts['test']}", flush=True)
    result = dict(keys=keys, actual=truth, persistence=persistence)
    with tempfile.TemporaryDirectory(prefix="fit_site_", dir=path.parent) as temporary:
        staged = Path(temporary) / "completed"
        staged.mkdir()
        for name in MODELS[1:]:
            change, state, epoch, loss = fit_model(torch, name, tensors, device)
            result[name] = persistence + change.astype(np.float64) * scale
            torch.save(
                dict(state_dict=state, history=HISTORY, train_mean=mean, train_scale=scale,
                     best_epoch=epoch, validation_mse=loss, sample_counts=counts),
                staged / f"{name}.pt",
            )
        np.savez_compressed(staged / "predictions.npz", **result)
        staged.rename(path)
    print(f"{path.name} | SITE PASS", flush=True)
    return result


def error_metrics(errors):
    return float(np.sqrt(np.mean(errors**2))), float(np.mean(np.abs(errors)))


def build_metrics(results):
    rows = []
    for period in ("all", *TEST_YEARS):
        for model in MODELS:
            parts = []
            location_metrics = []
            for site, data in results.items():
                errors = data[model] - data["actual"]
                if period != "all":
                    errors = errors[data["keys"][:, 0] == period]
                if not len(errors):
                    continue
                parts.append(errors)
                rmse, mae = error_metrics(errors)
                location_metrics.append((rmse, mae))
                rows.append(dict(period=period, location=site, model=model, n_pairs=len(errors),
                                 rmse_tecu=rmse, mae_tecu=mae))
            if parts:
                errors = np.concatenate(parts)
                pooled = error_metrics(errors)
                macro = np.mean(location_metrics, axis=0)
                for label, (rmse, mae) in (("pooled", pooled), ("location_macro", macro)):
                    rows.append(dict(period=period, location=label, model=model, n_pairs=len(errors),
                                     rmse_tecu=float(rmse), mae_tecu=float(mae)))
    return rows


def prediction_rows(results):
    start = {year: datetime(year, 1, 1, tzinfo=timezone.utc) for year in TEST_YEARS}
    for site, data in results.items():
        for index, (year, target_row) in enumerate(data["keys"]):
            nominal = start[int(year)] + timedelta(hours=int(target_row))
            yield dict(
                site=site, year=int(year), target_source_row_0based=int(target_row),
                nominal_target_utc=stamp(nominal), actual_tecu=float(data["actual"][index]),
                **{name: float(data[name][index]) for name in MODELS},
            )


def establish_config(out, config):
    path = out / "run_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Existing run has different inputs or settings. Choose another --out directory: {out}")
        return
    if (out / "sites").exists():
        raise ValueError("Completed site files exist without a run configuration")
    with tempfile.NamedTemporaryFile("w", dir=out, encoding="utf-8", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(config, stream, indent=2)
            stream.write("\n")
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


@contextmanager
def run_lock(out):
    path = out / "running.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute(args):
    import torch

    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    device = torch.device(device_name)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    args.out.mkdir(parents=True, exist_ok=True)
    with run_lock(args.out):
        values = load_archive(args.archive_dir)
        input_names = ["archive_config.json", "year_summary.csv", "vtec_9locations_2002_2023.csv"]
        config = dict(
            inputs={name: sha256_file(args.archive_dir / name) for name in input_names},
            script_sha256=sha256_file(Path(__file__)),
            split_years=SPLIT_YEARS, sites=list(SITES), models=list(MODELS),
            history_rows=HISTORY, lead_rows=1, train_validation_stride=TRAIN_STRIDE, test_stride=1,
            cross_year_windows=False, time_basis="nominal stored row order",
            seed=SEED, max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=TRAIN_BATCH,
            learning_rate=LEARNING_RATE, torch_version=str(torch.__version__),
            numpy_version=np.__version__, device=device_name,
            accelerator=torch.cuda.get_device_name(device) if device_name == "cuda" else None,
        )
        establish_config(args.out, config)
        by_site = args.out / "sites"
        by_site.mkdir(exist_ok=True)
        print(f"BASELINES | device={device} | years=22 | locations={len(SITES)} | lead=1 row", flush=True)
        results = {}
        for index, site in enumerate(SITES):
            samples, keys = build_samples(values, index)
            results[site] = fit_site(torch, device, samples, keys, by_site / site)
        metrics = build_metrics(results)
        write_csv(args.out / "metrics.csv", list(metrics[0]), metrics)
        write_csv(args.out / "predictions.csv",
                  ["site", "year", "target_source_row_0based", "nominal_target_utc", "actual_tecu", *MODELS],
                  prediction_rows(results))
        for row in metrics:
            if row["period"] == "all" and row["location"] in ("pooled", "location_macro"):
                print(f"{row['location']:14} {row['model']:12} N={row['n_pairs']:7} "
                      f"RMSE={row['rmse_tecu']:.4f} TECU")
        print(f"BASELINES PASS | {args.out / 'metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        execute(args)
    except (OSError, ImportError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
