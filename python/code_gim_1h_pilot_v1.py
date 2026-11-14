"""Windows-friendly one-hour VTEC pilot from hourly CODE final IONEX maps.

Run smoke first. Prepare downloads the Jan–Apr 2023 cohort and compare uses it
for an exploratory archive replay. Only compare needs NumPy and PyTorch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen


SITES = {
    "ADIS": (9.0, 38.8),
    "GUAM": (13.6, 144.9),
    "KOUR": (5.3, -52.8),
    "WUHN": (30.5, 114.4),
    "SANT": (-33.1, -70.7),
    "POTS": (52.4, 13.1),
    "CAS1": (-66.3, 110.5),
    "TRO1": (69.6, 18.9),
    "NYAL": (78.9, 11.9),
}
FIRST_DAY = date(2023, 1, 1)
LAST_DAY = date(2023, 4, 30)
VAL_START = datetime(2023, 3, 1, tzinfo=timezone.utc)
TEST_START = datetime(2023, 4, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
CODE_URL = "https://www.aiub.unibe.ch/download/CODE"
SEED = 42


def filename(day):
    return f"COD0OPSFIN_{day:%Y%j}0000_01D_01H_GIM.INX.gz"


def tag(line):
    return "".join(line[60:].split())


def fixed_reals(line, count):
    return tuple(float(line[2 + 6 * j : 8 + 6 * j]) for j in range(count))


def axis(bounds):
    start, end, step = bounds
    if not step:
        raise ValueError(f"Invalid IONEX grid step {bounds}")
    count = round((end - start) / step) + 1
    if count < 1 or not math.isclose(start + (count - 1) * step, end):
        raise ValueError(f"Invalid IONEX grid {bounds}")
    return tuple(round(start + j * step, 4) for j in range(count))


def utc_epoch(line):
    year, month, day, hour, minute, second = map(int, line[:60].split())
    if hour not in range(25) or minute != 0 or second != 0:
        raise ValueError(f"Expected a whole-hour IONEX epoch: {line[:60].strip()}")
    return datetime(year, month, day, tzinfo=timezone.utc) + hour * HOUR


def map_values(stream, latitudes, lon_bounds, selected, scale):
    epoch = None
    lat_seen = set()
    values = {}
    for line in stream:
        label = tag(line)
        if label == "EPOCHOFCURRENTMAP":
            if epoch is not None:
                raise ValueError("Repeated EPOCH OF CURRENT MAP")
            epoch = utc_epoch(line)
        elif label == "LAT/LON1/LON2/DLON/H":
            lat, lon1, lon2, step, _ = fixed_reals(line, 5)
            if lat not in latitudes or lat in lat_seen or (lon1, lon2, step) != lon_bounds:
                raise ValueError("TEC map row disagrees with the IONEX header")
            lat_seen.add(lat)
            row = []
            while len(row) < 73:
                data_line = next(stream, "").rstrip()
                if not data_line:
                    raise ValueError("Incomplete TEC map row")
                for k in range(0, len(data_line), 5):
                    row.append(int(data_line[k : k + 5]))
            if len(row) != 73:
                raise ValueError("Expected 73 TEC values in each latitude row")
            for site, (grid_lat, lon_index) in selected.items():
                if grid_lat == lat:
                    raw = row[lon_index]
                    values[site] = raw * scale if 0 <= raw < 9999 else None
        elif label == "ENDOFTECMAP":
            if epoch is None or len(lat_seen) != len(latitudes) or len(values) != len(SITES):
                raise ValueError("Incomplete TEC map")
            return epoch, values
    raise ValueError("Missing END OF TEC MAP")


def read_day(path, day):
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as stream:
        first = next(stream, "")
        if tag(first) != "IONEXVERSION/TYPE":
            raise ValueError(f"Expected a gridded IONEX file: {path}")
        header = {}
        for line in stream:
            label = tag(line)
            if label == "ENDOFHEADER":
                break
            header[label] = line
        else:
            raise ValueError(f"Missing END OF HEADER: {path}")

        interval = int(header["INTERVAL"][:6])
        exponent = int(header["EXPONENT"][:6])
        latitudes = axis(fixed_reals(header["LAT1/LAT2/DLAT"], 3))
        lon_bounds = fixed_reals(header["LON1/LON2/DLON"], 3)
        longitudes = axis(lon_bounds)
        if interval != 3600 or (len(latitudes), len(longitudes)) != (71, 73):
            raise ValueError(f"Expected hourly 71×73 CODE maps: {path}")

        selected = {}
        for site, (lat, lon) in SITES.items():
            grid_lat = min(latitudes, key=lambda value: abs(value - lat))
            lon_index = min(range(len(longitudes)), key=lambda j: abs(longitudes[j] - lon))
            selected[site] = (grid_lat, lon_index)

        times = {}
        map_count = 0
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end = start + 24 * HOUR
        for line in stream:
            if tag(line) == "STARTOFTECMAP":
                epoch, values = map_values(stream, latitudes, lon_bounds, selected, 10.0**exponent)
                map_count += 1
                if epoch < start or epoch > end:
                    raise ValueError(f"Map epoch outside the file's UTC day: {epoch}")
                if epoch < end:
                    if epoch in times:
                        raise ValueError(f"Repeated in-day map epoch: {epoch}")
                    times[epoch] = values

    expected = [start + j * HOUR for j in range(24)]
    if sorted(times) != expected:
        raise ValueError(f"{day} has {len(times)}/24 in-day hourly epochs in {path}")
    grid = {site: (lat, longitudes[j]) for site, (lat, j) in selected.items()}
    return [(epoch, times[epoch]) for epoch in expected], grid, map_count


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
    ) as stream:
        temp = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def stamp(epoch):
    return epoch.strftime("%Y-%m-%dT%H:%M:%SZ")


def save_series(path, records):
    rows = (dict(utc=stamp(epoch), **values) for epoch, values in records)
    write_csv(path, ["utc", *SITES], rows)


def load_series(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        records = [
            (
                datetime.fromisoformat(row["utc"]),
                {site: float(row[site]) if row[site] else None for site in SITES},
            )
            for row in rows
        ]
    epochs = [epoch for epoch, _ in records]
    if not epochs or epochs != sorted(set(epochs)):
        raise ValueError(f"Expected a nonempty strictly increasing UTC series: {path}")
    if any(e.utcoffset() != timedelta(0) or e.minute or e.second or e.microsecond for e in epochs):
        raise ValueError(f"Expected whole-hour UTC timestamps: {path}")
    return records


def fetch_day(day, sample_dir, cache):
    name = filename(day)
    url = f"{CODE_URL}/{day.year}/{name}"
    local = sample_dir / name
    cached = cache / name
    if local.is_file():
        return day, local, url
    if cached.is_file():
        return day, cached, url

    request = Request(url, headers={"User-Agent": "CODE-GIM-hourly-pilot/1.0"})
    limit = 5 * 1024**2
    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read(limit + 1)
        if len(payload) > limit:
            raise ValueError("Unexpectedly large download")
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            decoded = stream.read(2 * limit + 1)
        if len(decoded) > 2 * limit:
            raise ValueError("Unexpectedly large uncompressed IONEX")
        first = decoded.partition(b"\n")[0].decode("ascii", errors="replace")
        if tag(first) != "IONEXVERSION/TYPE":
            raise ValueError("Downloaded content does not start with an IONEX header")
    except (OSError, ValueError, EOFError, HTTPException, zlib.error) as exc:
        raise ValueError(f"{day} download failed ({exc}). Source: {url}") from exc

    with tempfile.NamedTemporaryFile("wb", dir=cache, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            os.replace(temp, cached)
        finally:
            temp.unlink(missing_ok=True)
    return day, cached, url


def smoke(source, out):
    records, grid, map_count = read_day(source, FIRST_DAY)
    rows = []
    for (origin, previous), (target, actual) in zip(records, records[1:]):
        for site in SITES:
            if previous[site] is not None and actual[site] is not None:
                rows.append(
                    {
                        "origin_utc": stamp(origin),
                        "target_utc": stamp(target),
                        "site": site,
                        "actual_tecu": actual[site],
                        "persistence_tecu": previous[site],
                    }
                )
    if not rows:
        raise ValueError("No valid one-hour persistence pairs")
    errors = [row["persistence_tecu"] - row["actual_tecu"] for row in rows]
    rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
    save_series(out / "smoke_hourly.csv", records)
    write_csv(out / "smoke_persistence.csv", list(rows[0]), rows)
    print(f"SMOKE PASS | maps_in_file={map_count} | in_day_hours={len(records)}")
    print(f"Sites={len(grid)} | valid_1h_pairs={len(rows)} | persistence_RMSE={rmse:.4f} TECU")
    print(f"UTC {stamp(records[0][0])} to {stamp(records[-1][0])}")
    print(f"Results: {out}")


def prepare(sample_dir, out):
    days = [FIRST_DAY + timedelta(days=j) for j in range((LAST_DAY - FIRST_DAY).days + 1)]
    cache = out / "official_cache"
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = []
    initial_grid = None
    print(f"Preparing {len(days)} CODE days ({FIRST_DAY} through {LAST_DAY})", flush=True)
    workers = 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(days), workers):
            batch = days[start : start + workers]
            downloads = pool.map(lambda day: fetch_day(day, sample_dir, cache), batch)
            for day, path, url in downloads:
                hours, grid, map_count = read_day(path, day)
                if initial_grid is None:
                    initial_grid = grid
                if grid != initial_grid:
                    raise ValueError(f"Nearest grid coordinates changed on {day}")
                records.extend(hours)
                manifest.append(
                    dict(
                        date=day.isoformat(),
                        file=str(path),
                        source_url=url,
                        sha256=sha256_file(path),
                        maps_in_file=map_count,
                        retained_hours=len(hours),
                        missing_site_values=sum(v is None for _, values in hours for v in values.values()),
                    )
                )
            print(f"Prepared {len(manifest)}/{len(days)} days through {batch[-1]}", flush=True)
    write_csv(out / "source_manifest.csv", list(manifest[0]), manifest)
    grid_rows = [
        dict(site=site, latitude=SITES[site][0], longitude=SITES[site][1], grid_lat=lat, grid_lon=lon)
        for site, (lat, lon) in initial_grid.items()
    ]
    write_csv(out / "station_grid.csv", list(grid_rows[0]), grid_rows)
    save_series(out / "hourly_vtec.csv", records)
    print(f"PREPARE PASS | UTC hours={len(records)} | {out / 'hourly_vtec.csv'}")


def read_dst(path):
    dst = {}
    with path.open(encoding="ascii") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) not in (55, 56):
                raise ValueError(f"Expected hourly OMNI2 ASCII at line {number}: {path}")
            year, doy, hour = map(int, fields[:3])
            if not 1 <= doy <= 366 or not 0 <= hour <= 23:
                raise ValueError(f"Invalid OMNI2 time at line {number}: {path}")
            epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hour)
            value = int(fields[40])
            if value != 99999:
                if epoch in dst:
                    raise ValueError(f"Duplicate OMNI2 time {epoch}: {path}")
                dst[epoch] = value
    return dst


def regime(dst):
    if dst is None:
        return "unknown"
    if dst <= -50:
        return "storm"
    return "quiet" if dst > -30 else "intermediate"


def samples_for_site(np, records, site, history, stride):
    times = [epoch for epoch, _ in records]
    series = np.array([values[site] for _, values in records], dtype=np.float64)
    choices = []
    for j in range(history, len(series)):
        if j % stride or times[j] - times[j - history] != history * HOUR:
            continue
        if np.isfinite(series[j - history : j + 1]).all():
            choices.append(j)
    x = np.array([series[j - history : j] for j in choices])
    y = series[choices]
    target_times = [times[j] for j in choices]
    split = np.array(
        ["train" if t < VAL_START else "validation" if t < TEST_START else "test" for t in target_times]
    )
    counts = {key: int(np.sum(split == key)) for key in ("train", "validation", "test")}
    if min(counts.values()) == 0:
        raise ValueError(f"{site} needs valid train, validation and test samples. Counts: {counts}")
    return x, y, target_times, split, counts


def load_svmd(path):
    spec = importlib.util.spec_from_file_location("local_svmd_pilot", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load the SVMD implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.svmd


def svmd_inputs(np, svmd, windows, site):
    residual = np.empty_like(windows)
    bases = np.empty(len(windows), dtype=np.float64)
    for j, window in enumerate(windows):
        if np.ptp(window) == 0:
            background = window.copy()
        else:
            modes, _, centers = svmd(
                window.copy(), maxAlpha=2000, tau=0.0, tol=1e-6, stopc=4, init_omega=0
            )
            centers = np.asarray(centers, dtype=np.float64).reshape(-1)
            modes = np.asarray(modes, dtype=np.float64)
            if modes.shape == (len(window), len(centers)):
                modes = modes.T
            if (
                not 1 <= len(centers) <= 60
                or modes.shape != (len(centers), len(window))
                or not np.isfinite(centers).all()
                or not np.isfinite(modes).all()
            ):
                raise ValueError(f"{site} SVMD returned invalid modes at sample {j}")
            slow = np.abs(centers) <= 1 / 12
            if not slow.any():
                slow[np.argmin(np.abs(centers))] = True
            background = modes[slow].sum(axis=0)
        residual[j] = window - background
        bases[j] = window[-1] + (background[-1] - background[-4]) / 3
        if j == 0 or (j + 1) % 100 == 0 or j + 1 == len(windows):
            print(f"{site} SVMD windows {j + 1}/{len(windows)}", flush=True)
    return residual, bases


def fit_network(torch, np, name, x, target, split, epochs):
    from torch import nn

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.LSTM(1, 24, batch_first=True)
            self.output = nn.Linear(24, 1)

        def forward(self, values):
            sequence, _ = self.recurrent(values)
            return self.output(sequence[:, -1, :]).squeeze(-1)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(1, 24)
            self.position = nn.Parameter(torch.zeros(1, x.shape[1], 24))
            layer = nn.TransformerEncoderLayer(
                d_model=24, nhead=4, dim_feedforward=48, dropout=0.1, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.output = nn.Linear(24, 1)

        def forward(self, values):
            history = self.project(values) + self.position
            return self.output(self.encoder(history)[:, -1, :]).squeeze(-1)

    torch.manual_seed(SEED)
    model = LSTM() if name == "lstm" else Transformer()
    x_tensor = torch.as_tensor(x[:, :, None], dtype=torch.float32)
    y_tensor = torch.as_tensor(target, dtype=torch.float32)
    train_indices = torch.as_tensor(np.flatnonzero(split == "train"))
    val_indices = torch.as_tensor(np.flatnonzero(split == "validation"))
    test_indices = torch.as_tensor(np.flatnonzero(split == "test"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    generator = torch.Generator().manual_seed(SEED)
    best = math.inf
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for batch in order.split(64):
            optimizer.zero_grad()
            error = loss(model(x_tensor[batch]), y_tensor[batch])
            if not torch.isfinite(error):
                raise ValueError(f"{name} produced a non-finite loss")
            error.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(loss(model(x_tensor[val_indices]), y_tensor[val_indices]))
        print(f"  {name} epoch {epoch + 1}/{epochs} | validation MSE={val:.5f}", flush=True)
        if val < best:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale == 3:
                break
    if best_state is None:
        raise ValueError(f"{name} produced no finite validation score")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor[test_indices]).numpy().astype(np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError(f"{name} produced non-finite test predictions")
    return predictions


def metric_rows(rows, sites, models):
    for site in ["ALL", *sites]:
        for subset in ("all", "quiet", "storm"):
            selected = [
                row for row in rows
                if (site == "ALL" or row["site"] == site)
                and (subset == "all" or row["regime"] == subset)
            ]
            for model in models:
                errors = [row[model] - row["actual_tecu"] for row in selected]
                count = len(errors)
                yield dict(
                    site=site,
                    subset=subset,
                    model=model,
                    n=count,
                    rmse_tecu=math.sqrt(math.fsum(e * e for e in errors) / count) if count else "",
                    mae_tecu=math.fsum(abs(e) for e in errors) / count if count else "",
                )


def compare(out, args, root):
    import numpy as np
    import torch

    if args.history < 24 or args.history % 2 or args.stride < 1 or args.epochs < 1:
        raise ValueError("Use an even history >= 24 and positive stride and epochs")
    sites = list(dict.fromkeys(args.sites))
    series_path = out / "hourly_vtec.csv"
    records = load_series(series_path)
    first = datetime.combine(FIRST_DAY, time.min, tzinfo=timezone.utc)
    count = ((LAST_DAY - FIRST_DAY).days + 1) * 24
    if [epoch for epoch, _ in records] != [first + j * HOUR for j in range(count)]:
        raise ValueError("Expected the complete Jan–Apr 2023 series. Run prepare first.")
    omni = args.omni or root / "画图脚本" / "F10.7 data" / "omni2_2023.dat"
    dst = read_dst(omni)
    test_dst = [dst[epoch] for epoch, _ in records if epoch >= TEST_START and epoch in dst]
    if not test_dst:
        raise ValueError(f"No Dst records for April 2023: {omni}")
    print(f"April Dst coverage={len(test_dst)}/720 hours | Dst<=-50 nT hours={sum(v <= -50 for v in test_dst)}")
    svmd_path = args.svmd_file or root / "_third_party_svmd" / "PyITD_svmd_patched_axisfix_powerfix.py"
    svmd = load_svmd(svmd_path) if args.with_svmd else None
    models = ["persistence", "lstm", "transformer"]
    if svmd is not None:
        models.append("svmd_transformer")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    run = out / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True)
    config = dict(
        data_sha256=sha256_file(series_path),
        omni_sha256=sha256_file(omni),
        svmd_sha256=sha256_file(svmd_path) if svmd is not None else None,
        sites=sites,
        history_hours=args.history,
        origin_stride_hours=args.stride,
        lead_hours=1,
        quiet_dst_gt_nt=-30,
        storm_dst_le_nt=-50,
        max_epochs=args.epochs,
        seed=SEED,
        torch_version=torch.__version__,
        train_start_utc=stamp(first),
        validation_start_utc=stamp(VAL_START),
        test_start_utc=stamp(TEST_START),
        test_end_utc=stamp(records[-1][0]),
        model_names=models,
        samples={},
    )
    predictions = []
    for site in sites:
        x, y, targets, split, counts = samples_for_site(np, records, site, args.history, args.stride)
        config["samples"][site] = counts
        print(f"{site} | train={counts['train']} validation={counts['validation']} test={counts['test']}")
        train = split == "train"
        test = split == "test"
        mean = float(x[train].mean())
        scale = max(float(x[train].std()), 0.1)
        base = x[:, -1]
        estimates = {"persistence": base[test]}
        raw = (x - mean) / scale
        for name in ("lstm", "transformer"):
            change = fit_network(torch, np, name, raw, (y - base) / scale, split, args.epochs)
            estimates[name] = base[test] + change * scale
        if svmd is not None:
            residual, trend_base = svmd_inputs(np, svmd, x, site)
            change = fit_network(
                torch, np, "svmd_transformer", residual / scale,
                (y - trend_base) / scale, split, args.epochs,
            )
            estimates["svmd_transformer"] = trend_base[test] + change * scale
        for k, j in enumerate(np.flatnonzero(test)):
            target = targets[j]
            index = dst.get(target)
            row = dict(
                origin_utc=stamp(target - HOUR),
                target_utc=stamp(target),
                site=site,
                dst_nt=index,
                regime=regime(index),
                actual_tecu=y[j],
            )
            row.update({name: estimates[name][k] for name in models})
            predictions.append(row)

    metrics = list(metric_rows(predictions, sites, models))
    write_csv(run / "predictions.csv", list(predictions[0]), predictions)
    write_csv(run / "metrics.csv", list(metrics[0]), metrics)
    (run / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for row in metrics:
        if row["site"] == "ALL":
            value = f"{row['rmse_tecu']:.4f}" if row["n"] else "NA"
            print(f"{row['subset']:5} {row['model']:17} N={row['n']:5} RMSE={value} TECU")
    print(f"COMPARE PASS | {run / 'metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "prepare", "compare"))
    parser.add_argument("--project-root", type=Path, default=Path.home() / "Desktop" / "SVMD+lstm")
    parser.add_argument("--ionex-dir", type=Path, help="Directory with existing official INX.gz files")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--omni", type=Path, help="Hourly OMNI2 ASCII used for evaluation groups")
    parser.add_argument("--sites", nargs="+", choices=list(SITES), default=["ADIS", "POTS", "NYAL"])
    parser.add_argument("--history", type=int, default=72)
    parser.add_argument("--stride", type=int, default=3, help="Stride between origins, not the lead")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--with-svmd", action="store_true")
    parser.add_argument("--svmd-file", type=Path)
    args = parser.parse_args()
    root = args.project_root
    sample_dir = args.ionex_dir or (
        root / "audit_evidence_svmd_v1" / "code_gim_trace_20261109_v1" / "official_samples"
    )
    out = args.out or root / "VTEC_1H_PILOT_V1"
    try:
        if args.stage == "smoke":
            smoke(sample_dir / filename(FIRST_DAY), out)
        elif args.stage == "prepare":
            prepare(sample_dir, out)
        else:
            compare(out, args, root)
    except (OSError, ValueError, ImportError, KeyError, RuntimeError, EOFError, zlib.error) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
