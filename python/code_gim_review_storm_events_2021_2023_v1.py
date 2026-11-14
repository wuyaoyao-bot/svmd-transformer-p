"""Build the historical Dst event catalogue and score paired forecasts.

Use the completed historical Fourier comparison. Dst is aligned to the
nominal archive origin hour. Forecasts are scored at the disturbed origins
belonging to each event.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import code_gim_review_fourier_thresholds_v1 as fourier


pilot, baseline, pmsc = fourier.pilot, fourier.baseline, fourier.pmsc
MODELS = fourier.MODEL_NAMES
YEARS = tuple(baseline.TEST_YEARS)
START = datetime(YEARS[0], 1, 1, tzinfo=timezone.utc)
END = datetime(YEARS[-1] + 1, 1, 1, tzinfo=timezone.utc)
MERGE_HOURS = 48
DISTURBED_DST = -30
MODERATE_DST = -50
INTENSE_DST = -100
LEVELS = ("weak", "moderate", "intense")


def load_dst(omni_dir):
    dst = np.full(int((END - START) / pilot.HOUR), np.nan)
    bounds, hashes = {}, {}
    for year in YEARS:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        stop = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        bounds[year] = (int((start - START) / pilot.HOUR), int((stop - start) / pilot.HOUR))
        path = omni_dir / f"omni2_{year}.dat"
        records = pilot.read_dst(path)
        if not records:
            raise ValueError(f"No usable Dst data for {year}: {path}")
        for epoch, value in records.items():
            if not start <= epoch < stop:
                raise ValueError(f"Dst record outside {year}: {epoch}")
            dst[int((epoch - START) / pilot.HOUR)] = value
        hashes[path.name] = pilot.sha256_file(path)
    return dst, bounds, hashes


def catalogue(dst):
    missing = ~np.isfinite(dst)
    active = np.flatnonzero(~missing & (dst <= DISTURBED_DST))
    if len(active) == 0:
        raise ValueError("No Dst events found in the historical test period")
    missing_count = np.r_[0, np.cumsum(missing)]
    gap = np.diff(active) > MERGE_HOURS
    unknown = missing_count[active[1:]] > missing_count[active[:-1] + 1]
    episodes = np.split(active, np.flatnonzero(gap | unknown) + 1)
    membership = np.full(len(dst), -1, dtype=np.int32)
    events = []
    for index, hours in enumerate(episodes):
        first, last = int(hours[0]), int(hours[-1])
        peak = int(hours[np.argmin(dst[hours])])
        start, minimum, end = (START + j * pilot.HOUR for j in (first, peak, last))
        value = float(dst[peak])
        severity = "intense" if value <= INTENSE_DST else "moderate" if value <= MODERATE_DST else "weak"
        before, after = first - MERGE_HOURS, last + MERGE_HOURS + 1
        complete = before >= 0 and after <= len(dst) and missing_count[before] == missing_count[after]
        membership[hours] = index
        events.append(dict(
            event_id=f"E{start:%Y%m%dT%H}", peak_year=minimum.year, storm_level=severity,
            first_disturbed_utc=pilot.stamp(start), peak_dst_utc=pilot.stamp(minimum),
            last_disturbed_utc=pilot.stamp(end), min_dst_nt=value,
            disturbed_origin_hours=len(hours), continuous_span_hours=last - first + 1,
            dst_context_complete=int(complete),
        ))
    return events, membership


def read_predictions(path, dst, bounds, membership, event_count):
    groups = [{site: [] for site in pilot.SITES} for _ in range(event_count)]
    seen = {site: set() for site in pilot.SITES}
    fields = ("site", "year", "target_source_row_0based", "dst_t0_nt", "origin_tecu", "actual_tecu", *MODELS)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        absent = set(fields) - set(reader.fieldnames or ())
        if absent:
            raise ValueError(f"Missing prediction columns: {sorted(absent)}")
        for row in reader:
            site, year = row["site"], int(row["year"])
            target = int(row["target_source_row_0based"])
            start, size = bounds[year]
            if site not in seen or not baseline.HISTORY <= target < size:
                raise ValueError(f"Invalid prediction row at CSV line {reader.line_num}")
            origin_hour = start + target - 1
            if origin_hour in seen[site]:
                raise ValueError(f"Duplicate prediction for {site} at CSV line {reader.line_num}")
            seen[site].add(origin_hour)
            if float(row["dst_t0_nt"]) != dst[origin_hour]:
                raise ValueError(f"Origin Dst mismatch for {site} at CSV line {reader.line_num}")
            numbers = np.asarray([float(row[key]) for key in ("origin_tecu", "actual_tecu", *MODELS)])
            if not np.isfinite(numbers).all():
                raise ValueError(f"Non-finite forecasts for {site} at CSV line {reader.line_num}")
            event = membership[origin_hour]
            if event >= 0:
                groups[event][site].append((origin_hour, numbers))
    if any(not hours for hours in seen.values()):
        raise ValueError("The prediction file must contain all nine historical sites")
    return groups


def score(events, groups, dst):
    rows = []
    for event, by_site in zip(events, groups, strict=True):
        total = event["disturbed_origin_hours"]
        sizes = [len(records) for records in by_site.values()]
        event.update(
            sites_with_forecasts=sum(n > 0 for n in sizes),
            sites_with_all_disturbed_hours=sum(n == total for n in sizes),
            min_site_pairs=min(sizes), max_site_pairs=max(sizes),
        )
        for site, records in by_site.items():
            if not records:
                continue
            ordered = sorted(records, key=lambda record: record[0])
            hours, samples = zip(*ordered, strict=True)
            data = np.stack(samples)
            origin, truth = data[:, 0], data[:, 1]
            observed_peak = int(np.argmax(truth))
            cap = pmsc.dynamic_cap(dst[list(hours)])
            for column, name in enumerate(MODELS, start=2):
                forecast = data[:, column]
                error = forecast - truth
                rmse, mae = baseline.error_metrics(error)
                excess = np.maximum(np.abs(forecast - origin) - cap, 0)
                rows.append(dict(
                    event_id=event["event_id"], peak_year=event["peak_year"], storm_level=event["storm_level"],
                    site=site, model=name, n_pairs=len(records), disturbed_origin_hours=total,
                    rmse_tecu=rmse, mae_tecu=mae, max_abs_error_tecu=float(np.abs(error).max()),
                    error_at_observed_peak_tecu=float(abs(error[observed_peak])),
                    jump_exceed_pct=float(100 * (excess > 0).mean()),
                    jump_excess_mean_tecu=float(excess.mean()),
                ))
    if not rows:
        raise ValueError("No historical event has matched forecasts")
    return rows


def event_counts(events):
    for year in ("all", *YEARS):
        for level in LEVELS:
            matches = [e for e in events if e["storm_level"] == level and (year == "all" or e["peak_year"] == year)]
            yield dict(
                peak_year=year, storm_level=level, dst_events=len(matches),
                events_scored_at_all_sites=sum(e["sites_with_forecasts"] == len(pilot.SITES) for e in matches),
                events_with_complete_coverage=sum(
                    e["dst_context_complete"] == 1 and e["sites_with_all_disturbed_hours"] == len(pilot.SITES)
                    for e in matches
                ),
                disturbed_origin_hours=sum(e["disturbed_origin_hours"] for e in matches),
            )


def run(args):
    dst, bounds, hashes = load_dst(args.omni_dir)
    events, membership = catalogue(dst)
    groups = read_predictions(args.predictions, dst, bounds, membership, len(events))
    metrics = score(events, groups, dst)
    counts = list(event_counts(events))
    baseline.establish_config(args.out, dict(
        years=list(YEARS), time_basis="nominal archive UTC", storm_assignment="forecast origin",
        disturbed_dst_at_most_nt=DISTURBED_DST, moderate_dst_at_most_nt=MODERATE_DST,
        intense_dst_at_most_nt=INTENSE_DST, max_merge_interval_hours=MERGE_HOURS,
        score_hours="disturbed origins", missing_dst_hours=int((~np.isfinite(dst)).sum()),
        dst_sha256=hashes, predictions_sha256=pilot.sha256_file(args.predictions),
        script_sha256=pilot.sha256_file(Path(__file__)),
    ))
    for name, data in (("events.csv", events), ("event_metrics.csv", metrics), ("event_counts.csv", counts)):
        pilot.write_csv(args.out / name, list(data[0]), data)
    for row in counts:
        if row["peak_year"] == "all":
            print(f"{row['storm_level']} | Dst events={row['dst_events']} | "
                  f"scored at all sites={row['events_scored_at_all_sites']} | "
                  f"complete coverage={row['events_with_complete_coverage']}")
    print(f"REVIEW ITEM 3 PASS | {args.out / 'event_counts.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/review_fourier_thresholds_v1/fourier/predictions.csv"))
    parser.add_argument("--omni-dir", type=Path,
                        default=Path.home() / "Desktop" / "SVMD+lstm" / "画图脚本" / "F10.7 data")
    parser.add_argument("--out", type=Path, default=Path("results/review_storm_events_2021_2023_v1"))
    args = parser.parse_args()
    try:
        if args.out.resolve() in (args.predictions.parent.resolve(), args.omni_dir.resolve()):
            raise ValueError("Choose a separate output directory for the event analysis")
        args.out.mkdir(parents=True, exist_ok=True)
        with baseline.run_lock(args.out):
            run(args)
    except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
