# Short-Term Ionospheric VTEC Forecasting

Python code for training and evaluating VTEC forecasts from CODE global ionospheric maps, with MATLAB scripts for the paper figures. The experiments cover nine locations, five regions, and the area-weighted global mean.

## Setup

Use Python 3.12 and install the dependencies from the repository root. The regional run used NumPy 2.4.3 and PyTorch 2.7.1+cu118 on CUDA. Use the recorded environment when reusing archived outputs.

```sh
python -m pip install -r requirements.txt
```

Use MATLAB R2024b for the figures. Figures 2 and 10 use Mapping Toolbox. Figure 4 uses Signal Processing Toolbox.

## Inputs

| Input | Files and source |
| --- | --- |
| Location VTEC | Place `vtec_9locations_2002_2023.csv`, `year_summary.csv`, and `archive_config.json` in `results/archive_2002_2023_v1/`. To build them from the study's processed `sequence_array_YYYY.npz` files, use the archive command below. The source maps are listed under [CODE products](https://www.aiub.unibe.ch/download) and in [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=e2f2b90f04d74b749d8fd0b0f8e39e6f). |
| Regional and global VTEC | Put `EU_REC_Raw.csv`, `NA_REC_Raw.csv`, `EA_REC_Raw.csv`, `SAM_REC_Raw.csv`, `LL_REC_Raw.csv`, and `GEC_REC_Raw.csv` in `data/regional/`. Use the six cosine-latitude-weighted mean series from the study. Each file supplies a timestamp column and `VTEC_True` in TECU. |
| Hourly Dst | Put the [NASA OMNI2](https://omniweb.gsfc.nasa.gov/html/overview.html) annual ASCII files in `data/omni2/`, retaining the names `omni2_YYYY.dat`. The aligned runs use 2002–2023 with 2020 excluded. |
| SVMD solver | Use the study copy of `PyITD_svmd_zero_energy_v1.py`. Place it in the local repository root or give its location with `--svmd-file`. Its SHA-256 is recorded in both reference configurations under `configs/`. |

The location series use nominal hourly labels from archive row order. The 2020 array has 8,760 rows versus 8,784 calendar hours. Its nominal timestamps are blank. The VTEC-only location runs used 2020 in validation. The Dst-input location runs and the regional runs excluded 2020.

The files in `configs/` record the original settings and input hashes. The regional reference lists each input by filename. Use the original run directories to open saved caches.

## Regional VTEC extraction

The two scripts below start from the processed annual NPZ arrays. The template supplies time labels and auxiliary columns. Results depend on the archive, the template, and the CSV writer version. Compare the exported files with the input hashes in `configs/review_regional_table5_v1.reference.json`.

```sh
python python/extract_rec_fixed.py --npz-dir data/annual_npz --template-csv data/legacy_locations/ADIS_SVMD_Physics_Residual_OMNI_2002_2023.csv --out-dir data/regional
python python/extract_sam_south_america.py --npz-dir data/annual_npz --template-csv data/legacy_locations/ADIS_SVMD_Physics_Residual_OMNI_2002_2023.csv --out-dir data/regional
```

The first script also writes `SA_REC_Raw.csv` for southern Africa. The regional experiments use `SAM_REC_Raw.csv` for South America. The extractors retain the original row-based timestamps and linear interpolation. The regional training scripts read VTEC from these files and obtain Dst from OMNI2.

## Run the experiments

Run the commands from the repository root. Start with a fresh working copy for a new run. To reuse archived outputs, keep the original inputs, code, and run configurations together. The scripts check their file hashes.

Build the nine-location archive if you have the annual NPZ files.

```sh
python python/code_gim_prepare_archive_2002_2023_v1.py --npz-dir data/annual_npz --out results/archive_2002_2023_v1
```

Run the location models and score the storm events.

```sh
python python/code_gim_main_baselines_2002_2023_v1.py --archive-dir results/archive_2002_2023_v1 --out results/main_2002_2023_baselines_v1
python python/code_gim_main_svmd_2002_2023_v1.py --archive-dir results/archive_2002_2023_v1 --baseline-dir results/main_2002_2023_baselines_v1 --svmd-file PyITD_svmd_zero_energy_v1.py --out results/main_2002_2023_svmd_zero_energy_v1
python python/code_gim_main_pmsc_2002_2023_v1.py --omni-dir data/omni2
python python/code_gim_review_fourier_thresholds_v1.py fourier --omni-dir data/omni2
python python/code_gim_review_storm_events_2021_2023_v1.py --omni-dir data/omni2
```

Table 2 uses the location metrics in `results/main_2002_2023_svmd_zero_energy_v1/`. The storm catalog in `results/review_storm_events_2021_2023_v1/` supplies the event counts for Table 3. Figure 8 returns the nine-location RMSE values for Table 4.

Run the regional and global evaluations.

```sh
python python/code_gim_review_regional_table5_v1.py --project-root data/regional --omni-dir data/omni2 --svmd-file PyITD_svmd_zero_energy_v1.py
python python/code_gim_review_regional_horizons_v1.py --project-root data/regional --omni-dir data/omni2 --leads 1 3 6
```

The first script writes `table5_rmse.csv`. The second writes `annual_test_rmse.csv`, which gives every model's annual RMSE for Table 6. These commands start from the six processed regional and global series.

## Make the figures

Run `addpath('matlab')` from the repository root in MATLAB.

| Figure | Command | Input |
| --- | --- | --- |
| 2 | `Fig02_global_station_map` | GNSS station coordinates stored in the script. |
| 3 | `Fig03_long_term_vtec_2002_2023('data/legacy_locations')` | NYAL, POTS, and ADIS `*_SVMD_Physics_Residual_OMNI_2002_2023.csv` files. The figure uses the `VTEC_True` column. |
| 4 | `Fig04_vmd_modes_3d('data/legacy_locations')` | The ADIS file used in Figure 3. This illustration uses MATLAB VMD and the original row-based date labels. |
| 5 | `Fig05_scatter_48point_t6_axes_swapped('data/Results_t6_Final')` | The nine archived `*_Predictions_t6.csv` files. Uses data rows 50–97 to plot eight models. |
| 6 | `Fig06_taylor_diagram_t6('data/Results_t6_Final')` | The same six-hour files. Plots the four direct models and SVMD–Transformer. |
| 7 | `Fig07_storm_bars_1h(pwd)` | Saved location baselines, Dst-input forecasts, and the storm catalog. |
| 8 | `[fig, metrics] = Fig08_intense_storm_1h(pwd, 2)` | Saved one-hour Dst-input forecasts and the storm catalog for the March 2023 example. |
| 9 | `Fig09_ablation_violin_1h(pwd)` | Saved one-hour Dst-input forecasts and the intense events in the storm catalog. |
| 10 | `Fig10_regional_distribution_map` | Region boundaries stored in the script. |

Figure 1 is a workflow schematic. Figures 5 and 6 plot archived six-hour location forecasts. The run that produced those forecasts reads a precomputed baseline from each target row. Verify the inputs used to construct that baseline before interpreting the figures as six-hour-ahead scores. Table 6 uses separately trained, origin-based regional and global forecasts.

## Software availability

The study used a modified Python SVMD implementation. The solver is an external dependency for this package. The recovered copy matches the SHA-256 from both original run configurations. Its header credits Nazari, Sakhaei, and the Python translation. See the [related Python source](https://github.com/falseywinchnet/PyITD/blob/main/svmd.py), its [repository terms](https://github.com/falseywinchnet/PyITD/blob/main/license.txt), and the [original MATLAB SVMD distribution](https://www.mathworks.com/matlabcentral/fileexchange/98649-successive-variational-mode-decomposition-svmd-m).

## Reference

Nazari, M., and Sakhaei, S. M. (2020). Successive variational mode decomposition. *Signal Processing*, 174, 107610. [https://doi.org/10.1016/j.sigpro.2020.107610](https://doi.org/10.1016/j.sigpro.2020.107610)
