function [summary, fig] = Fig07_storm_bars_1h(projectRoot)
%FIG07_STORM_BARS_1H Recompute Figure 7 from saved forecasts.
% Place this file in the project root and run Fig07_storm_bars_1h.
% An explicit project root can also be passed as the first argument.
% In the Figure 7 window, select an axes border or the legend and drag it.
% Save the figure manually after adjusting the layout.
% The Figure 7 RMSE and scored hours are written as CSV files.
% The one-step forecasts and event times use nominal archive UTC.
% Score only Dst <= -30 nT origins with forecasts at all nine locations.
% The frozen LSTM and Transformer baselines used 2018-2020 validation.
% The four Dst-input models used 2018-2019 validation.
% The fixed SVMD threshold is 12 TECU. The adaptive threshold is
% 8 + 0.05*abs(Dst at origin) TECU in the saved nine-location run.

    if nargin == 0
        projectRoot = pwd;
    end
    resultsDir = fullfile(projectRoot, 'results');
    baselineFile = fullfile(resultsDir, 'main_2002_2023_baselines_v1', 'predictions.csv');
    forecastFile = fullfile(resultsDir, 'review_fourier_thresholds_v1', 'fourier', 'predictions.csv');
    eventFile = fullfile(resultsDir, 'review_storm_events_2021_2023_v1', 'events.csv');
    outputRoot = fullfile(resultsDir, 'figure07_1h');

    sites = ["ADIS"; "GUAM"; "KOUR"; "WUHN"; "SANT"; "POTS"; "CAS1"; "TRO1"; "NYAL"];
    levels = ["Weak"; "Moderate"; "Intense"];
    modelLabels = {'LSTM', 'Transformer', 'Transformer + Dst', ...
                   'SVMD-Transformer + Dst (no penalty)', 'SVMD-Transformer + Dst (fixed 12)', ...
                   'SVMD-Transformer + Dst (adaptive b = 8)'};
    rmseNames = {'LSTM_TECU', 'Transformer_TECU', 'Transformer_Dst_TECU', ...
                 'SVMD_NoPenalty_TECU', 'SVMD_Fixed12_TECU', 'SVMD_Adaptive8_TECU'};
    transformerColumns = {'frozen_transformer_no_dst', 'transformer_dst', ...
                          'svmd_transformer_dst', 'svmd_transformer_fixed12', ...
                          'pmsc_svmd_transformer'};
    keyColumns = {'year', 'target_source_row_0based'};
    baseline = readRequiredCsv(baselineFile, {'site', 'nominal_target_utc'}, ...
        [keyColumns, {'actual_tecu', 'persistence', 'lstm', 'transformer'}]);
    forecasts = readRequiredCsv(forecastFile, {'site'}, ...
        [keyColumns, {'dst_t0_nt', 'origin_tecu', 'actual_tecu'}, transformerColumns]);
    [baselineKey, baselineTarget] = forecastKeys(baseline, sites, baselineFile);
    [forecastKey, targetUtc] = forecastKeys(forecasts, sites, forecastFile);
    assert(all(parseUtc(baseline.nominal_target_utc) == baselineTarget), ...
        'Baseline UTC timestamps disagree with the year and target row in %s', baselineFile);
    [matched, baselineRow] = ismember(forecastKey, baselineKey, 'rows');
    assert(all(matched), 'Some forecast rows have no matching LSTM baseline in %s', baselineFile);
    referenceFields = [baseline.actual_tecu(baselineRow), baseline.persistence(baselineRow), ...
                       baseline.transformer(baselineRow)];
    forecastFields = [forecasts.actual_tecu, forecasts.origin_tecu, ...
                      forecasts.frozen_transformer_no_dst];
    assert(all(abs(referenceFields - forecastFields) <= 1e-9, 'all'), ...
        'The two forecast files disagree on GIM reference values, origin VTEC or Transformer forecasts.');

    firstUtc = datetime(2021, 1, 1, 'TimeZone', 'UTC');
    lastUtc = datetime(2024, 1, 1, 'TimeZone', 'UTC');
    nHours = hours(lastUtc - firstUtc);
    nSites = numel(sites);
    % If the target is t+1, its zero-based hour is the origin's one-based index.
    originIndex = hours(targetUtc - firstUtc);
    rowIndex = sub2ind([nHours, nSites], originIndex, forecastKey(:, 1));
    dstBySite = nan(nHours, nSites);
    dstBySite(rowIndex) = forecasts.dst_t0_nt;
    dstAtHour = min(dstBySite, [], 2, 'omitnan');
    maxDstAtHour = max(dstBySite, [], 2, 'omitnan');
    hasForecast = isfinite(dstAtHour);
    assert(all(dstAtHour(hasForecast) == maxDstAtHour(hasForecast)), ...
        'Origin Dst differs between locations at the same UTC hour.');

    [events, eventAtHour, levelOfEvent] = loadStormEvents(eventFile, firstUtc, dstAtHour, levels);
    commonHours = find(all(isfinite(dstBySite), 2) & eventAtHour > 0);
    eventRows = eventAtHour(commonHours);
    hourLevels = levelOfEvent(eventRows);

    predictions = [baseline.lstm(baselineRow), forecasts{:, transformerColumns}];
    squaredErrors = (predictions - forecasts.actual_tecu).^2;
    nModels = numel(modelLabels);
    errorGrid = nan(nHours * nSites, nModels);
    errorGrid(rowIndex, :) = squaredErrors;
    errorGrid = reshape(errorGrid, nHours, nSites, nModels);

    nLevels = numel(levels);
    rmse = nan(nLevels, nSites, nModels);
    nPairs = zeros(nLevels, 1);
    nEvents = zeros(nLevels, 1);
    for category = 1:nLevels
        use = hourLevels == category;
        nPairs(category) = nnz(use);
        nEvents(category) = numel(unique(eventRows(use)));
        assert(nPairs(category) > 0, 'No common forecast hours for %s storms.', levels(category));
        rmse(category, :, :) = sqrt(mean(errorGrid(commonHours(use), :, :), 1));
        fprintf('%-8s  %3d catalogue events  %3d scored events  %4d common origin hours\n', ...
            levels(category), nnz(levelOfEvent == category), nEvents(category), nPairs(category));
    end

    summary = table(repelem(sites, nLevels), repmat(levels, nSites, 1), ...
        repmat(nPairs, nSites, 1), repmat(nEvents, nSites, 1), ...
        'VariableNames', {'Location', 'Storm_category', 'N_pairs', 'N_events_scored'});
    summary = [summary, array2table(reshape(rmse, nLevels * nSites, nModels), ...
        'VariableNames', rmseNames)];

    originUtc = firstUtc + hours(commonHours - 1);
    targetUtc = firstUtc + hours(commonHours);
    utcFormat = 'yyyy-MM-dd''T''HH:mm:ss''Z''';
    scoredHours = table(string(originUtc, utcFormat), string(targetUtc, utcFormat), ...
        events.event_id(eventRows), levels(hourLevels), dstAtHour(commonHours), ...
        'VariableNames', {'Origin_utc', 'Target_utc', 'Event_id', 'Storm_category', 'Dst_origin_nT'});

    fig = drawBars(rmse, sites, levels, modelLabels);
    if ~isfolder(outputRoot)
        mkdir(outputRoot);
    end
    stage = tempname(outputRoot);
    mkdir(stage);
    cleanup = onCleanup(@() removeStage(stage)); %#ok<NASGU>
    writetable(summary, fullfile(stage, 'Figure07_storm_rmse_1h.csv'));
    writetable(scoredHours, fullfile(stage, 'Figure07_scored_hours.csv'));
    [~, runId] = fileparts(stage);
    outputDir = fullfile(outputRoot, ['run_', runId]);
    [moved, message] = movefile(stage, outputDir);
    assert(moved, 'Could not save the Figure 7 outputs. %s', message);
    fprintf('FIGURE 7 PASS\n%s\n', outputDir);
end

function removeStage(stage)
    if isfolder(stage)
        rmdir(stage, 's');
    end
end

function T = readRequiredCsv(file, textColumns, numericColumns)
    assert(isfile(file), 'Missing input file %s', file);
    opts = detectImportOptions(file, 'Delimiter', ',', 'Encoding', 'UTF-8', ...
        'VariableNamingRule', 'preserve');
    columns = [textColumns, numericColumns];
    missingColumns = setdiff(columns, opts.VariableNames, 'stable');
    if ~isempty(missingColumns)
        error('Missing columns in %s (%s)', file, strjoin(missingColumns, ', '));
    end
    opts.SelectedVariableNames = columns;
    opts = setvartype(opts, textColumns, 'string');
    opts = setvartype(opts, numericColumns, 'double');
    opts.ImportErrorRule = 'error';
    T = readtable(file, opts);
    assert(height(T) > 0 && all(isfinite(T{:, numericColumns}), 'all') && ...
        ~any(ismissing(T{:, textColumns}), 'all'), 'Empty or invalid required data in %s', file);
end

function [key, targetUtc] = forecastKeys(T, sites, file)
    [knownSite, siteNumber] = ismember(T.site, sites);
    row = T.target_source_row_0based;
    assert(all(knownSite & ismember(T.year, 2021:2023) & row == fix(row) & row >= 72 & row < 8760), ...
        'Expected nine-location test targets for 2021–2023 with zero-based rows 72–8759 in %s', file);
    key = [siteNumber, T.year, row];
    assert(size(unique(key, 'rows'), 1) == height(T), ...
        'Duplicate location, year and target row in %s', file);
    one = ones(size(row));
    targetUtc = datetime(T.year, one, one, 'TimeZone', 'UTC') + hours(row);
end

function utc = parseUtc(values)
    utc = datetime(values, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss''Z''', 'TimeZone', 'UTC');
    assert(~any(isnat(utc)), 'A required UTC timestamp could not be read.');
end

function [events, eventAtHour, levelOfEvent] = loadStormEvents(file, firstUtc, dstAtHour, levels)
    timeColumns = {'first_disturbed_utc', 'peak_dst_utc', 'last_disturbed_utc'};
    events = readRequiredCsv(file, [{'event_id', 'storm_level'}, timeColumns], ...
        {'min_dst_nt', 'disturbed_origin_hours'});
    assert(numel(unique(events.event_id)) == height(events), 'Duplicate event ID in %s', file);
    [knownLevel, levelOfEvent] = ismember(lower(events.storm_level), lower(levels));
    minDst = events.min_dst_nt;
    expectedLevel = 1 + (minDst <= -50) + (minDst <= -100);
    assert(all(knownLevel & minDst <= -30 & levelOfEvent == expectedLevel), ...
        'Storm categories disagree with the catalogue Dst thresholds in %s', file);

    first = hours(parseUtc(events.first_disturbed_utc) - firstUtc) + 1;
    peak = hours(parseUtc(events.peak_dst_utc) - firstUtc) + 1;
    last = hours(parseUtc(events.last_disturbed_utc) - firstUtc) + 1;
    indices = [first, peak, last];
    nHours = numel(dstAtHour);
    nDisturbed = events.disturbed_origin_hours;
    assert(all(indices == fix(indices), 'all') && ...
        all(first >= 1 & first <= peak & peak <= last & last <= nHours & ...
            nDisturbed == fix(nDisturbed) & nDisturbed >= 1 & nDisturbed <= last - first + 1), ...
        'Invalid UTC bounds or disturbed-hour counts in %s', file);

    eventAtHour = zeros(nHours, 1);
    for event = 1:height(events)
        span = first(event):last(event);
        assert(all(eventAtHour(span) == 0), 'Overlapping storm catalogue windows in %s', file);
        eventAtHour(span) = event;
    end
    disturbed = isfinite(dstAtHour) & dstAtHour <= -30;
    assert(any(disturbed) && all(eventAtHour(disturbed) > 0), ...
        'Disturbed forecast origins are missing from the event catalogue %s', file);
    matchedEvents = eventAtHour(disturbed);
    counts = accumarray(matchedEvents, ones(nnz(disturbed), 1), [height(events), 1]);
    peakIsAvailable = isfinite(dstAtHour(peak));
    assert(all(dstAtHour(disturbed) >= minDst(matchedEvents)) && ...
        all(dstAtHour(peak(peakIsAvailable)) == minDst(peakIsAvailable)) && ...
        all(counts <= nDisturbed), 'Forecast Dst values disagree with the event catalogue %s', file);
    eventAtHour(~disturbed) = 0;
end

function fig = drawBars(rmse, sites, levels, modelLabels)
    font = 'Times New Roman';
    colors = [
        128 128 128
        213  94   0
        230 159   0
        178  34  34
          0 114 178
          0 158 115
    ] / 255;
    fig = figure('Name', 'Figure 7', 'NumberTitle', 'off', 'WindowStyle', 'normal', 'Visible', 'on', ...
        'Color', 'w', 'Units', 'centimeters', 'Position', [1, 1, 28, 22]);
    % Each axes has a figure-relative position that can be moved independently.
    columnLeft = [0.100, 0.401, 0.702];
    rowBottom = [0.641, 0.371, 0.101];
    panelSize = [0.267, 0.230];
    for location = 1:numel(sites)
        row = ceil(location / 3);
        column = mod(location - 1, 3) + 1;
        panelPosition = [columnLeft(column), rowBottom(row), panelSize];
        ax = axes('Parent', fig, 'Units', 'normalized', 'Position', panelPosition);
        bars = bar(ax, reshape(rmse(:, location, :), numel(levels), []), ...
            'grouped', 'BarWidth', 0.86, 'LineWidth', 0.45, 'EdgeColor', [0.18 0.18 0.18]);
        for model = 1:numel(bars)
            bars(model).FaceColor = colors(model, :);
        end
        if location == 1
            legendAxes = ax;
            legendBars = bars;
        end
        rowSites = (3 * row - 2):(3 * row);
        yMax = max(0.5, 1.25 * max(rmse(:, rowSites, :), [], 'all'));
        set(ax, 'FontName', font, 'FontSize', 11, 'Box', 'on', 'LineWidth', 0.8, ...
            'PositionConstraint', 'innerposition', 'Position', panelPosition, ...
            'YGrid', 'on', 'GridLineStyle', '--', 'GridAlpha', 0.24, ...
            'XLim', [0.5 3.5], 'YLim', [0 yMax], 'XTick', 1:numel(levels));
        if column ~= 1
            ax.YTickLabel = [];
        end
        if row == 3
            ax.XTickLabel = cellstr(levels);
        else
            ax.XTickLabel = [];
        end
        text(ax, 0.96, 0.94, sprintf('(%c) %s', 'a' + location - 1, sites(location)), ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
            'FontName', font, 'FontSize', 13, 'FontWeight', 'bold');
        if row == 2 && column == 1
            ylabel(ax, 'RMSE (TECU)', 'FontName', font, 'FontSize', 13);
        end
        if row == 3 && column == 2
            xlabel(ax, 'Storm category', 'FontName', font, 'FontSize', 13);
        end
    end
    lgd = legend(legendAxes, legendBars, modelLabels, 'NumColumns', 3, 'Orientation', 'horizontal', ...
        'Box', 'off', 'FontName', font, 'FontSize', 11, 'Units', 'normalized');
    drawnow;
    legendPosition = lgd.Position;
    legendPosition(1) = (1 - legendPosition(3)) / 2;
    legendPosition(2) = 0.975 - legendPosition(4);
    lgd.Position = legendPosition;
    plotedit(fig, 'on');
end
