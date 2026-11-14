function [fig, metrics, details] = Fig09_ablation_violin_1h(projectRoot)
%FIG09_ABLATION_VIOLIN_1H Compare absolute errors from saved 1-hour forecasts.
%   Run from the project root or pass the project folder as projectRoot.
%   The samples are matched across all nine locations and four model arms.
%   Include only origins with Dst <= -30 nT in catalogue events whose
%   minimum Dst is <= -100 nT. Each matched hour contributes one error
%   per model at each of the nine plotted locations.
%   Colored numbers above the violins give the RMSE of each model.
%   [fig, metrics, details] also returns the statistics, events and hours.
%   Adjust the editable figure window before saving manually.

    if nargin == 0
        projectRoot = pwd;
    end
    resultsDir = fullfile(projectRoot, 'results');
    forecastFile = fullfile(resultsDir, 'review_fourier_thresholds_v1', 'fourier', 'predictions.csv');
    eventFile = fullfile(resultsDir, 'review_storm_events_2021_2023_v1', 'events.csv');

    sites = ["ADIS"; "GUAM"; "KOUR"; "WUHN"; "SANT"; "POTS"; "CAS1"; "TRO1"; "NYAL"];
    modelColumns = {'transformer_dst', 'svmd_transformer_dst', ...
        'svmd_transformer_fixed12', 'pmsc_svmd_transformer'};
    models = ["Transformer + Dst"; "SVMD-Transformer + Dst (no penalty)"; ...
        "SVMD-Transformer + Dst (fixed 12 TECU)"; "SVMD-Transformer + Dst (adaptive b = 8 TECU)"];
    levels = ["weak"; "moderate"; "intense"];
    firstUtc = datetime(2021, 1, 1, 'TimeZone', 'UTC');
    nHours = hours(datetime(2024, 1, 1, 'TimeZone', 'UTC') - firstUtc);

    [values, dst, common] = loadForecasts(forecastFile, sites, modelColumns, firstUtc, nHours);
    [events, eventAtHour, levelOfEvent] = loadStormEvents(eventFile, firstUtc, dst, levels);
    scored = find(common & ismember(eventAtHour, find(levelOfEvent == 3)));
    nPairs = numel(scored);
    assert(nPairs > 0, 'No matched forecasts at disturbed origins in intense events.');
    eventRows = eventAtHour(scored);
    usedEvents = unique(eventRows);
    nEvents = numel(usedEvents);

    % Store one error per paired origin, model, and plotted location.
    reference = values(scored, :, 2);
    forecasts = values(scored, :, 3:end);
    errors = permute(abs(forecasts - reference), [1 3 2]);
    ordered = sort(errors, 1);
    quartiles = ordered(ceil(nPairs * [0.25 0.75]), :, :);
    medians = median(errors, 1);
    rmse = sqrt(mean(errors.^2, 1));
    fig = drawViolins(errors, quartiles, medians, rmse, sites, nEvents);

    if nargout > 1
        p95 = ordered(ceil(nPairs * 0.95), :, :);
        maximum = ordered(end, :, :);
        nModels = numel(models);
        nRows = numel(sites) * nModels;
        metrics = table(repelem(sites, nModels), repmat(models, numel(sites), 1), ...
            repmat(nPairs, nRows, 1), repmat(nEvents, nRows, 1), ...
            rmse(:), medians(:), p95(:), maximum(:), ...
            'VariableNames', {'Location', 'Model', 'N_pairs', 'N_events', ...
                'RMSE_TECU', 'Median_abs_TECU', 'P95_abs_TECU', 'Max_abs_TECU'});
    end
    if nargout > 2
        originUtc = firstUtc + hours(scored - 1);
        targetUtc = originUtc + hours(1);
        details.events = events(usedEvents, :);
        details.hours = table(originUtc, targetUtc, events.event_id(eventRows), dst(scored), ...
            'VariableNames', {'Nominal_origin_utc', 'Nominal_target_utc', 'Event_id', 'Dst_at_origin_nT'});
    end
    fprintf('Figure 9 | %d intense events | %d paired disturbed origins per location\n', nEvents, nPairs);
end

function [values, dst, common] = loadForecasts(file, sites, models, firstUtc, nHours)
    columns = [{'dst_t0_nt', 'actual_tecu'}, models];
    T = readRequiredCsv(file, {'site'}, [{'year', 'target_source_row_0based'}, columns]);

    [knownSite, siteIndex] = ismember(T.site, sites);
    row = T.target_source_row_0based;
    assert(all(knownSite & ismember(T.year, 2021:2023) & row == fix(row) & row >= 72 & row < 8760), ...
        'Expected nine-location one-hour test targets for 2021-2023 in %s.', file);

    yearStart = datetime(T.year, ones(height(T), 1), ones(height(T), 1), 'TimeZone', 'UTC');
    % Origin t = target t+1 minus one hour. The MATLAB origin index is one-based.
    originIndex = hours(yearStart - firstUtc) + row;
    nSites = numel(sites);
    rowIndex = sub2ind([nHours, nSites], originIndex, siteIndex);
    assert(numel(unique(rowIndex)) == height(T), 'Duplicate location and target hour in %s.', file);

    packed = nan(nHours * nSites, numel(columns));
    packed(rowIndex, :) = T{:, columns};
    values = reshape(packed, nHours, nSites, []);
    dstBySite = values(:, :, 1);
    dst = min(dstBySite, [], 2, 'omitnan');
    maxDst = max(dstBySite, [], 2, 'omitnan');
    knownDst = isfinite(dst);
    assert(all(dst(knownDst) == maxDst(knownDst)), ...
        'The saved origin Dst differs between locations at the same hour.');
    common = all(all(isfinite(values), 3), 2);
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

function utc = parseUtc(values)
    utc = datetime(values, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss''Z''', 'TimeZone', 'UTC');
    assert(~any(isnat(utc)), 'A required UTC timestamp could not be read.');
end

function T = readRequiredCsv(file, textColumns, numericColumns)
    assert(isfile(file), 'Missing input file: %s', file);
    opts = detectImportOptions(file, 'Delimiter', ',', 'Encoding', 'UTF-8', ...
        'VariableNamingRule', 'preserve');
    columns = [textColumns, numericColumns];
    missingColumns = setdiff(columns, opts.VariableNames, 'stable');
    assert(isempty(missingColumns), 'Missing columns in %s: %s', file, strjoin(missingColumns, ', '));
    opts.SelectedVariableNames = columns;
    opts = setvartype(opts, textColumns, 'string');
    opts = setvartype(opts, numericColumns, 'double');
    opts.ImportErrorRule = 'error';
    T = readtable(file, opts);
    assert(height(T) > 0 && ~any(ismissing(T{:, textColumns}), 'all'), 'Empty or invalid data in %s.', file);
end

function fig = drawViolins(errors, quartiles, medians, rmse, sites, nEvents)
    font = 'Times New Roman';
    colors = [213 94 0; 0 114 178; 0 135 102; 117 85 153] / 255;
    tickLabels = {'a', 'b', 'c', 'd'};
    legendLabels = {'a  Transformer', 'b  SVMD, no penalty', ...
        'c  SVMD, fixed 12 TECU', 'd  SVMD, adaptive b = 8 TECU'};

    fig = figure('Name', 'Figure 9 - One-hour VTEC error distributions', ...
        'NumberTitle', 'off', 'Color', 'w', 'Visible', 'on', ...
        'WindowStyle', 'normal', 'MenuBar', 'figure', 'ToolBar', 'figure', ...
        'Units', 'normalized', 'Position', [0.06 0.07 0.88 0.85], 'WindowState', 'maximized');
    left = [0.070 0.383 0.696];
    bottom = [0.661 0.385 0.109];
    panelSize = [0.295 0.269];
    [nPairs, nModels, nLocations] = size(errors);
    panels = gobjects(nLocations, 1);
    % The same reproducible horizontal offsets are used for every model.
    offsets = 0.26 * (mod((1:nPairs).' * ((sqrt(5) - 1) / 2), 1) - 0.5);
    for s = 1:nLocations
        row = ceil(s / 3);
        col = mod(s - 1, 3) + 1;
        ax = axes('Parent', fig, 'Units', 'normalized', ...
            'Position', [left(col), bottom(row), panelSize], ...
            'PositionConstraint', 'innerposition', 'Box', 'on', 'TickDir', 'in', ...
            'FontName', font, 'FontSize', 10, 'LineWidth', 0.8, ...
            'YGrid', 'on', 'GridLineStyle', '--', 'GridAlpha', 0.18, ...
            'Tag', char(sites(s)));
        panels(s) = ax;
        hold(ax, 'on');
        % Leave two clear lines for the station name and the RMSE values.
        maxError = max(errors(:, :, s), [], 'all');
        densityScale = max(1, 1.09 * maxError);
        upper = max(1, 1.44 * maxError);
        labelY = 0.79 * upper;
        set(ax, 'XLim', [0.60 nModels + 0.40], 'XTick', 1:nModels, ...
            'XTickLabel', [], 'YLim', [0 upper]);
        if row == 3
            ax.XTickLabel = tickLabels;
            ax.XAxis.FontSize = 14;
        end
        for m = 1:nModels
            vals = errors(:, m, s);
            color = colors(m, :);
            drawViolin(ax, m, vals, densityScale, color);
            scatter(ax, m + offsets, vals, 9, 'filled', 'MarkerFaceColor', color, ...
                'MarkerFaceAlpha', 0.28, 'MarkerEdgeColor', 'none');
            plot(ax, [m m], quartiles(:, m, s), '-', 'Color', [0.23 0.23 0.23], 'LineWidth', 2.2);
            middle = medians(1, m, s);
            plot(ax, m + [-0.105 0.105], [middle middle], 'k-', 'LineWidth', 1.3);
            text(ax, m, labelY, sprintf('%.2f', rmse(1, m, s)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Color', 0.75 * color, 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
        end
        text(ax, 0.025, 0.98, sprintf('(%c) %s', 'a' + s - 1, char(sites(s))), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 13, ...
            'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 1, 'Interpreter', 'none');
    end

    ylabel(panels(4), 'Absolute forecast error (TECU)', 'FontSize', 12);
    legendItems = gobjects(1, nModels);
    for m = 1:nModels
        legendItems(m) = plot(panels(1), nan, nan, 's', 'LineStyle', 'none', ...
            'MarkerSize', 6, 'MarkerFaceColor', colors(m, :), 'MarkerEdgeColor', colors(m, :));
    end
    lgd = legend(panels(1), legendItems, legendLabels, 'Orientation', 'horizontal', ...
        'NumColumns', nModels, 'Box', 'off', 'Units', 'normalized', ...
        'FontName', font, 'FontSize', 12, 'Interpreter', 'none', 'AutoUpdate', 'off');
    notes = {
        'RMSE (TECU) above violins  |  Hourly absolute errors (dots)  |  Median (horizontal)  |  Q1-Q3 (vertical)'
        sprintf('1-hour forecasts with Dst  |  %d intense events  |  %d paired hours per location', nEvents, nPairs)
    };
    annotation(fig, 'textbox', [left(1) 0.006 left(3) + panelSize(1) - left(1) 0.061], ...
        'String', notes, 'Margin', 0, 'FitBoxToText', 'off', ...
        'LineStyle', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontName', font, 'FontSize', 12, 'Interpreter', 'none');
    drawnow;
    p = lgd.Position;
    lgd.Position = [(1 - p(3)) / 2, 0.991 - p(4), p(3:4)];
    plotedit(fig, 'on');
end

function drawViolin(ax, center, values, upper, color)
    bounds = [min(values), max(values)];
    if bounds(1) == bounds(2)
        return;
    end

    % Reflect the kernel at zero and draw it over the observed error range.
    bandwidth = max(upper / 200, 1.06 * std(values, 1) * numel(values)^(-1/5));
    y = linspace(bounds(1), bounds(2), 160).';
    direct = (y - values.') / bandwidth;
    reflected = (y + values.') / bandwidth;
    density = mean(exp(-0.5 * direct.^2) + exp(-0.5 * reflected.^2), 2);
    width = 0.32 * density / max(density);
    patch(ax, [center - width; flipud(center + width)], [y; flipud(y)], color, ...
        'FaceAlpha', 0.23, 'EdgeColor', color, 'LineWidth', 0.9);
end
