function [fig, metrics, details] = Fig08_intense_storm_1h(projectRoot, eventRank)
%FIG08_INTENSE_STORM_1H Plot the one-hour forecasts for a catalogue storm.
%   Run from the project root or pass the project folder as projectRoot.
%   The figure is editable and can be saved manually from the figure window.
%   Blue RMSE labels indicate a decrease and red labels indicate an increase.
%   [fig, metrics, details] also returns the per-location RMSE, catalogue
%   event, ranked candidates, and the nominal origin and target hours.
%   The default is rank 2 among intense events with a window inside the test
%   period and matched forecasts at the Dst minimum. Ranks are ordered by
%   lowest Dst, then earliest minimum. Pass eventRank to choose another.

    if nargin == 0
        projectRoot = pwd;
    end
    if nargin < 2
        eventRank = 2;
    end
    validateattributes(eventRank, {'numeric'}, {'real', 'scalar', 'finite', 'integer', 'positive'}, ...
        mfilename, 'eventRank', 2);

    resultsDir = fullfile(projectRoot, 'results');
    forecastFile = fullfile(resultsDir, 'review_fourier_thresholds_v1', 'fourier', 'predictions.csv');
    eventFile = fullfile(resultsDir, 'review_storm_events_2021_2023_v1', 'events.csv');

    sites = ["ADIS"; "GUAM"; "KOUR"; "WUHN"; "SANT"; "POTS"; "CAS1"; "TRO1"; "NYAL"];
    modelColumns = {'transformer_dst', 'svmd_transformer_dst'};
    firstUtc = datetime(2021, 1, 1, 'TimeZone', 'UTC');
    nHours = hours(datetime(2024, 1, 1, 'TimeZone', 'UTC') - firstUtc);

    [values, dst, common] = loadForecasts(forecastFile, sites, modelColumns, firstUtc, nHours);
    [events, peaks] = loadStormEvents(eventFile, firstUtc, dst);
    before = 24;
    after = 47;
    eligible = find(events.min_dst_nt <= -100 & peaks > before & ...
        peaks + after <= nHours & common(peaks));
    assert(~isempty(eligible), ...
        'No intense event has a full window in the test period and matched forecasts at its Dst minimum.');
    [~, order] = sortrows([events.min_dst_nt(eligible), peaks(eligible)], [1 2]);
    eligible = eligible(order);
    candidatePeaks = peaks(eligible);
    cumulativePairs = [0; cumsum(common)];
    pairedHours = cumulativePairs(candidatePeaks + after + 1) - cumulativePairs(candidatePeaks - before);
    nominalPeaks = firstUtc + hours(candidatePeaks - 1);
    nominalPeaks.Format = 'yyyy-MM-dd HH:mm';
    candidates = table((1:numel(eligible)).', events.event_id(eligible), events.min_dst_nt(eligible), ...
        nominalPeaks, pairedHours, ...
        'VariableNames', {'Rank', 'Event_id', 'Min_Dst_nT', 'Nominal_peak_utc', 'Paired_hours'});
    fprintf('Figure 8 candidates, ordered by lowest Dst (paired hours out of %d):\n', before + after + 1);
    disp(candidates);
    assert(eventRank <= height(candidates), ...
        'Requested event rank %d, but only %d candidates are available. Pass a listed rank as the second argument.', ...
        eventRank, height(candidates));
    eventRow = eligible(eventRank);
    event = events(eventRow, :);
    peak = peaks(eventRow);
    rows = (peak - before : peak + after).';
    elapsedHours = (0 : numel(rows) - 1).';
    originUtc = firstUtc + hours(rows - 1);
    targetUtc = originUtc + hours(1);

    % Keep the time grid intact. Both models use the same hours at all nine sites.
    paired = common(rows);
    vtec = permute(values(rows, :, 2:end), [1 3 2]);
    vtec(~paired, :, :) = nan;
    delta = vtec(paired, 2:end, :) - vtec(paired, 1, :);
    rmse = squeeze(sqrt(mean(delta.^2, 1))).';
    nPairs = nnz(paired);
    reductionPct = nan(numel(sites), 1);
    positiveBaseline = rmse(:, 1) > 0;
    reductionPct(positiveBaseline) = 100 * ...
        (rmse(positiveBaseline, 1) - rmse(positiveBaseline, 2)) ./ rmse(positiveBaseline, 1);

    peakUtc = char(string(firstUtc + hours(peak - 1), 'yyyy-MM-dd HH:mm'));
    dstLabel = sprintf('Forecast-origin Dst | Event minimum %g nT at %s (nominal UTC)', ...
        event.min_dst_nt, peakUtc);
    fig = drawStorm(elapsedHours, dst(rows), vtec, rmse, reductionPct, sites, before, dstLabel);

    if nargout > 1
        metrics = table(sites, repmat(nPairs, numel(sites), 1), rmse(:, 1), rmse(:, 2), reductionPct, ...
            'VariableNames', {'Location', 'N_pairs', 'Transformer_Dst_RMSE_TECU', ...
                'SVMD_Transformer_Dst_RMSE_TECU', 'RMSE_reduction_pct'});
    end
    if nargout > 2
        details.event = event;
        details.event_rank = eventRank;
        details.candidates = candidates;
        details.hours = table(originUtc, targetUtc, dst(rows), paired, ...
            'VariableNames', {'Nominal_origin_utc', 'Nominal_target_utc', 'Dst_at_origin_nT', 'Included_in_RMSE'});
    end
    fprintf('Figure 8 | Rank %d | %s | nominal peak %s UTC | minimum Dst %g nT | %d/%d paired origins\n', ...
        eventRank, char(event.event_id), peakUtc, event.min_dst_nt, nPairs, numel(rows));
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

function [events, peak] = loadStormEvents(file, firstUtc, dst)
    events = readRequiredCsv(file, ...
        {'event_id', 'storm_level', 'first_disturbed_utc', 'peak_dst_utc', 'last_disturbed_utc'}, ...
        {'min_dst_nt', 'disturbed_origin_hours'});
    assert(numel(unique(events.event_id)) == height(events), 'Duplicate event ID in %s.', file);
    levels = ["weak"; "moderate"; "intense"];
    severity = 1 + (events.min_dst_nt <= -50) + (events.min_dst_nt <= -100);
    assert(all(isfinite(events.min_dst_nt) & events.min_dst_nt <= -30 & ...
        lower(events.storm_level) == levels(severity)), 'Invalid storm categories in %s.', file);

    utcFormat = 'yyyy-MM-dd''T''HH:mm:ss''Z''';
    first = datetime(events.first_disturbed_utc, 'InputFormat', utcFormat, 'TimeZone', 'UTC');
    minimum = datetime(events.peak_dst_utc, 'InputFormat', utcFormat, 'TimeZone', 'UTC');
    last = datetime(events.last_disturbed_utc, 'InputFormat', utcFormat, 'TimeZone', 'UTC');
    first = hours(first - firstUtc) + 1;
    peak = hours(minimum - firstUtc) + 1;
    last = hours(last - firstUtc) + 1;
    indices = [first, peak, last];
    nDisturbed = events.disturbed_origin_hours;
    assert(all(indices == fix(indices), 'all') && ...
        all(1 <= first & first <= peak & peak <= last & last <= numel(dst) & ...
            nDisturbed == fix(nDisturbed) & nDisturbed >= 1 & nDisturbed <= last - first + 1), ...
        'Invalid hourly event bounds or disturbed-hour counts in %s.', file);

    [starts, order] = sort(first);
    ends = last(order);
    assert(all(ends(1:end-1) < starts(2:end)), 'Overlapping event windows in %s.', file);
    eventAtHour = zeros(size(dst));
    for e = 1:height(events)
        eventAtHour(first(e):last(e)) = e;
    end
    disturbed = isfinite(dst) & dst <= -30;
    matched = eventAtHour(disturbed);
    assert(any(disturbed) && all(matched > 0), ...
        'No disturbed prediction hours or incomplete event coverage in %s.', file);
    counts = accumarray(matched, ones(nnz(disturbed), 1), [height(events), 1]);
    hasPeakDst = isfinite(dst(peak));
    assert(all(dst(disturbed) >= events.min_dst_nt(matched)) && ...
        all(dst(peak(hasPeakDst)) == events.min_dst_nt(hasPeakDst)) && all(counts <= nDisturbed), ...
        'The event catalogue and saved forecast Dst disagree.');
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

function fig = drawStorm(time, dst, vtec, rmse, reductionPct, sites, peakHour, dstLabel)
    font = 'Times New Roman';
    colors = [213 94 0; 0 114 178] / 255;
    lineStyles = {'--', '-'};
    lowerRmseColor = [0 87 155] / 255;
    higherRmseColor = [178 34 43] / 255;
    neutralColor = [0.30 0.31 0.33];
    fig = figure('Name', 'Figure 8 - One-hour VTEC forecasts during a storm', 'NumberTitle', 'off', ...
        'Color', 'w', 'Visible', 'on', 'WindowStyle', 'normal', ...
        'MenuBar', 'figure', 'ToolBar', 'figure', ...
        'Units', 'normalized', 'Position', [0.06 0.07 0.88 0.85], 'WindowState', 'maximized');

    left = [0.074 0.380 0.686];
    width = 0.282;
    dstAx = makeAxes(fig, [left(1), 0.850, left(3) + width - left(1), 0.094], font, 'Dst');
    shadeCurve(dstAx, time, dst, [0.95 0.76 0.76], 0.35);
    xline(dstAx, peakHour, ':', 'Color', [0.48 0.48 0.48], 'LineWidth', 0.85);
    yline(dstAx, -100, '--', 'Color', [0.78 0.24 0.24], 'LineWidth', 0.85);
    plot(dstAx, time, dst, 'Color', [0.16 0.16 0.16], 'LineWidth', 1.55);
    dstMin = min([-100; dst], [], 'omitnan');
    dstMax = max([0; dst], [], 'omitnan');
    dstSpan = dstMax - dstMin;
    ylim(dstAx, [dstMin - 0.08 * dstSpan, dstMax + 0.28 * dstSpan]);
    ylabel(dstAx, 'Dst (nT)', 'FontWeight', 'bold');
    text(dstAx, 0.012, 0.95, dstLabel, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 1, 'Interpreter', 'none');

    % Independent axes keep the individual panels movable in Plot Edit.
    bottom = [0.589 0.330 0.071];
    errorHeight = 0.076;
    vtecHeight = 0.157;
    panelGap = 0.007;
    nSites = numel(sites);
    vtecAxes = gobjects(nSites, 1);
    errorAxes = gobjects(nSites, 1);
    for s = 1:nSites
        row = ceil(s / 3);
        col = mod(s - 1, 3) + 1;
        site = char(sites(s));
        errAx = makeAxes(fig, [left(col), bottom(row), width, errorHeight], font, [site, '_error']);
        vtecBottom = bottom(row) + errorHeight + panelGap;
        vtecAx = makeAxes(fig, [left(col), vtecBottom, width, vtecHeight], font, site);
        vtecAxes(s) = vtecAx;
        errorAxes(s) = errAx;

        reference = vtec(:, 1, s);
        predicted = vtec(:, 2:end, s);
        absoluteError = abs(predicted - reference);
        for m = 1:2
            shadeCurve(errAx, time, absoluteError(:, m), colors(m, :), 0.16);
        end
        xline(vtecAx, peakHour, ':', 'Color', [0.60 0.60 0.60]);
        xline(errAx, peakHour, ':', 'Color', [0.60 0.60 0.60]);
        h = gobjects(1, 3);
        h(1) = plot(vtecAx, time, reference, 'k-', 'LineWidth', 1.45);
        for m = 1:2
            h(m + 1) = plot(vtecAx, time, predicted(:, m), lineStyles{m}, ...
                'Color', colors(m, :), 'LineWidth', 1.35);
            plot(errAx, time, absoluteError(:, m), lineStyles{m}, ...
                'Color', colors(m, :), 'LineWidth', 0.95);
        end

        limits = [min(vtec(:, :, s), [], 'all', 'omitnan'), max(vtec(:, :, s), [], 'all', 'omitnan')];
        span = max(1, diff(limits));
        ylim(vtecAx, limits + [-0.07 0.30] * span);
        ylim(errAx, [0, 1.48 * max(1, max(absoluteError, [], 'all', 'omitnan'))]);
        text(vtecAx, 0.025, 0.97, sprintf('(%c) %s', 'a' + s - 1, site), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', 'FontWeight', 'bold', ...
            'FontSize', 11, 'BackgroundColor', 'w', 'Margin', 1);

        change = reductionPct(s);
        changeColor = neutralColor;
        changeLabel = 'RMSE change n/a';
        if change > 0
            changeLabel = sprintf('RMSE decreased %.1f%%', change);
            changeColor = lowerRmseColor;
        elseif change < 0
            changeLabel = sprintf('RMSE increased %.1f%%', -change);
            changeColor = higherRmseColor;
        elseif change == 0
            changeLabel = 'RMSE unchanged';
        end
        text(vtecAx, 0.975, 0.97, changeLabel, 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
            'FontSize', 10.5, 'FontWeight', 'bold', 'Color', changeColor, ...
            'BackgroundColor', 'w', 'Margin', 1, 'Interpreter', 'none');
        rmseLabel = sprintf('RMSE %.2f \\rightarrow %.2f TECU', rmse(s, 1), rmse(s, 2));
        text(errAx, 0.975, 0.96, rmseLabel, 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
            'FontSize', 9, 'Color', neutralColor, ...
            'BackgroundColor', 'w', 'Margin', 0.5, 'Interpreter', 'tex');

        if col == 1
            ylabel(vtecAx, 'VTEC (TECU)', 'FontWeight', 'bold');
            ylabel(errAx, '|Error| (TECU)', 'FontWeight', 'bold');
        end
        if s == 1
            legendItems = h;
        end
    end

    allAxes = [dstAx; vtecAxes; errorAxes];
    set(allAxes, 'XLim', [time(1), time(end)], 'XTick', 0:12:time(end));
    linkaxes(allAxes, 'x');
    set([dstAx; vtecAxes; errorAxes(1:6)], 'XTickLabel', []);
    xlabel(errorAxes(8), 'Hours since first forecast origin', 'FontWeight', 'bold');
    lgd = legend(vtecAxes(1), legendItems, ...
        {'CODE GIM reference (t+1)', 'Transformer + Dst', 'SVMD-Transformer + Dst (no penalty)'}, ...
        'Orientation', 'horizontal', 'NumColumns', 3, 'Box', 'off', ...
        'FontName', font, 'FontSize', 11, 'Units', 'normalized', 'Interpreter', 'none');
    drawnow;
    p = lgd.Position;
    lgd.Position = [(1 - p(3)) / 2, 0.991 - p(4), p(3:4)];
    plotedit(fig, 'on');
end

function shadeCurve(ax, time, values, color, alpha)
    % Draw one patch per finite run so gaps remain visible.
    bounds = find(diff([false; isfinite(values); false]));
    for k = 1:2:numel(bounds)
        span = bounds(k):bounds(k + 1) - 1;
        t = time(span);
        patch(ax, [t; flipud(t)], [values(span); zeros(size(t))], color, ...
            'FaceAlpha', alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

function ax = makeAxes(fig, position, font, tag)
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', position, ...
        'PositionConstraint', 'innerposition', 'FontName', font, 'FontSize', 9, ...
        'Box', 'on', 'TickDir', 'in', 'LineWidth', 0.7, 'Layer', 'top', ...
        'GridLineStyle', '--', 'GridAlpha', 0.16, 'Tag', tag);
    hold(ax, 'on');
    grid(ax, 'on');
end
