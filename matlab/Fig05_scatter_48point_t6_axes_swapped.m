function fig = Fig05_scatter_48point_t6_axes_swapped(dataDir, startRow)
%FIG05_SCATTER_48POINT_T6_AXES_SWAPPED Open the original 9-by-8 Figure 5.
%   Fig05_scatter_48point_t6_axes_swapped uses data rows 50:97 from the
%   original Results_t6_Final folder. The rows exclude the CSV header.
%   Fig05_scatter_48point_t6_axes_swapped(dataDir, startRow) changes the
%   prediction folder or the starting row of the 48-sample window.
%   Forecast VTEC is on x and the CODE GIM reference is on y.
%   Select an axes border in Plot Edit mode to adjust a panel.

    if nargin < 1
        dataDir = fullfile('data', 'Results_t6_Final');
    end
    if nargin < 2
        startRow = 50;
    end
    validateattributes(startRow, {'numeric'}, {'scalar', 'finite', 'integer', 'positive'});
    rows = double(startRow) + (0:47);

    sites = {'ADIS', 'GUAM', 'KOUR', 'WUHN', 'SANT', 'POTS', 'CAS1', 'TRO1', 'NYAL'};
    fileSites = sites;
    fileSites{4} = 'Wuhan';
    models = {'CNN', 'LSTM', 'Bi-LSTM', 'Transformer', ...
        'SVMD-CNN', 'SVMD-LSTM', 'SVMD-BiLSTM', 'SVMD-Transformer'};
    columns = [{'True_VTEC'}, models];

    % Read every location and model before opening the matrix.
    nSites = numel(sites);
    nModels = numel(models);
    samples = zeros(numel(rows), numel(columns), nSites);
    for s = 1:nSites
        file = fullfile(dataDir, [fileSites{s}, '_Predictions_t6.csv']);
        assert(isfile(file), 'Missing prediction file for %s: %s', sites{s}, file);
        opts = detectImportOptions(file, 'Delimiter', ',', 'VariableNamingRule', 'preserve');
        missing = setdiff(columns, opts.VariableNames, 'stable');
        assert(isempty(missing), 'Missing columns in %s: %s', file, strjoin(missing, ', '));
        opts.SelectedVariableNames = columns;
        opts = setvartype(opts, columns, 'double');
        T = readtable(file, opts);
        assert(height(T) >= rows(end), 'Expected at least %d data rows in %s.', rows(end), file);
        samples(:, :, s) = T{rows, columns};
        assert(all(isfinite(samples(:, :, s)), 'all'), ...
            'Rows %d-%d contain missing or invalid VTEC values in %s.', rows(1), rows(end), file);
    end

    font = 'Times New Roman';
    baseColor = [0.40 0.40 0.40];
    svmdTransformerColor = [0 0.25 0.75];
    bestColor = [1 0.96 0.85];
    fig = figure('Name', 'Figure 5 - Six-hour VTEC forecasts', 'NumberTitle', 'off', ...
        'Color', 'w', 'Visible', 'on', 'WindowStyle', 'normal', ...
        'MenuBar', 'figure', 'ToolBar', 'figure', ...
        'Units', 'normalized', 'Position', [0.06 0.08 0.88 0.82]);

    labelAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', [0 0 1 1], ...
        'Visible', 'off', 'HitTest', 'off');
    text(labelAxes, 0.54, 0.024, 'Forecast VTEC (TECU)', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontName', font, 'FontSize', 13, 'FontWeight', 'bold');
    text(labelAxes, 0.021, 0.51, 'CODE GIM reference VTEC (TECU)', 'Units', 'normalized', ...
        'Rotation', 90, 'HorizontalAlignment', 'center', 'FontName', font, ...
        'FontSize', 13, 'FontWeight', 'bold');

    % Each panel has an independent position so it can be moved by hand.
    gapX = 0.008;
    gapY = 0.009;
    panelWidth = (0.895 - (nModels - 1) * gapX) / nModels;
    panelHeight = (0.858 - (nSites - 1) * gapY) / nSites;
    left = 0.090 + (0:nModels-1) * (panelWidth + gapX);
    bottom = 0.938 - panelHeight - (0:nSites-1) * (panelHeight + gapY);

    for s = 1:nSites
        rowData = samples(:, :, s);
        reference = rowData(:, 1);
        predictions = rowData(:, 2:end);
        correlations = corrcoef(rowData);
        rSquared = correlations(1, 2:end).^2;
        [~, bestModel] = max(rSquared, [], 'omitnan');
        limits = [min(rowData, [], 'all'), max(rowData, [], 'all')];
        pad = 0.10 * diff(limits);
        if pad == 0
            pad = 1;
        end
        limits = limits + [-pad pad];

        for m = 1:nModels
            dotColor = baseColor;
            titleColor = [0 0 0];
            if strcmp(models{m}, 'SVMD-Transformer')
                dotColor = svmdTransformerColor;
                titleColor = svmdTransformerColor;
            end
            panelColor = [1 1 1];
            labelColor = titleColor;
            metric = 'R^{2} = n/a';
            if isfinite(rSquared(m))
                metric = sprintf('R^{2} = %.3f', rSquared(m));
                if m == bestModel
                    panelColor = bestColor;
                    labelColor = [0.85 0.10 0.10];
                end
            end

            ax = axes('Parent', fig, 'Units', 'normalized', ...
                'Position', [left(m), bottom(s), panelWidth, panelHeight], ...
                'PositionConstraint', 'innerposition', 'Color', panelColor, ...
                'FontName', font, 'FontSize', 8.5, 'LineWidth', 0.7, ...
                'Box', 'on', 'Layer', 'top', 'GridColor', [0.85 0.85 0.85]);
            hold(ax, 'on');
            grid(ax, 'on');
            predicted = predictions(:, m);
            scatter(ax, predicted, reference, 15, 'filled', ...
                'MarkerFaceColor', dotColor, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');
            plot(ax, limits, limits, 'k--', 'LineWidth', 1.2);
            if any(reference ~= reference(1))
                % The original fit was forecast versus reference. Swap its coordinates too.
                fit = polyfit(reference, predicted, 1);
                plot(ax, polyval(fit, limits), limits, 'r-', 'LineWidth', 1.2);
            end

            xlim(ax, limits);
            ylim(ax, limits);
            xTicks = xticks(ax);
            yTicks = yticks(ax);
            xticks(ax, xTicks(xTicks >= 0));
            yticks(ax, yTicks(yTicks >= 0));
            text(ax, 0.05, 0.95, metric, 'Units', 'normalized', ...
                'FontName', font, 'FontSize', 9.5, 'FontWeight', 'bold', 'Color', labelColor, ...
                'BackgroundColor', panelColor, 'Margin', 1, ...
                'VerticalAlignment', 'top', 'Interpreter', 'tex');

            if s == 1
                title(ax, models{m}, 'Interpreter', 'none', 'FontName', font, ...
                    'FontSize', 10.5, 'FontWeight', 'bold', 'Color', titleColor);
            end
            if m == 1
                ylabel(ax, sites{s}, 'FontName', font, 'FontSize', 10, 'FontWeight', 'bold');
            else
                yticklabels(ax, []);
            end
            if s ~= nSites
                xticklabels(ax, []);
            end
        end
    end
    drawnow;
    plotedit(fig, 'on');
end
