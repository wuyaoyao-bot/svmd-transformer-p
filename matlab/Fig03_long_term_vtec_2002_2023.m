function fig = Fig03_long_term_vtec_2002_2023(dataDir)
%FIG03_LONG_TERM_VTEC_2002_2023 Plot smoothed VTEC at NYAL, POTS, and ADIS.

    if nargin == 0
        dataDir = fullfile('data', 'legacy_locations');
    end
    locations = {'NYAL', 'POTS', 'ADIS'};
    latitudeLabels = {'High latitude (78.9°N)', 'Mid latitude (52.4°N)', ...
        'Equatorial (9.0°N)'};
    fig = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [5, 5, 15, 12], 'PaperPositionMode', 'auto');
    layout = tiledlayout(fig, 3, 1, 'TileSpacing', 'tight', 'Padding', 'compact');

    for index = 1:numel(locations)
        location = locations{index};
        file = fullfile(dataDir, sprintf( ...
            '%s_SVMD_Physics_Residual_OMNI_2002_2023.csv', location));
        source = readtable(file, 'VariableNamingRule', 'preserve');
        vtec = source.VTEC_True;

        year = 2002 + (0:numel(vtec)-1)' / (365.25 * 24);
        smoothed = smoothdata(vtec, 'movmean', 24);
        rows = 1:24:numel(vtec);
        displayed = smoothed(rows);

        ax = nexttile(layout);
        plot(ax, year(rows), displayed, '.', 'Color', [0.1, 0.4, 0.8], ...
            'MarkerSize', 3.5);
        grid(ax, 'on');
        box(ax, 'on');
        set(ax, 'XLim', [2002, 2024], 'XTick', 2002:5:2022, ...
            'FontName', 'Times New Roman', 'FontSize', 12);
        ylim(ax, [0, max(40, ceil(max(displayed) / 20) * 20)]);

        label = sprintf('%c. %s [%s]', 'a' + index - 1, location, latitudeLabels{index});
        text(ax, 0.02, 0.85, label, 'Units', 'normalized', 'BackgroundColor', 'w', ...
            'EdgeColor', [0.8, 0.8, 0.8], 'FontName', 'Times New Roman', ...
            'FontSize', 12, 'FontWeight', 'bold');
        if index < numel(locations)
            xticklabels(ax, []);
        else
            xlabel(ax, 'Year', 'FontName', 'Times New Roman', 'FontSize', 14, ...
                'FontWeight', 'bold');
        end
    end
    ylabel(layout, 'VTEC (TECU)', 'FontName', 'Times New Roman', ...
        'FontSize', 14, 'FontWeight', 'bold');
end
