function fig = Fig06_taylor_diagram_t6(data_dir)
%FIG06_TAYLOR_DIAGRAM_T6 Plot the five-model Taylor diagram.
% The coordinate calculation follows the archived plotting script.

if nargin == 0
    data_dir = fullfile('data', 'Results_t6_Final');
end

stations_display = {'ADIS', 'GUAM', 'KOUR', 'WUHN', 'SANT', 'POTS', 'CAS1', 'TRO1', 'NYAL'};
stations_files   = {'ADIS', 'GUAM', 'KOUR', 'Wuhan', 'SANT', 'POTS', 'CAS1', 'TRO1', 'NYAL'};

models = {'CNN', 'LSTM', 'Bi-LSTM', 'Transformer'};
target_model = 'SVMD-Transformer';

color_baselines = {
    [0.85, 0.32, 0.09], [0.92, 0.69, 0.12], [0.49, 0.18, 0.55], [0.46, 0.67, 0.18]
};
marker_baselines = {'o', 's', 'd', '^'};
color_svmd_trans = [0, 0.25, 0.75];
marker_svmd_trans = 'p';

fig = figure('Position', [50, 50, 1400, 1300], 'Color', 'w');
t = tiledlayout(fig, 3, 3, 'TileSpacing', 'none', 'Padding', 'compact');

handles = gobjects(1, 6);

for i = 1:9
    display_name = stations_display{i};
    file_name    = stations_files{i};

    csv_file = fullfile(data_dir, [file_name, '_Predictions_t6.csv']);
    assert(isfile(csv_file), 'Missing forecast file: %s', csv_file);

    df = readtable(csv_file, detectImportOptions(csv_file, 'VariableNamingRule', 'preserve'));
    y_true = df.("True_VTEC");
    sd_true = std(y_true);

    ax_main = nexttile(t);

    max_sd = 1.5;
    is_left = (mod(i-1, 3) == 0);
    is_bottom = (i >= 7);
    draw_taylor_background(ax_main, max_sd, is_bottom, is_left);
    hold(ax_main, 'on');

    handles(1) = plot(ax_main, 1, 0, 'k*', 'MarkerSize', 12, 'LineWidth', 1.5, 'MarkerFaceColor', 'k');

    for m = 1:4
        m_name = models{m};
        y_pred = df.(m_name);

        R_val = max(corrcoef(y_true, y_pred), 0); R_val = R_val(1, 2);
        sd_norm = std(y_pred) / sd_true;

        X_base = sd_norm * R_val;
        Y_base = sd_norm * sin(acos(R_val));

        handles(m+1) = plot(ax_main, X_base, Y_base, marker_baselines{m}, ...
            'MarkerSize', 10, 'LineWidth', 1.2, 'MarkerFaceColor', color_baselines{m}, 'MarkerEdgeColor', 'k');
    end

    y_pred_ours = df.(target_model);
    R_val_ours = max(corrcoef(y_true, y_pred_ours), 0); R_val_ours = R_val_ours(1,2);
    sd_norm_ours = std(y_pred_ours) / sd_true;

    X_ours = sd_norm_ours * R_val_ours;
    Y_ours = sd_norm_ours * sin(acos(R_val_ours));

    handles(6) = plot(ax_main, X_ours, Y_ours, marker_svmd_trans, ...
        'MarkerSize', 18, 'LineWidth', 1.5, 'MarkerFaceColor', color_svmd_trans, 'MarkerEdgeColor', 'k');

    panel_letter = char(96 + i);
    text(ax_main, 0.96, 0.96, sprintf('\\bf{%c. %s}', panel_letter, display_name), ...
         'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
         'FontName', 'Times New Roman', 'FontSize', 16, 'Interpreter', 'tex', ...
         'BackgroundColor', [1 1 1], 'EdgeColor', [0.8 0.8 0.8], 'Margin', 4);
    if i == 1
        legendAxes = ax_main;
        legendHandles = handles;
    end
end
legend_labels = [{'CODE GIM reference'}, models, {target_model}];
lg = legend(legendAxes, legendHandles, legend_labels, ...
            'Orientation', 'horizontal', 'NumColumns', 6, 'FontSize', 15, ...
            'FontName', 'Times New Roman', 'Location', 'north');
lg.Box = 'off';
lg.Layout.Tile = 'north';

end

function draw_taylor_background(ax, max_sd, is_bottom, is_left)
    hold(ax, 'on'); axis(ax, 'equal');

    xlim(ax, [-max_sd*0.09, max_sd*1.12]);
    ylim(ax, [-max_sd*0.09, max_sd*1.12]);

    theta = linspace(0, pi/2, 100);

    for r = 0.25 : 0.25 : max_sd
        plot(ax, r*cos(theta), r*sin(theta), ':', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.2);
        if r ~= 1
            text(ax, r, -max_sd*0.035, sprintf('%.2f', r), 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontName', 'Times New Roman');
            text(ax, -max_sd*0.035, r, sprintf('%.2f', r), 'HorizontalAlignment', 'right', 'FontSize', 10, 'FontName', 'Times New Roman');
        end
    end
    plot(ax, 1*cos(theta), 1*sin(theta), '--', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5);
    text(ax, 1, -max_sd*0.035, '1.0', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    text(ax, -max_sd*0.035, 1, '1.0', 'HorizontalAlignment', 'right', 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

    R_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
    for R = R_ticks
        ang = acos(R);
        plot(ax, [0, max_sd*cos(ang)], [0, max_sd*sin(ang)], ':', 'Color', [0.6 0.6 0.8], 'LineWidth', 1);
        text(ax, 1.04*max_sd*cos(ang), 1.04*max_sd*sin(ang), sprintf('%.2f', R), 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0.1 0.1 0.6], 'FontName', 'Times New Roman', 'Rotation', ang*180/pi - 90);
    end
    text(ax, 0, 1.04*max_sd, '0.0', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0.1 0.1 0.6]);
    text(ax, 1.04*max_sd, 0, '1.0', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0.1 0.1 0.6]);

    plot(ax, [0, max_sd], [0, 0], 'k-', 'LineWidth', 1.5);
    plot(ax, [0, 0], [0, max_sd], 'k-', 'LineWidth', 1.5);
    plot(ax, max_sd*cos(theta), max_sd*sin(theta), 'k-', 'LineWidth', 1.5);
    ax.XAxis.Visible = 'off'; ax.YAxis.Visible = 'off';

    if is_bottom
        text(ax, max_sd*0.5, -max_sd*0.08, 'Normalized Standard Deviation', 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    end
    if is_left
        text(ax, -max_sd*0.08, max_sd*0.5, 'Normalized Standard Deviation', 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold', 'FontName', 'Times New Roman', 'Rotation', 90);
    end
end
