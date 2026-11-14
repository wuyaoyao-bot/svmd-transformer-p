function fig = Fig04_vmd_modes_3d(base_dir)
%FIG04_VMD_MODES_3D Plot seven VMD modes and the residual.

if nargin == 0
    base_dir = fullfile('data', 'legacy_locations');
end
station = 'ADIS';

file_path = fullfile(base_dir, sprintf('%s_SVMD_Physics_Residual_OMNI_2002_2023.csv', station));
df = readtable(file_path, 'VariableNamingRule', 'preserve');
vtec_raw = df.VTEC_True;
time_raw = datetime(2002,1,1) + hours(0:length(vtec_raw)-1)';

start_date = datetime(2015, 3, 16);
end_date = datetime(2015, 3, 22, 23, 0, 0);
event_mask = (time_raw >= start_date) & (time_raw <= end_date);

time_event = time_raw(event_mask);
vtec_event = vtec_raw(event_mask);

[imfs, residual] = vmd(vtec_event, 'NumIMFs', 7);
num_imfs = size(imfs, 2);
Z_matrix = [imfs, residual]';
num_components = size(Z_matrix, 1);

y_indices = zeros(1, num_components);
for k = 1:num_imfs
    y_indices(k) = k;
end
y_indices(end) = num_imfs + 2.5;

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [5, 5, 15, 10], 'PaperPositionMode', 'auto');
ax = axes('Position', [0.18, 0.15, 0.75, 0.78]);
hold on; box on; grid on;

colors = winter(num_imfs);
colors(end+1, :) = [0.8, 0.2, 0.2];

x_num = datenum(time_event)';

for i = 1:num_components
    X = x_num;
    Y = ones(size(X)) * y_indices(i);
    Z = Z_matrix(i, :);

    lw = 1.2;
    tag = sprintf('IMF %d', i);
    if i == num_components
        lw = 2.2; tag = 'Residual';
    elseif i >= num_imfs - 1
        lw = 1.6;
    end

    plot3(ax, X, Y, Z, 'Color', colors(i,:), 'LineWidth', lw);

    text(ax, X(1)-0.2, Y(1), Z(1), tag, ...
        'Color', colors(i,:), ...
        'FontName', 'Times New Roman', ...
        'FontSize', 10, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'middle');
end

set(ax, 'YDir', 'reverse');
view(ax, -35, 30);

set(ax, 'YTick', y_indices, 'YTickLabel', []);

target_ticks = datenum([datetime(2015,3,16), datetime(2015,3,18), ...
                        datetime(2015,3,20), datetime(2015,3,22)]);
set(ax, 'XLim', [datenum(start_date)-0.5, datenum(end_date)], 'XTick', target_ticks);
datetick(ax, 'x', 'mm-dd', 'keeplimits', 'keepticks');

ylim(ax, [0.5, y_indices(end) + 1.2]);
z_min = min(Z_matrix(:)); z_max = max(Z_matrix(:));
zlim(ax, [z_min - 5, z_max + 5]);

set(ax, 'FontName', 'Times New Roman', 'FontSize', 11, ...
         'ZGrid', 'on', 'YGrid', 'off', 'XGrid', 'on');

xlabel(ax, 'Date (2015)', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
zlabel(ax, 'Amplitude (TECU)', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');

text(ax, 0, 1.05, 'Amplitude (TECU)', 'Units', 'normalized', ...
    'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'bold');
end
