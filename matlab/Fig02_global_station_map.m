function fig = Fig02_global_station_map()
%FIG02_GLOBAL_STATION_MAP Plot the locations used to select grid points.

stations = {'ADIS', 'GUAM', 'KOUR', 'Wuhan', 'SANT', 'POTS', 'CAS1', 'TRO1', 'NYAL'};
lat = [9.03, 13.59, 5.25, 30.53, -33.45, 52.38, -66.28, 69.66, 78.93];
lon = [38.76, 144.87, -52.81, 114.36, -70.67, 13.06, 110.52, 18.94, 11.93];

fig = figure('Color', 'w');
set(fig, 'Units', 'centimeters');
set(fig, 'Position', [5, 5, 15, 8.5]);
set(fig, 'PaperPositionMode', 'auto');

axesm('robinson', 'Frame', 'on', 'Grid', 'on', ...
      'MapLatLimit', [-90 90], 'MapLonLimit', [-180 180], ...
      'MeridianLabel', 'on', 'ParallelLabel', 'on', ...
      'MLabelParallel', 'south', 'PLabelMeridian', 'west', ...
      'MLabelLocation', 60, 'PLabelLocation', 30, ...
      'FontName', 'Times New Roman', 'FontSize', 10.5);

land = shaperead('landareas', 'UseGeoCoords', true);
geoshow(land, 'FaceColor', [0.88 0.88 0.88], 'EdgeColor', 'none');
coast = load('coastlines.mat', 'coastlat', 'coastlon');
geoshow(coast.coastlat, coast.coastlon, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);

plotm(lat, lon, '^', 'MarkerSize', 7, ...
      'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'LineWidth', 0.8);

lat_offset = [ 3,   4,   4,   4,  -4,   4,  -4,   3,   3];
lon_offset = [ 5,   0,   5,   0,   6,   5,   8,  12,  15];

for i = 1:length(stations)
    textm(lat(i) + lat_offset(i), lon(i) + lon_offset(i), stations{i}, ...
          'FontName', 'Times New Roman', 'FontSize', 10.5, 'FontWeight', 'bold', ...
          'HorizontalAlignment', 'center');
end

annotation('arrow', [0.85 0.85], [0.75 0.88], 'LineWidth', 1.5, 'HeadStyle', 'vback2');
annotation('textbox', [0.835 0.89 0.05 0.05], 'String', 'N', ...
           'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold', ...
           'LineStyle', 'none');
end
