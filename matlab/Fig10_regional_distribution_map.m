function fig = Fig10_regional_distribution_map()
%FIG10_REGIONAL_DISTRIBUTION_MAP Plot the five study regions.

fig = figure('Position', [100, 100, 1200, 600], 'Color', 'w');

ax = axesm('robinson', 'Frame', 'on', 'Grid', 'on', ...
    'MapLatLimit', [-80 80], 'MapLonLimit', [-180 180], ...
    'MeridianLabel', 'on', 'ParallelLabel', 'on', ...
    'MLabelParallel', 'south', 'PLabelMeridian', 'west', ...
    'GColor', [0.88 0.88 0.88], 'GLineWidth', 0.5, ...
    'FontName', 'Times New Roman', 'FontSize', 12);

land = shaperead('landareas', 'UseGeocoords', true);
geoshow(ax, land, 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', [0.7 0.7 0.7], 'LineWidth', 0.5);

regions = {
    'EU (Europe)',           35,  60, -10,  40, [0.60, 0.15, 0.15], [ 0,  0];
    'NA (North America)',    25,  55,-130, -60, [0.60, 0.15, 0.15], [ 0,  0];
    'EA (East Asia)',        15,  55,  70, 140, [0.60, 0.15, 0.15], [ 0,  0];
    'LL (Low Latitude)',    -20,  20,  15,  50, [0.60, 0.15, 0.15], [ 0,  0];
    'SAM (South America)',  -55,  15, -85, -35, [0.60, 0.15, 0.15], [ 0,  0]
};

for i = 1:size(regions, 1)
    name    = regions{i, 1};
    lat_min = regions{i, 2}; lat_max = regions{i, 3};
    lon_min = regions{i, 4}; lon_max = regions{i, 5};
    color   = regions{i, 6};
    offset  = regions{i, 7};

    nPts = 100;
    lat_side1 = linspace(lat_min, lat_max, nPts); lon_side1 = repmat(lon_min, 1, nPts);
    lat_side2 = repmat(lat_max, 1, nPts);         lon_side2 = linspace(lon_min, lon_max, nPts);
    lat_side3 = linspace(lat_max, lat_min, nPts); lon_side3 = repmat(lon_max, 1, nPts);
    lat_side4 = repmat(lat_min, 1, nPts);         lon_side4 = linspace(lon_max, lon_min, nPts);

    lat_box = [lat_side1, lat_side2, lat_side3, lat_side4];
    lon_box = [lon_side1, lon_side2, lon_side3, lon_side4];

    patchm(lat_box, lon_box, 0, 'FaceColor', color, 'FaceAlpha', 0.25, ...
        'EdgeColor', color, 'LineWidth', 2.5);

    center_lat = mean([lat_min, lat_max]) + offset(1);
    center_lon = mean([lon_min, lon_max]) + offset(2);

    textm(center_lat, center_lon, name, ...
        'HorizontalAlignment', 'center', ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 12, ...
        'Color', color, ...
        'BackgroundColor', [1 1 1], ...
        'EdgeColor', color, ...
        'LineWidth', 1.2, ...
        'Margin', 3);
end

title('Geographic Boundaries of the Five Regional Electron Content (REC) Sectors', ...
    'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2]);
tightmap;
end
