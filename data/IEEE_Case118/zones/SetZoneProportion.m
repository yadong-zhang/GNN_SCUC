function SetZoneProportion()

% Set win and load proportion

rng('default');

% Read zone info
zone1_load_bus = readmatrix('zone1_load_bus.csv');
zone2_load_bus = readmatrix('zone2_load_bus.csv');
zone3_load_bus = readmatrix('zone3_load_bus.csv');

zone1_wind_bus = readmatrix('zone1_wind_bus.csv');
zone2_wind_bus = readmatrix('zone2_wind_bus.csv');
zone3_wind_bus = readmatrix('zone3_wind_bus.csv');


% Generate proportion
zone1_load_proportion = unifrnd(0.8, 1.2, [size(zone1_load_bus, 1), 1]);
zone2_load_proportion = unifrnd(0.8, 1.2, [size(zone2_load_bus, 1), 1]);
zone3_load_proportion = unifrnd(0.8, 1.2, [size(zone3_load_bus, 1), 1]);

zone1_wind_proportion = unifrnd(1, 2, [size(zone1_wind_bus, 1), 1]);
zone2_wind_proportion = unifrnd(1, 2, [size(zone2_wind_bus, 1), 1]);
zone3_wind_proportion = unifrnd(1, 2, [size(zone3_wind_bus, 1), 1]);


% Save files

writematrix(zone1_load_proportion, 'zone1_load_proportion.csv', 'WriteMode', 'overwrite');
writematrix(zone2_load_proportion, 'zone2_load_proportion.csv', 'WriteMode', 'overwrite');
writematrix(zone3_load_proportion, 'zone3_load_proportion.csv', 'WriteMode', 'overwrite');

writematrix(zone1_wind_proportion, 'zone1_wind_proportion.csv', 'WriteMode', 'overwrite');
writematrix(zone2_wind_proportion, 'zone2_wind_proportion.csv', 'WriteMode', 'overwrite');
writematrix(zone3_wind_proportion, 'zone3_wind_proportion.csv', 'WriteMode', 'overwrite');


end