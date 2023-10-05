function LatinHypercubeSampling(num_zones, num_samples, nt)
% Generate stochastic load and wind samples using LHS

% Set random seed
rng(25);

% Get num of variables
num_vars = num_zones * 2;

% Create initial state
mu = 0;
sigma = 1;

init_state = zeros(num_vars, num_samples);

for i = 1:num_vars
    init_state(i, :) = lhsnorm(mu, sigma, num_samples, 'on');
end


% % Generate stochastic samples (aggregated)
[agg_loads, agg_winds] = GenerateSamples(init_state, nt);


% % Load samples
% Read load bus info in each zone
load_bus = readmatrix('../zones/load_bus.csv');
zone1_load_bus = readmatrix('../zones/zone1_load_bus.csv');
zone2_load_bus = readmatrix('../zones/zone2_load_bus.csv');
zone3_load_bus = readmatrix('../zones/zone3_load_bus.csv');
zone4_load_bus = readmatrix('../zones/zone4_load_bus.csv');
zone5_load_bus = readmatrix('../zones/zone5_load_bus.csv');
zone6_load_bus = readmatrix('../zones/zone6_load_bus.csv');
zone7_load_bus = readmatrix('../zones/zone7_load_bus.csv');
zone8_load_bus = readmatrix('../zones/zone8_load_bus.csv');

num_loads = size(load_bus, 1);

% Read load proportion
zone1_load_prop = readmatrix('../zones/zone1_load_proportion.csv');
zone2_load_prop = readmatrix('../zones/zone2_load_proportion.csv');
zone3_load_prop = readmatrix('../zones/zone3_load_proportion.csv');
zone4_load_prop = readmatrix('../zones/zone4_load_proportion.csv');
zone5_load_prop = readmatrix('../zones/zone5_load_proportion.csv');
zone6_load_prop = readmatrix('../zones/zone6_load_proportion.csv');
zone7_load_prop = readmatrix('../zones/zone7_load_proportion.csv');
zone8_load_prop = readmatrix('../zones/zone8_load_proportion.csv');

% Calculate load for each zone
zone1_loads = tensorprod(zone1_load_prop, agg_loads(1, :, :), 2, 1);
zone2_loads = tensorprod(zone2_load_prop, agg_loads(2, :, :), 2, 1);
zone3_loads = tensorprod(zone3_load_prop, agg_loads(3, :, :), 2, 1);
zone4_loads = tensorprod(zone4_load_prop, agg_loads(4, :, :), 2, 1);
zone5_loads = tensorprod(zone5_load_prop, agg_loads(5, :, :), 2, 1);
zone6_loads = tensorprod(zone6_load_prop, agg_loads(6, :, :), 2, 1);
zone7_loads = tensorprod(zone7_load_prop, agg_loads(7, :, :), 2, 1);
zone8_loads = tensorprod(zone8_load_prop, agg_loads(8, :, :), 2, 1);


% Get combined load samples
zone1_load_bidx = ismember(load_bus, zone1_load_bus);   % Get boolean index of  zone 1 load bus out of all load buses
zone2_load_bidx = ismember(load_bus, zone2_load_bus);
zone3_load_bidx = ismember(load_bus, zone3_load_bus);
zone4_load_bidx = ismember(load_bus, zone4_load_bus);
zone5_load_bidx = ismember(load_bus, zone5_load_bus);
zone6_load_bidx = ismember(load_bus, zone6_load_bus);
zone7_load_bidx = ismember(load_bus, zone7_load_bus);
zone8_load_bidx = ismember(load_bus, zone8_load_bus);

loads = zeros(num_loads, num_samples, nt);
loads(zone1_load_bidx, :, :) = zone1_loads;
loads(zone2_load_bidx, :, :) = zone2_loads;
loads(zone3_load_bidx, :, :) = zone3_loads;
loads(zone4_load_bidx, :, :) = zone4_loads;
loads(zone5_load_bidx, :, :) = zone5_loads;
loads(zone6_load_bidx, :, :) = zone6_loads;
loads(zone7_load_bidx, :, :) = zone7_loads;
loads(zone8_load_bidx, :, :) = zone8_loads;

% Save the combined load data
for i = 1:nt
    sheetName = ['t=' num2str(i)];
    writematrix(zone1_loads(:, :, i), './zone1_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone2_loads(:, :, i), './zone2_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone3_loads(:, :, i), './zone3_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone4_loads(:, :, i), './zone4_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone5_loads(:, :, i), './zone5_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone6_loads(:, :, i), './zone6_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone7_loads(:, :, i), './zone7_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone8_loads(:, :, i), './zone8_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(loads(:, :, i), './load_samples.xlsx', 'Sheet', sheetName);
end


% % Wind samples
% Get wind buses at different zones
wind_bus = readmatrix('../zones/wind_bus.csv');
zone1_wind_bus = readmatrix('../zones/zone1_wind_bus.csv');
zone2_wind_bus = readmatrix('../zones/zone2_wind_bus.csv');
zone3_wind_bus = readmatrix('../zones/zone3_wind_bus.csv');
zone4_wind_bus = readmatrix('../zones/zone4_wind_bus.csv');
zone5_wind_bus = readmatrix('../zones/zone5_wind_bus.csv');
zone6_wind_bus = readmatrix('../zones/zone6_wind_bus.csv');
zone7_wind_bus = readmatrix('../zones/zone7_wind_bus.csv');
zone8_wind_bus = readmatrix('../zones/zone8_wind_bus.csv');

num_winds = size(wind_bus, 1);

% Read wind proportion
zone1_wind_prop = readmatrix('../zones/zone1_wind_proportion.csv');
zone2_wind_prop = readmatrix('../zones/zone2_wind_proportion.csv');
zone3_wind_prop = readmatrix('../zones/zone3_wind_proportion.csv');
zone4_wind_prop = readmatrix('../zones/zone4_wind_proportion.csv');
zone5_wind_prop = readmatrix('../zones/zone5_wind_proportion.csv');
zone6_wind_prop = readmatrix('../zones/zone6_wind_proportion.csv');
zone7_wind_prop = readmatrix('../zones/zone7_wind_proportion.csv');
zone8_wind_prop = readmatrix('../zones/zone8_wind_proportion.csv');



% Calculate wind for each zone
zone1_winds(:, :, :) = tensorprod(zone1_wind_prop, agg_winds(1, :, :), 2, 1);
zone2_winds(:, :, :) = tensorprod(zone2_wind_prop, agg_winds(2, :, :), 2, 1);
zone3_winds(:, :, :) = tensorprod(zone3_wind_prop, agg_winds(3, :, :), 2, 1);
zone4_winds(:, :, :) = tensorprod(zone4_wind_prop, agg_winds(4, :, :), 2, 1);
zone5_winds(:, :, :) = tensorprod(zone5_wind_prop, agg_winds(5, :, :), 2, 1);
zone6_winds(:, :, :) = tensorprod(zone6_wind_prop, agg_winds(6, :, :), 2, 1);
zone7_winds(:, :, :) = tensorprod(zone7_wind_prop, agg_winds(7, :, :), 2, 1);
zone8_winds(:, :, :) = tensorprod(zone8_wind_prop, agg_winds(8, :, :), 2, 1);

% Get combined wind samples
zone1_wind_bidx = ismember(wind_bus, zone1_wind_bus);   % Get boolean index of  zone 1 wind bus out of all wind buses
zone2_wind_bidx = ismember(wind_bus, zone2_wind_bus);
zone3_wind_bidx = ismember(wind_bus, zone3_wind_bus);
zone4_wind_bidx = ismember(wind_bus, zone4_wind_bus);
zone5_wind_bidx = ismember(wind_bus, zone5_wind_bus);
zone6_wind_bidx = ismember(wind_bus, zone6_wind_bus);
zone7_wind_bidx = ismember(wind_bus, zone7_wind_bus);
zone8_wind_bidx = ismember(wind_bus, zone8_wind_bus);

winds = zeros(num_winds, num_samples, nt);
winds(zone1_wind_bidx, :, :) = zone1_winds;
winds(zone2_wind_bidx, :, :) = zone2_winds;
winds(zone3_wind_bidx, :, :) = zone3_winds;
winds(zone4_wind_bidx, :, :) = zone4_winds;
winds(zone5_wind_bidx, :, :) = zone5_winds;
winds(zone6_wind_bidx, :, :) = zone6_winds;
winds(zone7_wind_bidx, :, :) = zone7_winds;
winds(zone8_wind_bidx, :, :) = zone8_winds;


% Save combined data
for i = 1:nt
    sheetName = ['t=' num2str(i)];
    writematrix(zone1_winds(:, :, i), './zone1_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone2_winds(:, :, i), './zone2_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone3_winds(:, :, i), './zone3_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone4_winds(:, :, i), './zone4_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone5_winds(:, :, i), './zone5_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone6_winds(:, :, i), './zone6_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone7_winds(:, :, i), './zone7_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone8_winds(:, :, i), './zone8_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(winds(:, :, i), './wind_samples.xlsx', 'Sheet', sheetName);
end



end
