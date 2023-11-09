function LatinHypercubeSampling(num_zones, num_samples, nt)
% Generate stochastic load and wind samples using LHS

% Set random seed
rng('default');

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

% % Save aggregated samples
for i = 1:nt
    sheetName = ['t=' num2str(i)];
    writematrix(agg_loads(:, :, i), './aggregated_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(agg_winds(:, :, i), './aggregated_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
end


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
zone9_load_bus = readmatrix('../zones/zone9_load_bus.csv');
zone10_load_bus = readmatrix('../zones/zone10_load_bus.csv');
zone11_load_bus = readmatrix('../zones/zone11_load_bus.csv');
zone12_load_bus = readmatrix('../zones/zone12_load_bus.csv');
zone13_load_bus = readmatrix('../zones/zone13_load_bus.csv');
zone14_load_bus = readmatrix('../zones/zone14_load_bus.csv');
zone15_load_bus = readmatrix('../zones/zone15_load_bus.csv');
zone16_load_bus = readmatrix('../zones/zone16_load_bus.csv');

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
zone9_load_prop = readmatrix('../zones/zone9_load_proportion.csv');
zone10_load_prop = readmatrix('../zones/zone10_load_proportion.csv');
zone11_load_prop = readmatrix('../zones/zone11_load_proportion.csv');
zone12_load_prop = readmatrix('../zones/zone12_load_proportion.csv');
zone13_load_prop = readmatrix('../zones/zone13_load_proportion.csv');
zone14_load_prop = readmatrix('../zones/zone14_load_proportion.csv');
zone15_load_prop = readmatrix('../zones/zone15_load_proportion.csv');
zone16_load_prop = readmatrix('../zones/zone16_load_proportion.csv');

% Calculate load for each zone
zone1_loads = tensorprod(zone1_load_prop, agg_loads(1, :, :), 2, 1);
zone2_loads = tensorprod(zone2_load_prop, agg_loads(2, :, :), 2, 1);
zone3_loads = tensorprod(zone3_load_prop, agg_loads(3, :, :), 2, 1);
zone4_loads = tensorprod(zone4_load_prop, agg_loads(4, :, :), 2, 1);
zone5_loads = tensorprod(zone5_load_prop, agg_loads(5, :, :), 2, 1);
zone6_loads = tensorprod(zone6_load_prop, agg_loads(6, :, :), 2, 1);
zone7_loads = tensorprod(zone7_load_prop, agg_loads(7, :, :), 2, 1);
zone8_loads = tensorprod(zone8_load_prop, agg_loads(8, :, :), 2, 1);
zone9_loads = tensorprod(zone9_load_prop, agg_loads(9, :, :), 2, 1);
zone10_loads = tensorprod(zone10_load_prop, agg_loads(10, :, :), 2, 1);
zone11_loads = tensorprod(zone11_load_prop, agg_loads(11, :, :), 2, 1);
zone12_loads = tensorprod(zone12_load_prop, agg_loads(12, :, :), 2, 1);
zone13_loads = tensorprod(zone13_load_prop, agg_loads(13, :, :), 2, 1);
zone14_loads = tensorprod(zone14_load_prop, agg_loads(14, :, :), 2, 1);
zone15_loads = tensorprod(zone15_load_prop, agg_loads(15, :, :), 2, 1);
zone16_loads = tensorprod(zone16_load_prop, agg_loads(16, :, :), 2, 1);

% Get combined load samples
zone1_load_bidx = ismember(load_bus, zone1_load_bus);   % Get Boolean index of  zone 1 load bus out of all load buses
zone2_load_bidx = ismember(load_bus, zone2_load_bus);
zone3_load_bidx = ismember(load_bus, zone3_load_bus);
zone4_load_bidx = ismember(load_bus, zone4_load_bus);
zone5_load_bidx = ismember(load_bus, zone5_load_bus);
zone6_load_bidx = ismember(load_bus, zone6_load_bus);
zone7_load_bidx = ismember(load_bus, zone7_load_bus);
zone8_load_bidx = ismember(load_bus, zone8_load_bus);
zone9_load_bidx = ismember(load_bus, zone9_load_bus); 
zone10_load_bidx = ismember(load_bus, zone10_load_bus);
zone11_load_bidx = ismember(load_bus, zone11_load_bus);
zone12_load_bidx = ismember(load_bus, zone12_load_bus);
zone13_load_bidx = ismember(load_bus, zone13_load_bus);
zone14_load_bidx = ismember(load_bus, zone14_load_bus);
zone15_load_bidx = ismember(load_bus, zone15_load_bus);
zone16_load_bidx = ismember(load_bus, zone16_load_bus);

loads = zeros(num_loads, num_samples, nt);
loads(zone1_load_bidx, :, :) = zone1_loads;
loads(zone2_load_bidx, :, :) = zone2_loads;
loads(zone3_load_bidx, :, :) = zone3_loads;
loads(zone4_load_bidx, :, :) = zone4_loads;
loads(zone5_load_bidx, :, :) = zone5_loads;
loads(zone6_load_bidx, :, :) = zone6_loads;
loads(zone7_load_bidx, :, :) = zone7_loads;
loads(zone8_load_bidx, :, :) = zone8_loads;
loads(zone9_load_bidx, :, :) = zone9_loads;
loads(zone10_load_bidx, :, :) = zone10_loads;
loads(zone11_load_bidx, :, :) = zone11_loads;
loads(zone12_load_bidx, :, :) = zone12_loads;
loads(zone13_load_bidx, :, :) = zone13_loads;
loads(zone14_load_bidx, :, :) = zone14_loads;
loads(zone15_load_bidx, :, :) = zone15_loads;
loads(zone16_load_bidx, :, :) = zone16_loads;


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
    writematrix(zone9_loads(:, :, i), './zone9_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone10_loads(:, :, i), './zone10_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone11_loads(:, :, i), './zone11_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone12_loads(:, :, i), './zone12_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone13_loads(:, :, i), './zone13_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone14_loads(:, :, i), './zone14_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone15_loads(:, :, i), './zone15_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone16_loads(:, :, i), './zone16_load_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
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
zone9_wind_bus = readmatrix('../zones/zone9_wind_bus.csv');
zone10_wind_bus = readmatrix('../zones/zone10_wind_bus.csv');
zone11_wind_bus = readmatrix('../zones/zone11_wind_bus.csv');
zone12_wind_bus = readmatrix('../zones/zone12_wind_bus.csv');
zone13_wind_bus = readmatrix('../zones/zone13_wind_bus.csv');
zone14_wind_bus = readmatrix('../zones/zone14_wind_bus.csv');
zone15_wind_bus = readmatrix('../zones/zone15_wind_bus.csv');
zone16_wind_bus = readmatrix('../zones/zone16_wind_bus.csv');

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
zone9_wind_prop = readmatrix('../zones/zone9_wind_proportion.csv');
zone10_wind_prop = readmatrix('../zones/zone10_wind_proportion.csv');
zone11_wind_prop = readmatrix('../zones/zone11_wind_proportion.csv');
zone12_wind_prop = readmatrix('../zones/zone12_wind_proportion.csv');
zone13_wind_prop = readmatrix('../zones/zone13_wind_proportion.csv');
zone14_wind_prop = readmatrix('../zones/zone14_wind_proportion.csv');
zone15_wind_prop = readmatrix('../zones/zone15_wind_proportion.csv');
zone16_wind_prop = readmatrix('../zones/zone16_wind_proportion.csv');

% Calculate wind for each zone
zone1_winds = tensorprod(zone1_wind_prop, agg_winds(1, :, :), 2, 1);
zone2_winds = tensorprod(zone2_wind_prop, agg_winds(2, :, :), 2, 1);
zone3_winds = tensorprod(zone3_wind_prop, agg_winds(3, :, :), 2, 1);
zone4_winds = tensorprod(zone4_wind_prop, agg_winds(4, :, :), 2, 1);
zone5_winds = tensorprod(zone5_wind_prop, agg_winds(5, :, :), 2, 1);
zone6_winds = tensorprod(zone6_wind_prop, agg_winds(6, :, :), 2, 1);
zone7_winds = tensorprod(zone7_wind_prop, agg_winds(7, :, :), 2, 1);
zone8_winds = tensorprod(zone8_wind_prop, agg_winds(8, :, :), 2, 1);
zone9_winds = tensorprod(zone9_wind_prop, agg_winds(9, :, :), 2, 1);
zone10_winds = tensorprod(zone10_wind_prop, agg_winds(10, :, :), 2, 1);
zone11_winds = tensorprod(zone11_wind_prop, agg_winds(11, :, :), 2, 1);
zone12_winds = tensorprod(zone12_wind_prop, agg_winds(12, :, :), 2, 1);
zone13_winds = tensorprod(zone13_wind_prop, agg_winds(13, :, :), 2, 1);
zone14_winds = tensorprod(zone14_wind_prop, agg_winds(14, :, :), 2, 1);
zone15_winds = tensorprod(zone15_wind_prop, agg_winds(15, :, :), 2, 1);
zone16_winds = tensorprod(zone16_wind_prop, agg_winds(16, :, :), 2, 1);

% Get combined wind samples
zone1_wind_bidx = ismember(wind_bus, zone1_wind_bus);   % Get Boolean index of  zone 1 wind bus out of all wind buses
zone2_wind_bidx = ismember(wind_bus, zone2_wind_bus);
zone3_wind_bidx = ismember(wind_bus, zone3_wind_bus);
zone4_wind_bidx = ismember(wind_bus, zone4_wind_bus);
zone5_wind_bidx = ismember(wind_bus, zone5_wind_bus);
zone6_wind_bidx = ismember(wind_bus, zone6_wind_bus);
zone7_wind_bidx = ismember(wind_bus, zone7_wind_bus);
zone8_wind_bidx = ismember(wind_bus, zone8_wind_bus);
zone9_wind_bidx = ismember(wind_bus, zone9_wind_bus);
zone10_wind_bidx = ismember(wind_bus, zone10_wind_bus);
zone11_wind_bidx = ismember(wind_bus, zone11_wind_bus);
zone12_wind_bidx = ismember(wind_bus, zone12_wind_bus);
zone13_wind_bidx = ismember(wind_bus, zone13_wind_bus);
zone14_wind_bidx = ismember(wind_bus, zone14_wind_bus);
zone15_wind_bidx = ismember(wind_bus, zone15_wind_bus);
zone16_wind_bidx = ismember(wind_bus, zone16_wind_bus);

winds = zeros(num_winds, num_samples, nt);
winds(zone1_wind_bidx, :, :) = zone1_winds;
winds(zone2_wind_bidx, :, :) = zone2_winds;
winds(zone3_wind_bidx, :, :) = zone3_winds;
winds(zone4_wind_bidx, :, :) = zone4_winds;
winds(zone5_wind_bidx, :, :) = zone5_winds;
winds(zone6_wind_bidx, :, :) = zone6_winds;
winds(zone7_wind_bidx, :, :) = zone7_winds;
winds(zone8_wind_bidx, :, :) = zone8_winds;
winds(zone9_wind_bidx, :, :) = zone9_winds;
winds(zone10_wind_bidx, :, :) = zone10_winds;
winds(zone11_wind_bidx, :, :) = zone11_winds;
winds(zone12_wind_bidx, :, :) = zone12_winds;
winds(zone13_wind_bidx, :, :) = zone13_winds;
winds(zone14_wind_bidx, :, :) = zone14_winds;
winds(zone15_wind_bidx, :, :) = zone15_winds;
winds(zone16_wind_bidx, :, :) = zone16_winds;

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
    writematrix(zone9_winds(:, :, i), './zone9_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone10_winds(:, :, i), './zone10_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone11_winds(:, :, i), './zone11_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone12_winds(:, :, i), './zone12_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone13_winds(:, :, i), './zone13_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone14_winds(:, :, i), './zone14_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone15_winds(:, :, i), './zone15_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(zone16_winds(:, :, i), './zone16_wind_samples.xlsx', 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    writematrix(winds(:, :, i), './wind_samples.xlsx', 'Sheet', sheetName);
end



end
