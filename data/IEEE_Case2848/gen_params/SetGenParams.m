function SetGenParams()

% Set params for gen bus params

gen_buses = readmatrix('../zones/gen_bus.csv');
thermal_buses = readmatrix('../zones/thermal_bus.csv');
zone1_thermals = readmatrix('../zones/zone1_thermal_buses.csv');
zone2_thermals = readmatrix('../zones/zone2_thermal_buses.csv');
zone3_thermals = readmatrix('../zones/zone3_thermal_buses.csv');
zone4_thermals = readmatrix('../zones/zone4_thermal_buses.csv');
zone5_thermals = readmatrix('../zones/zone5_thermal_buses.csv');
zone6_thermals = readmatrix('../zones/zone6_thermal_buses.csv');
zone7_thermals = readmatrix('../zones/zone7_thermal_buses.csv');
zone8_thermals = readmatrix('../zones/zone8_thermal_buses.csv');
zone9_thermals = readmatrix('../zones/zone9_thermal_buses.csv');
zone10_thermals = readmatrix('../zones/zone10_thermal_buses.csv');
zone11_thermals = readmatrix('../zones/zone11_thermal_buses.csv');
zone12_thermals = readmatrix('../zones/zone12_thermal_buses.csv');
zone13_thermals = readmatrix('../zones/zone13_thermal_buses.csv');
zone14_thermals = readmatrix('../zones/zone14_thermal_buses.csv');
zone15_thermals = readmatrix('../zones/zone15_thermal_buses.csv');
zone16_thermals = readmatrix('../zones/zone16_thermal_buses.csv');

num_gens = length(gen_buses);
num_thermals = length(thermal_buses);

% Max gen capacity
rng(10);                              
% Pmax = 500*ones([num_gens, 1]);
% Pmax = unifrnd(200, 10000, [num_gens, 1]);
Pmax = 1000*ones([num_gens, 1]);
file_path = './Pmax.csv';
writematrix(Pmax, file_path, WriteMode="overwrite");

% Min gen capacity
% Pmin = 50*ones([num_gens, 1]);
Pmin = zeros([num_gens, 1]);
file_path = './Pmin.csv';
writematrix(Pmin, file_path, WriteMode="overwrite");

% Ramp up/dowm limit within 30 mins
rng(12);                      
% ramp_30 = 100*ones([num_gens, 1]);
ramp_30 = zeros([num_gens, 1]);
file_path = './ramp_30.csv';
writematrix(ramp_30, file_path, WriteMode="overwrite");

% StartUp cost, cannot be too big
rng(15);                               
% startup_cost = 0*ones([num_gens, 1]);
startup_cost = unifrnd(10, 20, [num_gens, 1]);
file_path = './startup_cost.csv';
writematrix(startup_cost, file_path, WriteMode="overwrite");

% ShutDown cost, cannot be too big
rng(20);                            
% shutdown_cost = 0*ones([num_gens, 1]);
shutdown_cost = unifrnd(10, 20, [num_gens, 1]);
file_path = './shutdown_cost.csv';
writematrix(shutdown_cost, file_path, WriteMode="overwrite");

% Set cost coeff
rng(25);                                 
% gencost_params = 1*ones([num_gens, 1]);
gencost_params = unifrnd(1, 2, [num_gens, 1]);
file_path = './gencost_params.csv';
writematrix(gencost_params, file_path, WriteMode="overwrite");

% Reserve req
bidx1 = ismember(gen_buses, zone1_thermals);
bidx2 = ismember(gen_buses, zone2_thermals);
bidx3 = ismember(gen_buses, zone3_thermals);
bidx4 = ismember(gen_buses, zone4_thermals);
bidx5 = ismember(gen_buses, zone5_thermals);
bidx6 = ismember(gen_buses, zone6_thermals);
bidx7 = ismember(gen_buses, zone7_thermals);
bidx8 = ismember(gen_buses, zone8_thermals);
bidx9 = ismember(gen_buses, zone9_thermals);
bidx10 = ismember(gen_buses, zone10_thermals);
bidx11 = ismember(gen_buses, zone11_thermals);
bidx12 = ismember(gen_buses, zone12_thermals);
bidx13 = ismember(gen_buses, zone13_thermals);
bidx14 = ismember(gen_buses, zone14_thermals);
bidx15 = ismember(gen_buses, zone15_thermals);
bidx16 = ismember(gen_buses, zone16_thermals);
reserve_req = [max(Pmax(bidx1)); max(Pmax(bidx2)); max(Pmax(bidx3)); 
    max(Pmax(bidx4)); max(Pmax(bidx5)); max(Pmax(bidx6)); max(Pmax(bidx7)); 
    max(Pmax(bidx8)); max(Pmax(bidx9)); max(Pmax(bidx10)); max(Pmax(bidx11)); 
    max(Pmax(bidx12)); max(Pmax(bidx13)); max(Pmax(bidx14)); max(Pmax(bidx15)); 
    max(Pmax(bidx16))];
file_path = './reserve_req.csv';
writematrix(reserve_req, file_path, WriteMode="overwrite");
% Reserve cost
rng(30);
reserve_cost = ones([num_thermals, 1]); 
file_path = './reserve_cost.csv';
writematrix(reserve_cost, file_path, WriteMode="overwrite");
% Reserve qty
rng(35);
reserve_qty = 200*ones([num_thermals, 1]); 
file_path = './reserve_qty.csv';
writematrix(reserve_qty, file_path, WriteMode="overwrite");

end
