function SetGenParams()

% Set params for gen bus params

gen_buses = readmatrix('../zones/gen_bus.csv');
thermal_buses = readmatrix('../zones/thermal_bus.csv');
zone1_thermals = readmatrix('../zones/zone1_thermal_buses.csv');
zone2_thermals = readmatrix('../zones/zone2_thermal_buses.csv');
zone3_thermals = readmatrix('../zones/zone3_thermal_buses.csv');

num_gens = length(gen_buses);
num_thermals = length(thermal_buses);

% Max gen capacity
rng(10);                              
Pmax = 250*ones([num_gens, 1]);
file_path = './Pmax.csv';
writematrix(Pmax, file_path, WriteMode="overwrite");

% Min gen capacity
Pmin = 50*ones([num_gens, 1]);
file_path = './Pmin.csv';
writematrix(Pmin, file_path, WriteMode="overwrite");

% Ramp up/dowm limit within 30 mins
rng(12);                      
ramp_30 = 200*ones([num_gens, 1]);
file_path = './ramp_30.csv';
writematrix(ramp_30, file_path, WriteMode="overwrite");

% StartUp cost, cannot be too big
rng(15);                               
startup_cost = 0*ones([num_gens, 1]);
% startup_cost = zeros(num_gens, 1);
file_path = './startup_cost.csv';
writematrix(startup_cost, file_path, WriteMode="overwrite");

% ShutDown cost, cannot be too big
rng(20);                            
shutdown_cost = 0*ones([num_gens, 1]);
% shutdown_cost = zeros(num_gens, 1);
file_path = './shutdown_cost.csv';
writematrix(shutdown_cost, file_path, WriteMode="overwrite");

% Set cost coeff
rng(25);                                 
gencost_params = 1*ones([num_gens, 1]);
file_path = './gencost_params.csv';
writematrix(gencost_params, file_path, WriteMode="overwrite");

% Reserve req
bidx1 = ismember(gen_buses, zone1_thermals);
bidx2 = ismember(gen_buses, zone2_thermals);
bidx3 = ismember(gen_buses, zone3_thermals);
reserve_req = [max(Pmax(bidx1)); max(Pmax(bidx2)); max(Pmax(bidx3))];
file_path = './reserve_req.csv';
writematrix(reserve_req, file_path, WriteMode="overwrite");
% Reserve cost
rng(30);
reserve_cost = 1.*ones([num_thermals, 1]); 
file_path = './reserve_cost.csv';
writematrix(reserve_cost, file_path, WriteMode="overwrite");
% Reserve qty
rng(35);
reserve_qty = 50*ones([num_thermals, 1]); 
file_path = './reserve_qty.csv';
writematrix(reserve_qty, file_path, WriteMode="overwrite");

end
