function [mpc, xgd_table, wind_UC] = GetFiles()

% Generate necessary files for running MOST

%% Define constants
define_constants;

%% Read bus and zone info
clc;

% Read bus info
load_buses = readmatrix('zones/load_bus.csv');
gen_buses = readmatrix('zones/gen_bus.csv');
wind_buses = readmatrix('zones/wind_bus.csv');

zone1_thermals = readmatrix('zones/zone1_thermal_buses.csv');
zone2_thermals = readmatrix('zones/zone2_thermal_buses.csv');
zone3_thermals = readmatrix('zones/zone3_thermal_buses.csv');

% Set numbers
num_zones = 3;
num_gens = size(gen_buses, 1);
num_loads = size(load_buses, 1);

% Get gen params
Pmax = readmatrix('./gen_params/Pmax.csv');
Pmin = readmatrix('./gen_params/Pmin.csv');
ramp_30 = readmatrix('./gen_params/ramp_30.csv');
startup_cost = readmatrix('./gen_params/startup_cost.csv');
shutdown_cost = readmatrix('./gen_params/shutdown_cost.csv');
gencost_params = readmatrix('./gen_params/gencost_params.csv');
reserve_req = readmatrix('./gen_params/reserve_req.csv');
reserve_cost = readmatrix('./gen_params/reserve_cost.csv');
reserve_qty = readmatrix('./gen_params/reserve_qty.csv');

% Get branch params
%%%%%%%%%%%%%% These numbers are determined separately %%%%%%%%%%%%%%%%%
PF_max_catgory1 = logical(readmatrix('./branch_params/PF_max_category1.csv'));
PF_max_catgory2 = logical(readmatrix('./branch_params/PF_max_category2.csv'));
PF_max_catgory3 = logical(readmatrix('./branch_params/PF_max_category3.csv'));
PF_max1 = 1000;
PF_max2 = 500;
PF_max3 = 300;


%% Set power grid case file
clc;

% Read casefile
casefile = PowerGrid(num_zones);     
mpc = loadcase(casefile);

% Set mpc.bus 
mpc.bus(load_buses, [PD QD]) = 0;  

% Set mpc.gen
mpc.gen(1:num_gens, [PG QG QMAX QMIN ...
                     PC1 PC2 ...
                     QC1MIN QC1MAX ...
                     QC2MIN QC2MAX ...
                     RAMP_Q APF]) = 0;       % Set unnecessary values as 0
mpc.gen(1:num_gens, PMAX) = Pmax;            %%%%%%%%% Max gen capacity    
mpc.gen(1:num_gens, PMIN) = Pmin;            %%%%%%%%% Min gen capacity (cannot be zero)
mpc.gen(1:num_gens, RAMP_10) = 0; 
mpc.gen(1:num_gens, RAMP_AGC) = 0; 
mpc.gen(1:num_gens, RAMP_30) = ramp_30;      %%%%%%%%% Ramp up/dowm limit within 30 mins
% Dispatchable loads
mpc.gen(num_gens+1:end, [PG QG QMAX QMIN ...
                         PC1 PC2 QC1MIN ...
                         QC1MAX QC2MIN ...
                         QC2MAX RAMP_Q APF]) = 0;
mpc.gen(num_gens+1:end, VG) = 1;           % Set Vg
mpc.gen(num_gens+1:end, 7) = 100;          % Set mBase = 100
mpc.gen(num_gens+1:end, 8) = 1;            % Set status = 1
mpc.gen(num_gens+1:end, PMAX) = 0;         %%%%%%%%% Must be zero
mpc.gen(num_gens+1:end, PMIN) = -1000;        %%%%%%%%% Must be < 0
mpc.gen(num_gens+1:end, RAMP_AGC) = 0;
mpc.gen(num_gens+1:end, RAMP_10) = 0;
mpc.gen(num_gens+1:end, RAMP_30) = 1e6;    % Set unlimited ramp up/down limit


% Set mpc.branch
%%%%%%%%%%%%%%% Set line flow limit %%%%%%%%%%%%%%%%%%%%
mpc.branch(PF_max_catgory1, RATE_A) = PF_max1;       
mpc.branch(PF_max_catgory2, RATE_A) = PF_max2;    
mpc.branch(PF_max_catgory3, RATE_A) = PF_max3;    


% Set mpc.gencost
% Gens
mpc.gencost(1:num_gens, 1) = 2;                     % Set cost model
mpc.gencost(1:num_gens, 2) = startup_cost;          %%%%%%%%%%%%%%%%%%%%%% StartUp cost, cannot be too big    
mpc.gencost(1:num_gens, 3) = shutdown_cost;         %%%%%%%%%%%%%%%%%%%%%% ShutDown cost, cannot be too big   
mpc.gencost(1:num_gens, 4) = 2;                     % Set num of params
mpc.gencost(1:num_gens, 5) = gencost_params;        %%%%%%%%%%%%%%%%%%%%%% Set cost coeff
mpc.gencost(1:num_gens, 6) = 0;
% Dispatchable loads
mpc.gencost(num_gens+1:end, 1) = 2;
mpc.gencost(num_gens+1:end, [2 3]) = 0;     % Set StartUp and StutDown cost = 0
mpc.gencost(num_gens+1:end, 4) = 2;         % Set number of parameters
mpc.gencost(num_gens+1:end, 5) = 1e6;       %%%%%%%%%%%%%%%%%%%%%% Set as +Inf
mpc.gencost(num_gens+1:end, 6) = 0;


% Set mpc.reserves (wind gens have no reserve)
temp1 = zeros(1, num_gens);
bidx = ismember(gen_buses, zone1_thermals);
temp1(bidx) = 1;

temp2 = zeros(1, num_gens);
bidx = ismember(gen_buses, zone2_thermals);
temp2(bidx) = 1;

temp3 = zeros(1, num_gens);
bidx = ismember(gen_buses, zone3_thermals);
temp3(bidx) = 1;

temp4 = zeros(1, num_loads);    % Dispatchable loads

mpc.reserves.zones(1, :) = cat(2, temp1, temp4);
mpc.reserves.zones(2, :) = cat(2, temp2, temp4);
mpc.reserves.zones(3, :) = cat(2, temp3, temp4);

% reserve requirements for each zone in MW
mpc.reserves.req = reserve_req;             % Should be consistent with "num_zones"

% reserve costs in $/MW for each gen that belongs to at least 1 zone
% (same order as gens, but skipping any gen that does not belong to any zone)
mpc.reserves.cost = reserve_cost;    % Skip winds and loads

% OPTIONAL max reserve quantities for each gen that belongs to at least 1 zone
% (same order as gens, but skipping any gen that does not belong to any zone)
mpc.reserves.qty = reserve_qty;      % Skip winds and loads


%% Set XGD file
clc;

% Read xgd table
xgd_table = XGDTable;

% Gens
xgd_table.data(1:num_gens, 1) = 1;          % Set thermal gens UC = commitable
xgd_table.data(1:num_gens, 2) = 1;      
xgd_table.data(1:num_gens, [3 4]) = 2;      %%%%%%%%%%%%%%%%%%%%%% Set thermal gens MinUp and MinDown
xgd_table.data(1:num_gens, 5) = 0;          % PositiveActiveReservePrice
xgd_table.data(1:num_gens, 6) = 0;          % PositiveActiveReserveQuantity
xgd_table.data(1:num_gens, 7) = 0;          % NegativeActiveReservePrice
xgd_table.data(1:num_gens, 8) = 0;          % NegativeActiveReserveQuantity
xgd_table.data(1:num_gens, 9) = 0;          % PositiveActiveDeltaPrice
xgd_table.data(1:num_gens, 10) = 0;         % NegativeActiveDeltaPrice
xgd_table.data(1:num_gens, 11) = 0;         % PositiveLoadFollowReservePrice
xgd_table.data(1:num_gens, 12) = 1e6;       % PositiveLoadFollowReserveQuantity (already specified by RAMP_30)
xgd_table.data(1:num_gens, 13) = 0;         % NegativeLoadFollowReservePrice
xgd_table.data(1:num_gens, 14) = 1e6;       % NegativeLoadFollowReserveQuantity (already specified by RAMP_30)

%%%%%%%%%%%% Set UC = off for thermals at winds %%%%%%%%%%%%%%%%%%
temp1 = ismember(gen_buses, wind_buses);    
temp2 = zeros(num_loads, 1);
bidx = logical(cat(1, temp1, temp2));
xgd_table.data(bidx, 1) = -1;

% Dispatchable loads
xgd_table.data(num_gens+1:end, 1) = 2;      % Set dispatchable loads UC = 2
xgd_table.data(num_gens+1:end, 2) = 1;
xgd_table.data(num_gens+1:end, [3 4]) = 1;  % Set dispatchable loads MinUp/MinDown
xgd_table.data(num_gens+1:end, 5) = 0;      % PositiveActiveReservePrice
xgd_table.data(num_gens+1:end, 6) = 0;      % PositiveActiveReserveQuantity
xgd_table.data(num_gens+1:end, 7) = 0;      % NegativeActiveReservePrice
xgd_table.data(num_gens+1:end, 8) = 0;      % NegativeActiveReserveQuantity
xgd_table.data(num_gens+1:end, 9) = 0;      % PositiveActiveDeltaPrice
xgd_table.data(num_gens+1:end, 10) = 0;     % NegativeActiveDeltaPrice
xgd_table.data(num_gens+1:end, 11) = 0;     % PositiveLoadFollowReservePrice
xgd_table.data(num_gens+1:end, 12) = 1e6;   % PositiveLoadFollowReserveQuantity
xgd_table.data(num_gens+1:end, 13) = 0;     % NegativeLoadFollowReservePrice (set as unlimited)
xgd_table.data(num_gens+1:end, 14) = 1e6;   % NegativeLoadFollowReserveQuantity (set as unlimited)


%% Set WindUC file
clc;

% Read wind UC file
wind_UC = WindUC;

% Gens
wind_UC.gen(:, PMAX) = 100;         %%%%%%%%%%%%%%%%%%% This value will be replaced by stochastic winds
wind_UC.gen(:, PMIN) = 0; 
wind_UC.gen(:, RAMP_AGC) = 0;       
wind_UC.gen(:, RAMP_10) = 0;
wind_UC.gen(:, RAMP_30) = 1e6;      % Set unlimited ramp rate
% xgd_table
wind_UC.xgd_table.data(:, 1) = 2;   %%%%%%%%%%%%%%%%%%% Set UC = on all the time
wind_UC.xgd_table.data(:, 2) = 1;
wind_UC.xgd_table.data(:, 3) = 0;   
wind_UC.xgd_table.data(:, 4) = 0;   % Set cost = 0 to make sure wind is always preferably used
wind_UC.xgd_table.data(:, 5:end) = 0;  % Set all = 0
wind_UC.xgd_table.data(:, 12) = 1e6;   % Set unlimited reserve and load follow capacity
wind_UC.xgd_table.data(:, 14) = 1e6;   % Set unlimited reserve and load follow capacity

end