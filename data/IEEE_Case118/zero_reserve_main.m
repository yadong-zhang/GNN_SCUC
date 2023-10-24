clc;
clear all;
close all;

%% Get necessary files
clc;

define_constants;

% Read bus info
load_buses = readmatrix('zones/load_bus.csv');
gen_buses = readmatrix('zones/gen_bus.csv');
wind_buses = readmatrix('zones/wind_bus.csv');

% Set numbers
num_loads = size(load_buses, 1);
num_gens = size(gen_buses, 1);
num_winds = size(wind_buses, 1);

% Get thermal in gen Boolean index
thermal_gen_bidx = ~ismember(gen_buses, wind_buses);
% Get thermal in (gen + dispatchable loads) Boolean index
thermal_dispatchable_load_bidx = logical(cat(1, thermal_gen_bidx, zeros(num_loads, 1)));
% All gens no including dispatchable loads
all_gens = (1:length(thermal_dispatchable_load_bidx))';
% Thermal gens no
thermal_gen_no = all_gens(thermal_dispatchable_load_bidx);


%% Read wind and load samples that have SCUC solution, i.e., from "inputs"
clc;

num_thermals = size(gen_buses, 1) - size(wind_buses, 1);

nt = 12;

max_num = 1199;     % This number is determined according to max number in "outputs"


% 'max_num' instead of 'num_samples' is used to keep the same sequence for later processing
wind_samples = zeros(nt, num_winds, max_num);   
load_samples = zeros(nt, num_loads, max_num);
UC_samples = zeros(nt, num_thermals, max_num);

% Read wind, load and UC solutions from solved samples
for i = 1:max_num
    % Sample path
    wind_path = ['./inputs/wind/sample_' num2str(i) '.csv'];
    load_path = ['./inputs/load/sample_' num2str(i) '.csv'];
    UC_path = ['./outputs/UC/sample_' num2str(i) '.csv'];
    
    % Skip nonexisting samples
    try
        wind = readmatrix(wind_path);
        load = readmatrix(load_path);
        UC = readmatrix(UC_path);
    catch
        fprintf("Sample %d does not exist, skipped!\n", i);
        continue;
    end

    % Concatenate all samples into one
    wind_samples(:, :, i) = wind';
    load_samples(:, :, i) = load';
    UC_samples(:, :, i) = UC';
end



%%
clc;

% Change UC = 1 to UC = 2 (must ON), and UC = 0 to UC = -1 (must OFF)
UC_samples(UC_samples == 1) = 2;
UC_samples(UC_samples == 0) = -1;



%%
% Prepare files
clc;

% [mpc, xgd_table, wind_UC] = zero_reserve_GetFiles();
[mpc, xgd_table, wind_UC] = GetFiles();


%% Prepare file
clc;
   
% Prepare files
xgd = loadxgendata(xgd_table, mpc);                 % Add mpc into xgd_table
[iwind, mpc, xgd] = addwind(wind_UC, mpc, xgd);     % Add wind gen data to 'mpc' and 'xgd'


%% Set numbers
clc;

% nt = size(wind_samples, 1);     % Number of time steps



%%%%%%%%%%%%%% Be careful about this number %%%%%%%%%%%%%%%%%%%%%
% Number of stochastic SCUC samples
num_samples = 1000;    



%%%%%%%%%%%%%% Be careful about this number %%%%%%%%%%%%%%%%%%%%%
% The number of stochastic winds/loads
% max_num_samples = 2000;    

load_shed_bidx = ones(num_samples, 1);      % Record load shedding bidx, 1 for shedding, 0 for no shedding
% wind_usage_bidx = zeros(num_samples, 1);      % Record wind usage bidx, 1 for 100% usage, 0 for not


% % Run MC simulation
%%%%%%%%%%%%%% Be careful about this number %%%%%%%%%%%%%%%%%%%%%
% i = 0; 

%%%%%%%%%%%%%% Be careful about this number %%%%%%%%%%%%%%%%%%%%%
num = 1;

start_time = tic;   % Starting time
for i = 1:max_num
%     if num > num_samples
%         break;
%     end
% 
%     i = i + 1;
%     if i > max_num_samples
%         break;
%     end
    % Check whether i-th sample has a solution
    if all(all(wind_samples(:, :, i) == 0.))
        fprintf('Sample %d skipped\n', i);
        continue;
    end

    % Add wind and load profile
    wind_profile = WindProfile(wind_samples(:, :, i), iwind);
    load_profile = LoadProfile(load_samples(:, :, i), load_buses);


    UC_profile = UCProfile(UC_samples(:, :, i), thermal_gen_no);


    profiles = getprofiles(wind_profile);
    profiles = getprofiles(load_profile, profiles);

    profiles = getprofiles(UC_profile, profiles);



    % Package all files
    mdi = loadmd(mpc, nt, xgd, [], [], profiles);



    % Set solver options
    mpopt = mpoption('verbose', 0, 'most.dc_model', 1, 'most.solver', 'GUROBI');



    %%%%%%%%%%%%%%%%%%%%% Run solver %%%%%%%%%%%%%%%%%%%%%
    try
        mdo = most(mdi, mpopt);
    catch
        fprintf("mdo failed for sample %d, skipped\n", i);
        continue;
    end

    %%%%%%%%%%%%%%%%%%%%%%%% Summary %%%%%%%%%%%%%%%%%%%%%
    try
        ms = most_summary(mdo);
    catch
        fprintf("ms failed for sample %d, skipped\n", i);
        continue;
    end



    %%%%%%% Check whether wind is preferably used %%%%%%%%
    deployed_wind = ms.Pg((num_gens+num_loads+1):end, :);
    wind_Pg = profiles(1).values;
    wind_Pg = reshape(wind_Pg, [nt, num_winds])';
    if any(any(round(wind_Pg, 0) ~= round(deployed_wind, 0)))
        fprintf('Sample %d wind is not completely used, skipped!\n', i);
        continue;
    end
    % Save deployed wind power
    save_path = ['./zero_reserve_outputs/deployed_wind/sample_' num2str(i) '.csv'];
    writematrix(deployed_wind, save_path, 'WriteMode', 'overwrite');

    

    %%%%%%% Check whether there is load shedding %%%%%%%%%
    deployed_load = -ms.Pg(num_gens+1:num_gens+num_loads, :);
    load_demand = profiles(2).values;
    load_demand = reshape(load_demand, [nt, num_loads])';
    if round(load_demand, 0) == round(deployed_load, 0)
        load_shed_bidx(num) = 0;
    end
    % Save deployed load
    save_path = ['./zero_reserve_outputs/deployed_load/sample_' num2str(i) '.csv'];
    writematrix(deployed_load, save_path, 'WriteMode', 'overwrite');

    num = num + 1;


    %%%%%% Save wind and load inputs into MATPOWER %%%%%%%
    save_path = ['./zero_reserve_inputs/wind/sample_' num2str(i) '.csv'];
    writematrix(wind_samples(:, :, i)', save_path, WriteMode="overwrite");

    save_path = ['./zero_reserve_inputs/load/sample_' num2str(i) '.csv'];
    writematrix(load_samples(:, :, i)', save_path, WriteMode="overwrite");




    %%%%%%%%%%%%%%%%%% Save solution %%%%%%%%%%%%%%%%%%%%%
    % Get Boolean index of wind gens
    bidx = ismember(gen_buses, wind_buses);

    % Save thermal gens UC
    save_path = ['./zero_reserve_outputs/UC/sample_' num2str(i) '.csv'];
    res_UC = ms.u(1:num_gens, :);
    res_UC = res_UC(~bidx, :);
    writematrix(res_UC, save_path, 'WriteMode', 'overwrite');

    % Save thermal gens Pg
    save_path = ['./zero_reserve_outputs/PG/sample_' num2str(i) '.csv'];
    res_PG = ms.Pg(1:num_gens, :);
    res_PG = res_PG(~bidx, :);
    writematrix(res_PG, save_path, 'WriteMode', 'overwrite');

    % Save complete Pg
    save_path = ['./zero_reserve_outputs/complete_PG/sample_' num2str(i) '.csv'];
    writematrix(ms.Pg, save_path, 'WriteMode', 'overwrite');

    % Save Pf
    save_path = ['./zero_reserve_outputs/PF/sample_' num2str(i) '.csv'];
    writematrix(ms.Pf, save_path, 'WriteMode', 'overwrite');

    fprintf('The %d-th sample is generated successfully!\n', i);
end

% Record overall computation time
end_time = toc(start_time);
fprintf('The computation time for %.d SCUC samples is %.f s.\n', num_samples, end_time);

% Save wind usage bool index 
% save_path = './outputs/wind_usage/bidx.csv';
% writematrix(wind_usage_bidx, save_path, WriteMode="overwrite");

% Save load shedding bool index
save_path = './zero_reserve_outputs/load_shed/bidx.csv';
writematrix(load_shed_bidx, save_path, WriteMode="overwrite");





















































