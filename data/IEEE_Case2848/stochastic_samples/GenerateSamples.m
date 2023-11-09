function [load_samples, wind_samples] = GenerateSamples(init_state, nt)

% Generate spatio-temporal correlated load and wind samples
% nt: time steps

% Set random seed
rng('default');

% Initial state
num_vars = size(init_state, 1);
num_samples = size(init_state, 2);

% Array to store CDF values
correlated_CDF = zeros(num_vars, num_samples, nt);    
% Read correlation matrix
corr_matrix = readmatrix('../corr_mat/corr_matrix.csv');   

% Calculate Cholesky factorization
L = chol(corr_matrix, 'lower'); 

% Create matrix to store random samples
X = zeros(num_vars, num_samples, nt);            % Intermediate normal variables
load_samples = zeros(num_vars/2, num_samples, nt);       % Load samples
wind_samples = zeros(num_vars/2, num_samples, nt);       % Wind samples

% Generate nt = 1 intermediate samples
X(:, :, 1) = init_state;

% Generate nt = 2, 3, ... intermediate variables
for i = 2:nt
    % Random walk Markov chain
    X(:, :, i) = X(:, :, i-1) + randn([num_vars, num_samples]);
end

for i = 1:nt
    % Convert back to std normal
    X(:, :, i) = X(:, :, i) / sqrt(i);

    % Convert to correlated std normal
    X(:, :, i) = L * X(:, :, i);

    % Calculate CDF using std normal distribution
    correlated_CDF(:, :, i) = normcdf(X(:, :, i));
end

% % Define load distribution
mu1 = 50;   % Define distribution params
sigma1 = 15;
left_truc1 = 10;
right_truc1 = 90;

mu2 = 80;
sigma2 = 20;
left_truc2 = 25;
right_truc2 = 125;

mu3 = 100;
sigma3 = 15;
left_truc3 = 60;
right_truc3 = 140;

mu4 = 48;
sigma4 = 12;
left_truc4 = 9;
right_truc4 = 87;

mu5 = 81;
sigma5 = 21;
left_truc5 = 23;
right_truc5 = 139;

mu6 = 98;
sigma6 = 15;
left_truc6 = 60;
right_truc6 = 136;

mu7 = 52;
sigma7 = 14;
left_truc7 = 7;
right_truc7 = 97;

mu8 = 83;
sigma8 = 20;
left_truc8 = 21;
right_truc8 = 145;

mu9 = 45;   % Define distribution params
sigma9 = 12;
left_truc9 = 14;
right_truc9 = 76;

mu10 = 85;
sigma10 = 18;
left_truc10 = 25;
right_truc10 = 145;

mu11 = 98;
sigma11 = 16;
left_truc11 = 55;
right_truc11 = 141;

mu12 = 53;
sigma12 = 13;
left_truc12 = 9;
right_truc12 = 97;

mu13 = 84;
sigma13 = 20;
left_truc13 = 23;
right_truc13 = 139;

mu14 = 98;
sigma14 = 15;
left_truc14 = 55;
right_truc14 = 141;

mu15 = 47;
sigma15 = 12;
left_truc15 = 8;
right_truc15 = 86;

mu16 = 85;
sigma16 = 20;
left_truc16 = 21;
right_truc16 = 149;

normPDF1 = makedist('Normal', 'mu', mu1, 'sigma', sigma1);   % Zone 1
trucNormPDF1 = normPDF1.truncate(left_truc1, right_truc1);

normPDF2 = makedist('Normal', 'mu', mu2, 'sigma', sigma2);   % Zone 2
trucNormPDF2 = normPDF2.truncate(left_truc2, right_truc2);

normPDF3 = makedist('Normal', 'mu', mu3, 'sigma', sigma3);   % Zone 3
trucNormPDF3 = normPDF3.truncate(left_truc3, right_truc3);

normPDF4 = makedist('Normal', 'mu', mu4, 'sigma', sigma4);   % Zone 4
trucNormPDF4 = normPDF4.truncate(left_truc4, right_truc4);

normPDF5 = makedist('Normal', 'mu', mu5, 'sigma', sigma5);   % Zone 5
trucNormPDF5 = normPDF5.truncate(left_truc5, right_truc5);

normPDF6 = makedist('Normal', 'mu', mu6, 'sigma', sigma6);   % Zone 6
trucNormPDF6 = normPDF6.truncate(left_truc6, right_truc6);

normPDF7 = makedist('Normal', 'mu', mu7, 'sigma', sigma7);   % Zone 7
trucNormPDF7 = normPDF7.truncate(left_truc7, right_truc7);

normPDF8 = makedist('Normal', 'mu', mu8, 'sigma', sigma8);   % Zone 8
trucNormPDF8 = normPDF8.truncate(left_truc8, right_truc8);

normPDF9 = makedist('Normal', 'mu', mu9, 'sigma', sigma9);   % Zone 9
trucNormPDF9 = normPDF9.truncate(left_truc9, right_truc9);

normPDF10 = makedist('Normal', 'mu', mu10, 'sigma', sigma10);   % Zone 10
trucNormPDF10 = normPDF10.truncate(left_truc10, right_truc10);

normPDF11 = makedist('Normal', 'mu', mu11, 'sigma', sigma11);   % Zone 11
trucNormPDF11 = normPDF11.truncate(left_truc11, right_truc11);

normPDF12 = makedist('Normal', 'mu', mu12, 'sigma', sigma12);   % Zone 12
trucNormPDF12 = normPDF12.truncate(left_truc12, right_truc12);

normPDF13 = makedist('Normal', 'mu', mu13, 'sigma', sigma13);   % Zone 13
trucNormPDF13 = normPDF13.truncate(left_truc13, right_truc13);

normPDF14 = makedist('Normal', 'mu', mu14, 'sigma', sigma14);   % Zone 14
trucNormPDF14 = normPDF14.truncate(left_truc14, right_truc14);

normPDF15 = makedist('Normal', 'mu', mu15, 'sigma', sigma15);   % Zone 15
trucNormPDF15 = normPDF15.truncate(left_truc15, right_truc15);

normPDF16 = makedist('Normal', 'mu', mu16, 'sigma', sigma16);   % Zone 16
trucNormPDF16 = normPDF16.truncate(left_truc16, right_truc16);


% % Define wind distribution
% Set PDF params
a1 = 8;  % Scale param
b1 = 2;  % Shape param

a2 = 8.2; 
b2 = 1.8; 

a3 = 12; 
b3 = 8; 

a4 = 8.1; 
b4 = 1.9; 

a5 = 7.9; 
b5 = 2.1; 

a6 = 7.7; 
b6 = 1.9; 

a7 = 8; 
b7 = 2.1; 

a8 = 8.3; 
b8 = 2.0; 

a9 = 7.8;  % Scale param
b9 = 1.8;  % Shape param

a10 = 8.1; 
b10 = 1.8; 

a11 = 12.5; 
b11 = 8.1; 

a12 = 8.5; 
b12 = 2.2; 

a13 = 7.5; 
b13 = 1.8; 

a14 = 7.7; 
b14 = 2; 

a15 = 8; 
b15 = 2; 

a16 = 8.3; 
b16 = 2.5; 

WeibullPDF1 = makedist('Weibull', 'a', a1, 'b', b1);             % Zone 1
WeibullPDF2 = makedist('Weibull', 'a', a2, 'b', b2);             % Zone 2
WeibullPDF3 = makedist('Weibull', 'a', a3, 'b', b3);             % Zone 3
WeibullPDF4 = makedist('Weibull', 'a', a4, 'b', b4);             % Zone 4
WeibullPDF5 = makedist('Weibull', 'a', a5, 'b', b5);             % Zone 5
WeibullPDF6 = makedist('Weibull', 'a', a6, 'b', b6);             % Zone 6
WeibullPDF7 = makedist('Weibull', 'a', a7, 'b', b7);             % Zone 7
WeibullPDF8 = makedist('Weibull', 'a', a8, 'b', b8);             % Zone 8
WeibullPDF9 = makedist('Weibull', 'a', a9, 'b', b9);             % Zone 9
WeibullPDF10 = makedist('Weibull', 'a', a10, 'b', b10);             % Zone 10
WeibullPDF11 = makedist('Weibull', 'a', a11, 'b', b11);             % Zone 11
WeibullPDF12 = makedist('Weibull', 'a', a12, 'b', b12);             % Zone 12
WeibullPDF13 = makedist('Weibull', 'a', a13, 'b', b13);             % Zone 13
WeibullPDF14 = makedist('Weibull', 'a', a14, 'b', b14);             % Zone 14
WeibullPDF15 = makedist('Weibull', 'a', a15, 'b', b15);             % Zone 15
WeibullPDF16 = makedist('Weibull', 'a', a16, 'b', b16);             % Zone 16

% % Generate load and wind samples
for i = 1:nt
    % Load samples
    load_samples(1, :, i) = trucNormPDF1.icdf(correlated_CDF(1, :, i));
    load_samples(2, :, i) = trucNormPDF2.icdf(correlated_CDF(2, :, i));
    load_samples(3, :, i) = trucNormPDF3.icdf(correlated_CDF(3, :, i));
    load_samples(4, :, i) = trucNormPDF4.icdf(correlated_CDF(4, :, i));
    load_samples(5, :, i) = trucNormPDF5.icdf(correlated_CDF(5, :, i));
    load_samples(6, :, i) = trucNormPDF6.icdf(correlated_CDF(6, :, i));
    load_samples(7, :, i) = trucNormPDF7.icdf(correlated_CDF(7, :, i));
    load_samples(8, :, i) = trucNormPDF8.icdf(correlated_CDF(8, :, i));
    load_samples(9, :, i) = trucNormPDF9.icdf(correlated_CDF(9, :, i));
    load_samples(10, :, i) = trucNormPDF10.icdf(correlated_CDF(10, :, i));
    load_samples(11, :, i) = trucNormPDF11.icdf(correlated_CDF(11, :, i));
    load_samples(12, :, i) = trucNormPDF12.icdf(correlated_CDF(12, :, i));
    load_samples(13, :, i) = trucNormPDF13.icdf(correlated_CDF(13, :, i));
    load_samples(14, :, i) = trucNormPDF14.icdf(correlated_CDF(14, :, i));
    load_samples(15, :, i) = trucNormPDF15.icdf(correlated_CDF(15, :, i));
    load_samples(16, :, i) = trucNormPDF16.icdf(correlated_CDF(16, :, i));

    % Wind samples
    wind_samples(1, :, i) = WeibullPDF1.icdf(correlated_CDF(17, :, i));
    wind_samples(2, :, i) = WeibullPDF2.icdf(correlated_CDF(18, :, i));
    wind_samples(3, :, i) = WeibullPDF3.icdf(correlated_CDF(19, :, i));
    wind_samples(4, :, i) = WeibullPDF4.icdf(correlated_CDF(20, :, i));
    wind_samples(5, :, i) = WeibullPDF5.icdf(correlated_CDF(21, :, i));
    wind_samples(6, :, i) = WeibullPDF6.icdf(correlated_CDF(22, :, i));
    wind_samples(7, :, i) = WeibullPDF7.icdf(correlated_CDF(23, :, i));
    wind_samples(8, :, i) = WeibullPDF8.icdf(correlated_CDF(24, :, i));
    wind_samples(9, :, i) = WeibullPDF9.icdf(correlated_CDF(25, :, i));
    wind_samples(10, :, i) = WeibullPDF10.icdf(correlated_CDF(26, :, i));
    wind_samples(11, :, i) = WeibullPDF11.icdf(correlated_CDF(27, :, i));
    wind_samples(12, :, i) = WeibullPDF12.icdf(correlated_CDF(28, :, i));
    wind_samples(13, :, i) = WeibullPDF13.icdf(correlated_CDF(29, :, i));
    wind_samples(14, :, i) = WeibullPDF14.icdf(correlated_CDF(30, :, i));
    wind_samples(15, :, i) = WeibullPDF15.icdf(correlated_CDF(31, :, i));
    wind_samples(16, :, i) = WeibullPDF16.icdf(correlated_CDF(32, :, i));
end


% % Convert wind speed to wind power
left_truc = 1.;    % Left truncation
right_truc = 15;    % Right truncation

% Set maximum power generation capacity
Pr = 100;   

% Convert wind speed to wind power
temp = Pr*(wind_samples.^3 - left_truc^3)/(right_truc^3 - left_truc^3);

% Delete values less than zero
temp = max(temp, zeros([num_vars/2, num_samples, nt]));

% Delete values larger than Pr
wind_samples = min(temp, Pr*ones([num_vars/2, num_samples, nt]));


end




























