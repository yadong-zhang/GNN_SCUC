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
corr_matrix = readmatrix('../corr_mat/corr_matrix.xlsx');   

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

normPDF1 = makedist('Normal', 'mu', mu1, 'sigma', sigma1);   % Zone 1
trucNormPDF1 = normPDF1.truncate(left_truc1, right_truc1);

normPDF2 = makedist('Normal', 'mu', mu2, 'sigma', sigma2);   % Zone 2
trucNormPDF2 = normPDF2.truncate(left_truc2, right_truc2);

normPDF3 = makedist('Normal', 'mu', mu3, 'sigma', sigma3);   % Zone 3
trucNormPDF3 = normPDF3.truncate(left_truc3, right_truc3);

% % Define wind distribution
% Set PDF params
a1 = 8;  % Scale param
b1 = 2;  % Shape param

a2 = 8.2; 
b2 = 1.8; 

a3 = 7.8; 
b3 = 2.2; 

WeibullPDF1 = makedist('Weibull', 'a', a1, 'b', b1);             % Zone 1
WeibullPDF2 = makedist('Weibull', 'a', a2, 'b', b2);             % Zone 2
WeibullPDF3 = makedist('Weibull', 'a', a3, 'b', b3);             % Zone 3


% % Generate load and wind samples
for i = 1:nt
    % Load samples
    load_samples(1, :, i) = trucNormPDF1.icdf(correlated_CDF(1, :, i));
    load_samples(2, :, i) = trucNormPDF2.icdf(correlated_CDF(2, :, i));
    load_samples(3, :, i) = trucNormPDF3.icdf(correlated_CDF(3, :, i));

    % Wind samples
    wind_samples(1, :, i) = WeibullPDF1.icdf(correlated_CDF(4, :, i));
    wind_samples(2, :, i) = WeibullPDF2.icdf(correlated_CDF(5, :, i));
    wind_samples(3, :, i) = WeibullPDF3.icdf(correlated_CDF(6, :, i));
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




























