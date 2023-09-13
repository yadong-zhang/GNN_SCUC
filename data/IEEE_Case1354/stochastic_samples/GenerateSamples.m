function [load_samples, wind_samples] = GenerateSamples(init_state, nt)

% Generate spatio-temporal correlated load and wind samples
% nt: time steps

% Set random seed
rng(1);

% Initial state
num_vars = size(init_state, 1);
num_samples = size(init_state, 2);

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


% % Copula (strictly, not copula, as the input is not uniform variable)
correlated_CDF = zeros(num_samples, nt);    % Array to store CDF values
corr_matrix = readmatrix('../corr_mat/corr_matrix.xlsx');   % Read correlation matrix

for i = 1:nt
    % Get covariance matrix from correlation matrix
    temp = sqrt(diag(i*ones(num_vars, 1)));     % Diagonal matrix of variance
    cov_matrix =  temp * corr_matrix * temp;
    correlated_CDF(:, i) = mvncdf(X(:, :, i)', zeros(1, num_vars), cov_matrix);
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
a1 = 12;  % Scale param
b1 = 20;  % Shape param

a2 = 14; 
b2 = 8; 

a3 = 12; 
b3 = 8; 

WeibullPDF1 = makedist('Weibull', 'a', a1, 'b', b1);             % Zone 1
WeibullPDF2 = makedist('Weibull', 'a', a2, 'b', b2);             % Zone 2
WeibullPDF3 = makedist('Weibull', 'a', a3, 'b', b3);             % Zone 3


% % Generate load and wind samples
for i = 1:nt
    % Load samples
    load_samples(1, :, i) = trucNormPDF1.icdf(correlated_CDF(:, i));
    load_samples(2, :, i) = trucNormPDF2.icdf(correlated_CDF(:, i));
    load_samples(3, :, i) = trucNormPDF3.icdf(correlated_CDF(:, i));

    % Wind samples
    wind_samples(1, :, i) = WeibullPDF1.icdf(correlated_CDF(:, i));
    wind_samples(2, :, i) = WeibullPDF2.icdf(correlated_CDF(:, i));
    wind_samples(3, :, i) = WeibullPDF3.icdf(correlated_CDF(:, i));
end


% % Convert wind speed to wind power
left_truc1 = 0.5;    % Left truncation
right_truc1 = 15;    % Right truncation

left_truc2 = 2;  
right_truc2 = 18; 

left_truc3 = 1;  
right_truc3 = 20;  

Pr1 = 300;   % Set maximum power generation capacity as 300 MW
Pr2 = 300;
Pr3 = 300;

wind_samples(1, :, :) = Pr1*(wind_samples(1, :, :).^3 - left_truc1^3)/(right_truc1^3 - left_truc1^3);
wind_samples(2, :, :) = Pr2*(wind_samples(2, :, :).^3 - left_truc2^3)/(right_truc2^3 - left_truc2^3);
wind_samples(3, :, :) = Pr3*(wind_samples(3, :, :).^3 - left_truc3^3)/(right_truc3^3 - left_truc3^3);



end




























