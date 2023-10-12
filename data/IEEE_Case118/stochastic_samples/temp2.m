
% Generate spatio-temporal correlated load and wind samples
% nt: time steps

% Set random seed
rng('default');

num_zones = 3;

num_samples = 1000;

nt = 5;

% Get num of variables
num_vars = num_zones * 2;

% Create initial state
mu = 0;
sigma = 1;

init_state = zeros(num_vars, num_samples);

for i = 1:num_vars
    init_state(i, :) = lhsnorm(mu, sigma, num_samples, 'on');
end

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
    load_samples(1, :, i) = trucNormPDF1.icdf(correlated_CDF(1, :, i));
    load_samples(2, :, i) = trucNormPDF2.icdf(correlated_CDF(2, :, i));
    load_samples(3, :, i) = trucNormPDF3.icdf(correlated_CDF(3, :, i));

    % Wind samples
    wind_samples(1, :, i) = WeibullPDF1.icdf(correlated_CDF(4, :, i));
    wind_samples(2, :, i) = WeibullPDF2.icdf(correlated_CDF(5, :, i));
    wind_samples(3, :, i) = WeibullPDF3.icdf(correlated_CDF(6, :, i));
end


% % % Convert wind speed to wind power
% left_truc1 = 0.5;    % Left truncation
% right_truc1 = 15;    % Right truncation
% 
% left_truc2 = 2;  
% right_truc2 = 18; 
% 
% left_truc3 = 1;  
% right_truc3 = 20;  
% 
% Pr1 = 300;   % Set maximum power generation capacity as 300 MW
% Pr2 = 300;
% Pr3 = 300;
% 
% wind_samples(1, :, :) = Pr1*(wind_samples(1, :, :).^3 - left_truc1^3)/(right_truc1^3 - left_truc1^3);
% wind_samples(2, :, :) = Pr2*(wind_samples(2, :, :).^3 - left_truc2^3)/(right_truc2^3 - left_truc2^3);
% wind_samples(3, :, :) = Pr3*(wind_samples(3, :, :).^3 - left_truc3^3)/(right_truc3^3 - left_truc3^3);
% 


%%
a = 8;  % Scale param
b = 2;  % Shape param
WeibullPDF = makedist('Weibull', 'a', a, 'b', b);             % Zone 1

x = linspace(0, 50, 1000);
y = WeibullPDF.pdf(x);

rv = WeibullPDF.random([1000 1]);

temp = 100 * (rv.^3 - 0.5^3) / (15.^3 - 0.5^3);
temp = max(temp, zeros([size(temp, 1)], 1));
converted_rv = min(temp, 100*ones([size(temp, 1)], 1));

figure(1);
plot(x, y);

figure(2);
subplot(1, 2, 1);
histogram(rv, 100, 'Normalization', 'pdf');

subplot(1, 2, 2);
histogram(converted_rv, 100, 'Normalization', 'pdf');


%%
a = 8;  % Scale param
b = 2.;  % Shape param
WeibullPDF = makedist('Weibull', 'a', a, 'b', b);             % Zone 1

x = linspace(0, 30, 1000);
y = WeibullPDF.pdf(x);

rv = WeibullPDF.random([1000 1]);

plot(x, y, 'LineWidth', 3);
hold on;
histogram(rv, 100, 'Normalization', 'pdf');

%%
mu = 50;   % Define distribution params
sigma = 15;
left_truc = 10;
right_truc = 90;

normPDF = makedist('Normal', 'mu', mu, 'sigma', sigma);   % Zone 1
trucNormPDF = normPDF1.truncate(left_truc, right_truc);

x = linspace(0, 100, 1000);
pdf = trucNormPDF.pdf(x);

rv = trucNormPDF.icdf(correlated_CDF(1, :, 5));

plot(x, pdf);
hold on;
histogram(rv, 10, 'Normalization', 'pdf');






















