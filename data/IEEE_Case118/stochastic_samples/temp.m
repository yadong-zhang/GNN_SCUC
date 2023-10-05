rng('default');

num_zones = 3;

% Get num of variables
num_vars = num_zones * 2;

num_samples = 1000;

% Create initial state
mu = 0;
sigma = 1;

init_state = zeros(num_vars, num_samples);

for i = 1:num_vars
    init_state(i, :) = lhsnorm(mu, sigma, num_samples, 'on');
end


%%
corr_matrix = readmatrix('../corr_mat/corr_matrix.xlsx');   % Read correlation matrix

% Calculate Cholesky factorization
L = chol(corr_matrix, 'lower');

correlated_std_x = L*init_state;

cdf = normcdf(correlated_std_x);


%%
% Define marginals

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

%%
X1 = trucNormPDF1.icdf(cdf(1, :))';
X2 = trucNormPDF2.icdf(cdf(2, :))';
X3 = trucNormPDF3.icdf(cdf(3, :))';
X4 = WeibullPDF1.icdf(cdf(4, :))';
X5 = WeibullPDF2.icdf(cdf(5, :))';
X6 = WeibullPDF3.icdf(cdf(6, :))';

X = [X1, X2, X3, X4, X5, X6];

%%
subplot(1, 2, 1);
imagesc(corr(X));
subplot(1, 2, 2);
imagesc(corr_matrix);



























