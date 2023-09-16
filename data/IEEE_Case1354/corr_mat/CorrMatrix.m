function CorrMatrix()

% Generate correlation matrix


% Set random seed
rng(20);

% Generate random matrix
mat = randn(16);
for i = 1:10
    mat = mat + randn(16);
end

% Get the correlation corfficient matrix
corr_matrix = corrcoef(mat);

% Save the data
file_path = './corr_matrix.csv';
writematrix(corr_matrix, file_path, WriteMode="overwrite");

writematrix(eig(corr_matrix), 'eigs.csv', WriteMode="overwrite");

end