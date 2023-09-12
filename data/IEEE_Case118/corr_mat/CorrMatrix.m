function CorrMatrix()

% Generate correlation matrix

corr_matrix = [[1., 0.7, 0.5, 0.1, 0.05, 0.03];
               [0.7, 1., 0.4, 0.02, 0.08, 0.05];
               [0.5, 0.4, 1., 0.06, 0.04, 0.1];
               [0.1, 0.02, 0.06, 1., 0.3, 0.4];
               [0.05, 0.08, 0.04, 0.3, 1., 0.6];
               [0.03, 0.05, 0.1, 0.4, 0.6, 1.]];

% Save the data
file_path = './corr_matrix.xlsx';
writematrix(corr_matrix, file_path, WriteMode="replacefile");

end