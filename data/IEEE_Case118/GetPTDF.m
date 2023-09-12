function PTDF = GetPTDF()

% Get PTDF matrix 

[mpc, xgd_table, wind_UC] = GetFiles();

PTDF = makePTDF(mpc);

% Save PTDF matrix
file_path = './PTDF/PTDF_matrix.csv';
writematrix(PTDF, file_path, 'WriteMode','overwrite');

end