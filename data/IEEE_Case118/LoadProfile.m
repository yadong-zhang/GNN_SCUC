function profile = LoadProfile(load_sample, load_buses)
%EX_LOAD_PROFILE  Example load profile data file for stochastic unit commitment.

%   MOST
%   Copyright (c) 2015-2016, Power Systems Engineering Research Center (PSERC)
%   by Ray Zimmerman, PSERC Cornell
%
%   This file is part of MOST.
%   Covered by the 3-clause BSD License (see LICENSE file for details).
%   See https://github.com/MATPOWER/most for more info.

% define constants
[CT_LABEL, CT_PROB, CT_TABLE, CT_TBUS, CT_TGEN, CT_TBRCH, CT_TAREABUS, ...
    CT_TAREAGEN, CT_TAREABRCH, CT_ROW, CT_COL, CT_CHGTYPE, CT_REP, ...
    CT_REL, CT_ADD, CT_NEWVAL, CT_TLOAD, CT_TAREALOAD, CT_LOAD_ALL_PQ, ...
    CT_LOAD_FIX_PQ, CT_LOAD_DIS_PQ, CT_LOAD_ALL_P, CT_LOAD_FIX_P, ...
    CT_LOAD_DIS_P, CT_TGENCOST, CT_TAREAGENCOST, CT_MODCOST_F, ...
    CT_MODCOST_X] = idx_ct;

profile = struct( ...
    'type', 'mpcData', ...
    'table', CT_TLOAD, ...
    'rows', 1, ...
    'col', CT_LOAD_ALL_PQ, ...
    'chgtype', 1, ...
    'values', [] ); % dim = (nt x num_scenarios x num_load_sites)

nt = size(load_sample, 1);
num_loads = size(load_sample, 2);

% Get load bus info
profile.rows = reshape(load_buses, [1, num_loads]);

% Assign loads to "profile.values"
profile.values = reshape(load_sample, [nt, 1, num_loads]);

end










































