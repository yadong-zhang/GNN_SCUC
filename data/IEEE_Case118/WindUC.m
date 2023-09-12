function wind_UC = WindUC
%EX_WIND_UC  Example Wind data file for stochastic unit commitment.

%   MOST
%   Copyright (c) 2015-2016, Power Systems Engineering Research Center (PSERC)
%   by Ray Zimmerman, PSERC Cornell
%
%   This file is part of MOST.
%   Covered by the 3-clause BSD License (see LICENSE file for details).
%   See https://github.com/MATPOWER/most for more info.

%%-----  wind  -----
% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
wind_UC.gen = [
    6	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    12	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    24	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    36	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    42	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    55	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    59	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    70	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    74	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    77	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    85	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    89	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    91	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    100	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    105	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
    113	0	0	0	0	1	100	1	100	0	0	0	0	0	0	0	0	0	0	0	0	;
];
% xGenData
wind_UC.xgd_table.colnames = {
	'CommitKey', ...
		'CommitSched', ...
			'InitialPg', ...
				'RampWearCostCoeff', ...
					'PositiveActiveReservePrice', ...
						'PositiveActiveReserveQuantity', ...
							'NegativeActiveReservePrice', ...
								'NegativeActiveReserveQuantity', ...
									'PositiveActiveDeltaPrice', ...
										'NegativeActiveDeltaPrice', ...
											'PositiveLoadFollowReservePrice', ...
												'PositiveLoadFollowReserveQuantity', ...
													'NegativeLoadFollowReservePrice', ...
														'NegativeLoadFollowReserveQuantity', ...
};

wind_UC.xgd_table.data = [
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
    2	1	0	0	0	0	0	0	0	0	0	0	0	0	;
];

end
