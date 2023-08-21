% Human_Motion_Gen : A fast customized optimization solver.
% 
% Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.
% 
% 
% This software is intended for simulation and testing purposes only. 
% Use of this software for any commercial purpose is prohibited.
% 
% This program is distributed in the hope that it will be useful.
% EMBOTECH makes NO WARRANTIES with respect to the use of the software 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
% PARTICULAR PURPOSE. 
% 
% EMBOTECH shall not have any liability for any damage arising from the use
% of the software.
% 
% This Agreement shall exclusively be governed by and interpreted in 
% accordance with the laws of Switzerland, excluding its principles
% of conflict of laws. The Courts of Zurich-City shall have exclusive 
% jurisdiction in case of any dispute.
% 
% [OUTPUTS] = Human_Motion_Gen(INPUTS) solves an optimization problem where:
% Inputs:
% - lb - matrix of size [195x1]
% - ub - matrix of size [195x1]
% - x0 - matrix of size [210x1]
% - xinit - matrix of size [15x1]
% - all_parameters - matrix of size [230x1]
% - num_of_threads - scalar
% Outputs:
% - outputs - column vector of length 210
function [outputs] = Human_Motion_Gen(lb, ub, x0, xinit, all_parameters, num_of_threads)
    
    [output, ~, ~] = Human_Motion_GenBuildable.forcesCall(lb, ub, x0, xinit, all_parameters, num_of_threads);
    outputs = coder.nullcopy(zeros(210,1));
    outputs(1:21) = output.x01;
    outputs(22:42) = output.x02;
    outputs(43:63) = output.x03;
    outputs(64:84) = output.x04;
    outputs(85:105) = output.x05;
    outputs(106:126) = output.x06;
    outputs(127:147) = output.x07;
    outputs(148:168) = output.x08;
    outputs(169:189) = output.x09;
    outputs(190:210) = output.x10;
end
