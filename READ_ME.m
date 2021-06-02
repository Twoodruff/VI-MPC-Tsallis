%% Code Description for planar_navigation.m

%%% STRUCTURE %%%

% planar_navigation.m
%   policy_update.m
%       uniGaussian.m
%       cost.m
%           obstacles.m
%       uniGaussianUpdate.m
%           uniGaussianpdf.m


%%% DESCRIPTIONS %%%

% planar_navigation.m - 
% main file that initializes all system parameters and runs the outer
% control loop. Finds initial state samples, calls policy_update, shifts
% parameters for next time step, propagates dynamics, takes measurements,
% and does state estimation. 

% policy_update.m - 
% function that performs most of the VI-MPC algorithm. Generates trajectory
% samples, computes cost, performs parameter updates, and returns updated
% policy parameters.

% cost.m - 
% computes cost function for a given trajectory.

% obstacles.m - 
% generates the obstacles in the environment and is used in the cost
% function.

% uniGaussian, uniGaussianUpdate, uniGaussianpdf - 
% Unimodal Guassian functions to draw samples, do parameter update, and
% compute sample probabilities.
