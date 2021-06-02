function U = uniGaussian(parameters)
%UNIGAUSSIAN Unimodal Gaussian policy class.
%   Create a control sample over the MPC horizon using the policy
%   parameters. Input is a cell array with the mean and covariance in the
%   first and second elements, respectively. mu can be (T,1) col vector and
%   Sigma a (m,m,T) array.

% see main file (planar_navigation.m) for definition of params.policy.param
mu = parameters{1};
Sigma = parameters{2};

U = mvnrnd(mu',Sigma);

end

