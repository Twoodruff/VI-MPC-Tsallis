function [updated_policy,wq] = uniGaussianUpdate(Ju,U,params)
%UNIGAUSSIANUPDATE Update the mean and variance for a unimodal Gaussian
%control policy.
%   Detailed explanation goes here

% Initialize output using input (to ensure matching data structures)
updated_policy = params.policy.param;
T = size(updated_policy{1},2); %note this is actually T-1

% Generate samples of optimal policy
global r gamma
r = params.optpolicy.r;
gamma = params.optpolicy.gamma;
N = params.opt.N;
qs = zeros(N,1); %optimal policy, q*(U)

for n = 1:N
    %q_samples(n) = OptPolicy(Ju(n),squeeze(U(n,:,:)),params); %NOT USED
    
    % optimality likelihood of cost sample J
    likelihood = expr(Ju(n));

    % prior policy of control sample U
    mu_p = params.policy.param{1};
    Sigma_p = params.policy.param{2};
    pU = params.policy.pdf(squeeze(U(n,:,:)),mu_p,Sigma_p);
    
    % computing optimal policy q* (Eqn 10 numerator)
    %NOTE: the denominator is not required as the policy will be
    %normalized when computing the weights via Eq 13
    qs(n) = likelihood*pU; 
end

% Calculate the weights (Eqn 13) 
sumqs = sum(qs);
if sumqs == 0
    wq = zeros(size(qs));
else
    wq = qs/sumqs;
end

% Update the mean (Eqn 14)
mu = updated_policy{1};
for t = 1:T
    mu(:,t) = U(:,:,t)'*wq;
end
updated_policy{1} = mu;

%NOTE: covariance is constant for unimodal Gaussian case
% Update the covariance (Eqn 15)
% Sigma = updated_policy{2};
% for t = 1:T
%     for n = 1:N
%         diff = (U(n,:,t)'-mu(:,t));
%         Sigma(:,:,t) = Sigma(:,:,t) + wq(n)*(diff*diff'); 
%     end
% end
% updated_policy{2} = Sigma;

end

%% Deformed Exponential (see Eqns 6,28)
function f = expr(x)
% Deformed exponential with reparameterized lambda
% r : deformation parameter
% gamma : threshold
global r gamma

if r <= 0
    error('ERROR: r must be greater than zero in ''expr''. r = %f \n',r)
end

f = 0;
if x < gamma
    f = exp( (1/(r-1)) * log(1 - x/gamma) );
end

end

