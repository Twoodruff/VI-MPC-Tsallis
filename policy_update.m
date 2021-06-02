function [posterior_policy,Ju,X,weights] = policy_update(x0,params)
%POLICY_UPDATE Update the optimal policy parameters.
%   [posterior_policy,Ju,X,weights] = POLICY_UPDATE(x0,params) takes in an 
%   intial state distribution and set of parameters. The output is the 
%   updated policy parameters, the marginalized cost, the set of 
%   sampled trajectories resulting from the control samples, and the 
%   weights of each sample.

% x0: samples of initial state (N,M,n) matrix
% params: structure containing policy class and parameters

m   = params.system.m;  %control dim
n   = params.system.n;  %state dim
T   = params.system.T;  %MPC horizon
N   = params.opt.N;     %control samples
M   = params.opt.M;     %state samples
prior_policy = params.policy; %policy structure
sys = params.system.dyn;%system dynamics

posterior_policy = prior_policy;
posteriorParams = prior_policy.param;

U = zeros(N,m,T-1); %control samples
X = zeros(N,M,n,T); %state samples
J = zeros(N,M);     %cost samples J(X,U)
Ju = zeros(N,1);    %cost samples J(U)

if params.system.first == true
    K = params.opt.Kwarm;
else
   K = params.opt.K;  
end

%for l = 1:L
for k = 0:K %(line 5)
    X(:,:,:,1) = x0; %initial (current) state
    
    for i = 1:N %(line 6)
        % control samples (line 8a)
        U(i,:,:) = prior_policy.class(prior_policy.param)'; 

        for j = 1:M %(line 7)
            % trajectory samples (line 8b)
            for t = 2:T
                X(i,j,:,t) = sys(squeeze(X(i,j,:,t-1)), squeeze(U(i,:,t-1))', randn(n,1));
            end

            % Evaluate the cost of each trajectory (line 9)
            J(i,j) = cost(squeeze(X(i,j,:,:)), squeeze(U(i,:,:)),params);
        end %(line 10)

    end %(line 12)
    % Evaluate the expected likelihood cost (line 11)
    minJ = min(J,[],'all');
    maxJ = max(J,[],'all');
    Jnorm = (J(:,:)-minJ)/(maxJ-minJ);
    %Ju = (1/M)*sum(f_like(Jnorm),2); %in paper, but incorrect (I believe)
    Ju = (1/M)*sum(Jnorm,2);
    
    % Update policy (line 13)
    [posteriorParams,weights] = prior_policy.update(Ju,U,params); 
    prior_policy.param = posteriorParams;
end

posterior_policy.param = posteriorParams;

end

