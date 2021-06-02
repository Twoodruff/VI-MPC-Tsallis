function J = cost(X,U,params)
%COST Problem-based cost function evaluated over one (1) trajectory.
%   Calculate the cost function for a single trajectory. X is an n-by-T 
%   matrix of states and U is an m-by-(T-1) matrix of controls. It returns 
%   a single scalar value.

x_g     = params.system.goal;
T       = params.system.T;
obs     = params.cost.obs;
Qmat    = params.cost.Q;
Qf      = params.cost.Qf;
Rmat    = params.cost.R;
crash_cost = params.cost.crash;
Xg = repmat(x_g,T-1,1);

% TEMPORARY ----------------
% Plot the sample trajectory
% close all
% figure
% plot(obs)
% hold on
% plot(X(1,:),X(2,:),'*-')
% --------------------------

% Check collisions for the given trajectories
collide = isinterior(obs,X(1,:),X(2,:));
c = any(collide);

% Cost Function
Xc = reshape(X(:,1:end-1),[],1);
J = crash_cost*c + ...  %crash cost
    (X(:,end)-x_g)'*Qf*(X(:,end)-x_g) + ... %terminal cost
    (Xc-Xg)'*Qmat*(Xc-Xg) + ...   %stage cost (state)
    U(:)'*Rmat*U(:);  %stage cost (control)

end
