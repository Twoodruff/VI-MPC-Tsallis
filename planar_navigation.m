%% 
% Variational Inference MPC using Tsallis Divergence
% Ziyi Wang et al.
% arxiv 2021

% var = value; %description, default value (if given)
%% Problem Setup

T = 48; % MPC horizon, 96

%%% DYNAMICS %%%
n = 4; % state dimension
m = 2; % control dimension
p = 2; % measurement dimension


dt = 0.02; % time step, 0.01
sig_n = 1; % noise covariance, 1

% double integrator
A = zeros(n);               %transition matrix
A(1:n/2,n/2+1:end) = eye(n/2);
B = zeros(n,m);             %control matrix
B(n/2+1:end,:) = eye(m);
Qw = sig_n*eye(4);          %noise matrix
Qw(1:2,1:2) = zeros(2);

% Euler discretization
F = eye(n)+A*dt;
G = B*dt;
Q = dt*Qw;

xkp1 = @(xk,uk,wk) F*xk + G*uk + sqrt(Q)*wk; %d-t dynamics fcn

% Measurement
H = zeros(n/2,n); %measurement matrix
H(1:n/2,1:n/2) = eye(n/2);
Rv = 0.1*eye(p);  %measurement noise cov.

meas = @(xk,vk) H*xk + vk; %measurement fcn

%%% ENVIRONMENT %%%
start_pos = [-9; -9; 0; 0];
goal_pos  = [9; 9; 0; 0];

obs_origin = [-7; -7];
obs_size = 2; 
num_obs = 4;
obs = obstacles(obs_origin,obs_size,num_obs);
%plot(obs)

%%% COST %%%
Qc = diag([0.5 0.5 0.2 0.2]);       %state cost, [0.5 0.5 0.2 0.2]
Qcf = diag([0.25 0.25 1.25 1.25]);  %state terminal cost, [0.25 0.25 1 1]
Rc = diag([0.01 0.01]);             %control cost, [0.01, 0.01]
crash_cost = 10e3; %10e3

Qmat = [];
Rmat = [];
for t = 1:(T-1)
   Qmat = blkdiag(Qmat, Qc); 
   Rmat = blkdiag(Rmat, Rc);
end

%%% LEARNED POLICY %%%
% Unimodal Gaussian
% @uniGaussian, @uniGaussianUpdate, @uniGaussianpdf
init_mu = zeros(m,T-1);
init_Sigma = repmat(18.667^2*eye(m),1,1,T-1); %18.667^2

% Gaussian Mixture
% @GaussianMix, @GaussianMixUpdate
% {phi,{mu_t,Sigma_t}t=0:T-1}l=1:L

% Stein Variational Policy
% @SVP, @SVPupdate

% Optimality Likelihood function
f_like = @(J) exp(-J);

% (from Table VI)
N = 60;     %control samples, 256
M = 7;      %trajectory rollouts
K = 1;      %optimization iterations, 1
Kwarm = 7;  %opt iters for first time step, 8

% OPTIMAL POLICY (from Table VII)
gamma = 0.18;  %threshold, elite fraction, 0.070 ?
r = 1.05;      %deformed exp/log parameter, 1.796
lambdat = (r-1)*gamma; 

%% Parameter Structure
% Store all the problem parameters in a structure for ease of information
% flow
params = struct('system',struct,'cost',struct,'opt',struct,'policy',struct','optpolicy',struct);

params.system.n = n;                            %state dimension
params.system.m = m;                            %control dimension
params.system.T = T;                            %MPC horizon
params.system.goal = goal_pos;                  %goal position
params.system.first = true;                     %first time step flag
params.system.dyn = xkp1;                       %system dynamics function

params.cost.Q = Qmat;                           %state cost matrix
params.cost.Qf = Qcf;                           %terminal cost matrix
params.cost.R = Rmat;                           %control cost matrix
params.cost.crash = crash_cost;                 %cost for a collision
params.cost.obs = obs;                          %obstacle polyshape

params.opt.N = N;                               %control samples
params.opt.M = M;                               %state rollouts
params.opt.K = K;                               %optimization iterations
params.opt.Kwarm = Kwarm;                       %opt iters for first time step

params.policy.class = @uniGaussian;             %policy class function (to draw samples)
params.policy.pdf   = @uniGaussianpdf;          %policy class pdf (to evaluate probabilities)
params.policy.update = @uniGaussianUpdate;      %parameter update function
params.policy.param = {init_mu,init_Sigma};     %policy parameters
params.policy.weights = zeros(N,10);            %sample weights (at each time)
params.policy.like  = f_like;                   %optimal likelihood fcn

params.optpolicy.r = r;                         %deformation parameter
params.optpolicy.gamma = gamma;                 %reparameterization variable
params.optpolicy.lambdat = lambdat;             %normalization weight (not required)

%% Control Loop
x = zeros(n,10);        %state
xh = zeros(n,10);       %state estimate
xhP = zeros(n,n,10);
y = zeros(p, 10);       %measurement
u = zeros(m, 10);       %controls
J = zeros(N,10);        %cost
mu_k = zeros(m,T-1,10); %policy mean vector at each time step
X = zeros(N,M,n,T,10);  %state samples

x(:,1) = start_pos; 
xh(:,1) = x(:,1); 
xhP(1:2,1:2,1) = diag([0.2, 0.2]);
k = 1; %discrete time index
sim_start = tic;
while norm(xh(1:2,k)-goal_pos(1:2)) > 0.8
    % CONTROL
    %sample initial state (line 3)
    xk_samples = mvnrnd(x(:,k)',xhP(:,:,k),N*M);
    xk_samples = reshape(xk_samples,N,M,n);
    
    %policy evaluation (lines 5-14)
    if k > 1
        params.system.first = false;
    end
    [params.policy,J(:,k),X(:,:,:,:,k),wq] = policy_update(xk_samples,params);
    params.policy.weights(:,k) = wq;
    
    %determine control (line 15)
    %use the mean value of the policy for unimodal Gaussian
    %apply only the first value in sequence
    u(:,k) = params.policy.param{1}(:,1);
    mu_k(:,:,k) = params.policy.param{1};
    
    %shift control policy (line 16)
    params.policy.param{1}(:,1:end-1) = params.policy.param{1}(:,2:end);
    params.policy.param{1}(:,end) = params.policy.param{1}(:,end);
    params.policy.param{2}(:,:,1:end-1) = params.policy.param{2}(:,:,2:end);
    params.policy.param{2}(:,:,end) = params.policy.param{2}(:,:,end);
    
    
    % PROPAGATE DYNAMICS
    x(:,k+1) = xkp1(x(:,k), u(:,k), randn(n,1));
    
    % Check for collision
    collide = isinterior(obs,x(1,k+1),x(2,k+1));
    if any(collide)
        fprintf('There was a collision!')
        break
    end
    
    % Advance time step
    k = k + 1;
    if k == 200
        break
    end
    
    
    % MEASUREMENT
    y(:,k) = meas(x(:,k),sqrtm(Rv)*randn(p,1));
    
    
    % STATE ESTIMATION (KF)
    %prediction
    xhp = F*xh(:,k-1) + G*u(:,k-1);
    xhPp = F*xhP(:,:,k-1)*F' + Q;
    
    %update
    Kk = xhPp*H'/(H*xhPp*H' + Rv);
    xh(:,k) = xhp + Kk*(y(:,k) - H*xhp);
    xhP(:,:,k) = xhPp - Kk*H*xhPp;
    
end
sim_end = toc(sim_start);
fprintf('Simulation Time: %2.2f min \n',sim_end/60)

%% Plotting
close all

% Plot the real and estimated position
figure(1)
plot(obs,'FaceColor',[0 0.4470 0.7410]) 
hold on
plot(start_pos(1),start_pos(2),'k*','MarkerSize',15)
plot(goal_pos(1),goal_pos(2),'b*','MarkerSize',15)
plot(x(1,:),x(2,:),'b-*') %real
plot(xh(1,:),xh(2,:),'g-*') %est
xlabel('x')
ylabel('y')
title('Position')
legend('Obstacles','Start','Goal','Real','Est.','Location','northwest')
axis([-15 15 -15 15])

% Plot real and estimated velocities
figure(2)
plot(start_pos(3),start_pos(4),'k*','MarkerSize',15)
plot(goal_pos(3),goal_pos(4),'b*','MarkerSize',15)
plot(x(3,:),x(4,:),'r-*'), hold on
% plot(xh(3,:),xh(4,:),'g-*')
plot(x(3,1),x(4,1),'k*','MarkerSize',20)
plot(x(3,end),x(4,end),'b*','MarkerSize',20)

% Plot the cost
figure(3)
plot(1:k-1,mean(J))
xlabel('k')
ylabel('J(U)')
title('Average Marginal Control Cost')

%%
% Animated plotting of trajectory samples
figure(4)
for i = 1:k-1
   plot(obs,'FaceColor',[0 0.4470 0.7410])
   hold on
   axis([-15 15 -15 15])
   plot(x(1,i),x(2,i),'r*') %current position
   for j = 1:N
      plot(mean(squeeze(X(j,:,1,:,i))),mean(squeeze(X(j,:,2,:,i)))) %samples averaged over the M state rollouts
   end
   pause(0.1)
   hold off
end
%% Animated plotting of weighted samples
figure(5)
save = 0; %1: save, 0: don't save
if save
    set(gca,'nextplot','replacechildren');
    video = VideoWriter('weighted-traj1.avi'); %set video name here
    open(video);
end

for i = 1:k-1
   plot(obs,'FaceColor',[0 0.4470 0.7410])
   hold on
   axis([-15 15 -15 15])
   plot(x(1,i),x(2,i),'r*') %current position
   for j = 1:N
      plot(mean(squeeze(X(j,:,1,:,i))),mean(squeeze(X(j,:,2,:,i))),'b',...
          'LineWidth',5*(params.policy.weights(j,i)+1e-10)) %weighted samples
   end
   if save
       frame = getframe(gcf);
       writeVideo(video,frame);
   end
   pause(0.1)
   hold off
end

if save
    close(video);
end



