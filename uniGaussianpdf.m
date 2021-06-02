function pdf = uniGaussianpdf(U, mu, Sigma)
%UNIGAUSSIANPDF Evaluate the policy values for a unimodal Gaussian with
%given mu and Sigma at the control samples u.
%   U should be specified with each row as the control at t such that the
%   number of columns equals T-1. [U:(m,T-1)] mu and Sigma should be
%   specified accordingly.

if size(U) ~= size(mu)
   U = repmat(U,size(mu,1),size(mu,2)); 
end

% product of individual control distributions at each time step
pdf = 1;
for t=1:size(mu,2)
    pdf = pdf*mvnpdf(U(:,t)',mu(:,t)',Sigma(:,:,t))';
end

end

