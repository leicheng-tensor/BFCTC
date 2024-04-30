function [model] = BBC_M_col_Wishart_TC(Y,or_X,varargin)
%   Bayesian Tucker decomposition for Tensor Completion
%
%  [model] = BBC_M_col_Wishart_TC(Y, X, 'PARAM1', val1, 'PARAM2', val2, ...)
%
%  INPUTS
%     Y              - Input tensor
%     or_X           - Original data for recovery (for monitoring purposes only)
%     'obs'          - Binary (0-1) missing indicator tensor of same size as Y
%                      (0: missing; 1: observed) 
%     'init'         - Initialization method
%                     - 'ml'  : SVD initilization (default)
%                     - 'rand': Random matrices
%     'maxRank'      - The initialization of rank (larger than true rank)
%     'maxiters'     - max number of iterations (default: 100)
%     'tol'          - lower band change tolerance for convergence dection
%                      (default: 1e-5)
%     'verbose'      - visualization of results
%                       - 0: no
%                       - 1: yes (default)
%   OUTPUTS
%      model         - Model parameters and hyperparameters
%
%   Example:
%
%        [model] = BBC_M_col_Wishart_TC(Y, X, 'obs', O, 'init', 'rand', 'maxRank', 50, 'maxiters', 100, ...
%                                'tol', 1e-6, 'verbose', 1);
%
% < Bayesian Tucker decomposition of Incomplete Tensor>
% Copyright (C) 2023  Xueke Tong
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Set parameters from input or by using defaults
randn('state',1); rand('state',1);
dimY = size(Y);
N    = ndims(Y);

ip = inputParser;
ip.addParamValue('obs', ones(dimY), @(x) (isnumeric(x) || islogical(x)));
ip.addParamValue('init', 'rand', @(x) (ismember(x,{'ml','rand'})));
ip.addParamValue('maxRank', max(dimY), @isscalar);
ip.addParamValue('maxiters', 100, @isscalar);
ip.addParamValue('tol', 1e-5, @isscalar);
ip.addParamValue('verbose', 1, @isscalar);
ip.parse(varargin{:});

O         = ip.Results.obs;
init      = ip.Results.init;
R         = ip.Results.maxRank;
maxiters  = ip.Results.maxiters;
tol       = ip.Results.tol;
verbose   = ip.Results.verbose;



%% Initialization
Y    = tensor(Y.*O);
O    = tensor(O);
nObs = sum(O(:));

a_beta0      = 1e-6;          % parameters about noise
b_beta0      = 1e-6;

dscale = sum((Y(:)-sum(Y(:))/nObs).^2)/nObs;    % for synthetic data
dscale = sqrt(dscale)/N;            
% dscale = 1;                                   % for image data    

Y = Y./dscale;                    
beta = 1e1;
%beta = 1e-1;

switch init
    case 'ml'    % Maximum likelihood
        Z = cell(N,1);
        ZSigma = cell(N,1);
        if ~isempty(find(O==0))
            Y(find(O==0)) = sum(Y(:))/nObs;
        end
        for n = 1:N
            ZSigma{n} = (repmat(1e0*eye(R), [1 1 dimY(n)]));
            
            [U, S, V] = svd(double(tenmat(Y,n)), 'econ');
            if R <= size(U,2)
                Z{n} = U(:,1:R)*(S(1:R,1:R)).^(0.5);
            else
                Z{n} = [U*(S.^(0.5)) randn(dimY(n), R-size(U,2))];
            end
        end
              
        Y = Y.*O;
    case 'rand'   % Random initialization
        Z = cell(N,1);
        ZSigma = cell(N,1);
        for n = 1:N
            Z{n} = randn(dimY(n),R);
            ZSigma{n} = repmat(eye(R), [1 1 dimY(n)]);
        end
end

inv_V_Z = cell(N,1);

% hyperparameters of column wishart prior

col_v_1 = 10;               % for synthetic data
col_v_2 = 10; 
col_v_3 = 10; 

% col_v_1 = dimY(1)/4;      % for image data 
% col_v_2 = dimY(2)/4;
% col_v_3 = dimY(3);


f1 = max([dimY(1),dimY(2),dimY(3)]);
F(1,1) = -2; F(1,2) = 1;
F(f1,f1) = -2; F(f1,f1-1) = 1;
for ff = 2:f1-1            
    F(ff,ff-1) = 1;
    F(ff,ff) = -2;
    F(ff,ff+1) = 1;
end
F1 = F(1:dimY(1),1:dimY(1));
F2 = F(1:dimY(2),1:dimY(2));
F3 = F(1:dimY(3),1:dimY(3));

col_W1 = 1e0*(F1'*F1);        % for image data
col_W2 = 1e0*(F2'*F2);
col_W3 = 1e0*(F3'*F3);

col_W1 = 1e10*eye(dimY(1));    % for synthetic data
col_W2 = 1e10*eye(dimY(2));
col_W3 = 1e10*eye(dimY(3));

% --------- E(aa') = cov(a,a) + E(a)E(a')----------------
EZZT = cell(N,1);
for n=1:N
      %  EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]))';
         EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]))' + khatrirao_fast(Z{n}',Z{n}')';
end

Fit =0;
%%%%%%%%%%%%%%%%%%%%
% adding
Y2sum = sum(Y(:).^2);
scale2 = Y2sum / nObs;
if dscale == 1
    scale = sqrt(scale2);
else
    scale = 1;
end

est_R{1} = 0;
est_R{2} = 0;
est_R{3} = 0;

%% Model learning
for it=1:maxiters,
    %% Update factor matrices 
    for n=1:N
        if it > 1 
            col_Aw{n} = col_vW{n};
        else
            col_Aw{n} = scale*diag(ones(dimY(n),1));
        end               
        % compute E(Z_{\n}^{T} Z_{\n})
        ENZZT = reshape(khatrirao_fast(EZZT{[1:n-1, n+1:N]},'r')' * double(tenmat(O,n)'), [R,R,dimY(n)]);
        % compute E(Z_{\n})
        krZ = khatrirao_fast(Z{[1:n-1, n+1:N]},'r');
        
        Pi{n} = beta * ENZZT;   
        inv_V_Z{n} = zeros(dimY(n),dimY(n),R);
        for r = 1:R
           % disp(r)
            inv_V_Z{n}(:,:,r) = eye(dimY(n))/(col_Aw{n} + diag(reshape( Pi{n}(r,r,:),1,dimY(n))) );                   
            Z{n}(:,r)         = - inv_V_Z{n}(:,:,r) * beta * (  diag(reshape(ENZZT(r,[1:r-1,r+1:R],:),(R-1),dimY(n))' * Z{n}(:,[1:r-1,r+1:R])')  - (  double( tenmat(O,n)).*double(tenmat(Y,n) ) * krZ(:,r)     )   );  % (ENZZT(r,[1:r-1,r+1:R]).*second_order_moment_lamda(r,[1:r-1,r+1:R])) * Z{n}(:,[1:r-1,r+1:R])'             
        end
        for i=1:dimY(n)
            ZSigma{n}(:,:,i) = diag(reshape(inv_V_Z{n}(i,i,:),1,R));
        end                       
        EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]) + khatrirao_fast(Z{n}',Z{n}'))';
     
        [~,V,~] = svd(Z{n});

        %% estimated rank     
        est_R{n} = sum( diag(V)/V(1,1) >  1e-2 );
        disp(diag(V)')    

    end


    
    %% Update latent tensor X
%    X = double(ktensor(Z));
    
    if it > 1 
       X_last_time = X;
    end
    
    X = zeros(dimY(1),dimY(2),dimY(3));
    for r = 1:R
        X = X + reshape(kr(Z{3}(:,r),Z{2}(:,r),Z{1}(:,r)),dimY(1),dimY(2),dimY(3));
    end
       
    err_X = X(:)*dscale - or_X(:);
    rrse1 = sqrt(sum(err_X.^2)/sum(or_X(:).^2));    
    disp(rrse1);       % err_X and rrse1 are got by or_X, which is the original tensor data. 
                       % In real world aplications, or_X is unknown, here it is used for monitoring purposes only.

        
    %% update noise beta
    %  The most time and space consuming part
    if 0 % save time but large space needed
        EX2 =  O(:)' * khatrirao_fast(EZZT,'r') * ones(R*R,1);
    else  % save space but slow
        temp1 = cell(N,1);
        EX2 =0;
        for i =1:R
            for n=1:N
                temp1{n} = EZZT{n}(:,(i-1)*R+1: i*R);
            end
            EX2 = EX2 + O(:)' * khatrirao_fast(temp1,'r')* ones(R,1);
        end
    end    
    err     = Y(:)'*Y(:) - 2*Y(:)'*X(:) + EX2;    
    a_betaN = a_beta0 + 0.5*nObs;
    b_betaN = b_beta0 + 0.5*err;
    beta    = a_betaN/b_betaN;
    Fit     = 1 - sqrt(sum(err(:)))/norm(Y(:));
    

  %  fprintf('beta_uni'); disp(beta);
    
    %% update the parameters of col \Sigma 

    col_v_A = col_v_1 + R; 
    col_v_B = col_v_2 + R; 
    col_v_C = col_v_3 + R;
       
    col_W_A = eye(dimY(1))/(eye(dimY(1))/col_W1 + Z{1}*Z{1}' + sum(inv_V_Z{1},3) );
    col_W_B = eye(dimY(2))/(eye(dimY(2))/col_W2 + Z{2}*Z{2}' + sum(inv_V_Z{2},3) );
    col_W_C = eye(dimY(3))/(eye(dimY(3))/col_W3 + Z{3}*Z{3}' + sum(inv_V_Z{3},3) );
        
    col_vW{1} = col_v_A*col_W_A;
    col_vW{2} = col_v_B*col_W_B;
    col_vW{3} = col_v_C*col_W_C;             



    %% Convergence check
    if it > 1
        err_X = ( X*dscale - X_last_time*dscale ).^2;
        LBRelChan = sum(err_X(:).^0.5)/sum( abs(X(:))*dscale );  
    else
        LBRelChan = 10^10;
    end

    if verbose
        fprintf('Iter. %d: LBRelChan = %g, Fit = %g,R1 = %g,R2 = %g, R3 = %d \n', it, LBRelChan, Fit, est_R{1}, est_R{2}, est_R{3});
    end

    if it > 5 && LBRelChan < tol
           break;
    end     
      
end

%% Prepare the results
X = X*dscale;

%% Output
model.X = X;
model.alpha = beta;
model.est_R = est_R;
model.ZSigma = ZSigma;
model.Z = Z;
model.Fit = Fit;


function y = safelog(x)
x(x<1e-300)=1e-200;
x(x>1e300)=1e300;
y=log(x);

