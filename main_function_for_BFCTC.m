
%%  a demo for the paper:
%%      X. Tong, L. Cheng, and Y.-C. Wu, “Bayesian Tensor Tucker Completion
%%      With A Flexible Core,” in IEEE Trans. Signal Process., vol. 71, pp.
%%      4077-4091, 2023.
  
%%  code by Xueke Tong, 2023
 
close all;
clc;
clear;
addpath(genpath(pwd));

mode            = 1;    % mode = 1 denote to process the synthetic data and mode = 2 is to process the image data    
number_of_trial = 2 ;   

for num_of_noise = 1:number_of_trial
    
    fprintf('the %g-th iter \n',num_of_noise);
    randn('state',num_of_noise); rand('state',num_of_noise);
    
    if  mode == 1
        %% Generate a low-rank tensor
        DIM = [30,30,30];     % Dimensions of data
        R1 = 10;     % R = 40;
        R2 = 10;
        R3 = 10;  
                         
        VV{3} = 1*eye(R3) + diag(randperm(R3));
        VV{1} = 1*eye(R1)  + diag(randperm(R1)) ;      
        VV{2} = 1*eye(R2) + diag(randperm(R2)) ;

        U{1}  = orth(randn(DIM(1),R1));        
        U{2}  = orth(randn(DIM(2),R2));
        U{3}  = orth(randn(DIM(3),R3));
         
        core  = ttm(tensor(randn(R1,R2,R3)),VV);
        X     = double(ttensor(core,U)); 
        X     = (prod(DIM)/sqrt(sum(X(:).^2)))*X;                          
        
    else        
        %% Generate an image tensor 
        ima = double(imread('TestImages/peppers.bmp'));
        %  ima = double(imread('TestImages/lena.bmp'));
        %  ima = double(imread('TestImages/barbara.bmp'));
        %  ima = double(imread('TestImages/house.bmp'));
        %  ima = double(imread('TestImages/airplane.bmp'));
        %  ima = double(imread('TestImages/sailboat.bmp'));              
        %  ima = double(imread('TestImages/facade.bmp'));
        %  ima = double(imread('TestImages/baboon.bmp'));                                    
        X = ima;

        %% Generate an MRI tensor
        %  fid=fopen('/Users/tongxueke/Documents/BBC/data/data/t1_icbm_normal_1mm_pn0_rf0.rawb');
        %  temp = fread(fid, 181 * 217 * 181);
        %  images = reshape(temp, 181 * 217, 181);  
        %  L = 181;
        %  for num = 1:L
        %      image(:,:,num) = reshape(images(:, num), 181, 217);
        %  end
        %  X = image(:,:,11:16);

    end       
    %% Random missing values
    ObsRatio  = 0.3;    % Observation rate: [0 ~ 1] 
    DIM       = size(X);
    [~,Omega] = sort(rand(1,prod(DIM)));
    Omega     = Omega(1:round(ObsRatio*prod(DIM)));
    O         = zeros(DIM); 
    O(Omega)  = 1;
    nObs      = sum(O(:));
    disp(nObs/numel(X))
    
    %% Add noise
    SNR       = 10;                     % Noise levels
    sigma2    = var(X(:))*(1/(10^(SNR/10)));
    GN        = sqrt(sigma2)*randn(DIM);        %  mean(GN(:).^2)
    
    %% Generate observation tensor Y
    Y         = X + GN;
    Y         = O.*Y;

    %% settings for the data
    iter      = 100;  


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% RUN BFCTC-W

    fprintf('------BFCTD-W---------- \n');
    
    ts1(num_of_noise)   = tic;        
    [model1]            = BBC_M_col_Wishart_TC(Y, X, 'obs', O, 'init', 'ml', 'maxRank', 100, 'tol', 1e-6, 'maxiters', iter, 'verbose', 1);
    t_total(num_of_noise) = toc(ts1(num_of_noise));
    
    est_R1 = model1.est_R{1};   % estimated rank
    est_R2 = model1.est_R{2};
    est_R3 = model1.est_R{3};
    
    est_X1 = double(model1.X);
    err1   = est_X1(:) - X(:);
    rrse1  = sqrt(sum(err1.^2)/sum(X(:).^2)); 
    fprintf('the rrse of BFCTD-W \n'),disp(rrse1);  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    RR(num_of_noise,1) = est_R1;
    RR(num_of_noise,2) = est_R2;
    RR(num_of_noise,3) = est_R3;
    
    RSE(num_of_noise)                   = rrse1;
    recovered_image(:,:,:,num_of_noise) = est_X1;

end
       
fprintf('the averaged rank \n'), disp(mean(RR))

fprintf('average RSE \n'), disp(mean(RSE))
fprintf('the RSE std  \n'), disp(std(RSE))

fprintf('the averaged run time,.. \n'), disp(mean(t_total))
fprintf('the run time std \n'), disp(std(t_total))

save('BFCTC.mat','RSE','RR','t_total','recovered_image');



