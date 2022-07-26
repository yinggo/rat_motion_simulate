% GradMC 2
% Blind Retrospective Motion Correction of MR Images
% 2D demo Matlab/Octave script
% By Alexander Loktyushin and Hannes Nickisch
% Max Planck Institute for Intelligent Systems, 2013

clear all, close all, clc

OCT = exist('OCTAVE_VERSION') ~= 0;      % check if we run Matlab or Octave
me = mfilename;                                       % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));   % where am I located
if OCT
  addpath([mydir,'code/'])
  addpath([mydir,'code/@matFastFFTmotion/private'])
else
  addpath code/
end

nMVM = 120;                                 % Number of function evaluations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%   CHOOSE DATA   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Image name ---
% object = 'rigid_simulated.mat';                     % Simulated sine-form motion
% object = 'rigid_FLASH_monkey.mat';            % real data FLASH sequence, monkey
% object = 'rigid_TSE_monkey.mat';                % real data TSE sequence, monkey
object = 'gctest_20.mat';                  % real data TSE sequence, human



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rpath = 'data/';                                         % Path to raw data

slicenum = 17;
imag = load([rpath object]);
rawImag=imag.motionData;
% rawImag = permute(rawImag,[2,1,3]);
rawImag=rawImag(:,:,slicenum);
kspace=FFTXSpace2KSpace(FFTXSpace2KSpace(rawImag,1),2);

% 
% RawDataPath='E:\Qingjia_work\NewMatlabCode\matlabcode\Motion_Detection\gradmc2\GradMC2\data\RatMotionInvivoData\QingjiaMotionTest_rat10814.3E2\4';
% ImagFile=[RawDataPath '\pdata\1\']
% imageObj = ImageDataObject(ImagFile);
% ImageData=imageObj.data;
% rawObj = RawDataObject(RawDataPath);
% PVM_EncSteps1=rawObj.Method.PVM_EncSteps1;
% PVM_EncSteps1=PVM_EncSteps1+70;
% rawObjComplex=squeeze(rawObj.data{1});
% 
% rawObjComplex1=reshape(rawObjComplex,[4,256,4,2,24,196/4]);
% rawObjComplex1=permute(rawObjComplex1,[1,2,4,3,6,5]);
% rawObjComplexAll=zeros([4,256,2,196,24]);
% for i=1:196/4
%     rawObjComplexAll(:,:,:,PVM_EncSteps1((i-1)*4+1:i*4),:)=squeeze(rawObjComplex1(:,:,:,:,i,:));
% end
% rawObjComplexAll=permute(rawObjComplexAll,[2,4,5,3,1]);
% Images = FFTXSpace2KSpace(FFTXSpace2KSpace(rawObjComplexAll,1),2); 
% figure;imshow3Dfull(abs(Images(:,:,:,1,4)))
% raw=rawObjComplexAll(:,:,6,1,4);

% TempSRdata0=squeeze(ImageData);
% figure(11);imshow3Dfull(squeeze(TempSRdata0(:,:,1,:)))
% TempSRdata0=TempSRdata0(:,:,1,12);
% imag=TempSRdata0;
% raw = FFTXSpace2KSpace(FFTXSpace2KSpace(imag,1),2);                      % Load data


raw = kspace/max(abs(kspace(:)));                                     % Normalize

sz = size(raw); F = matFFTN(sz);
fprintf('Volume size: sx = %d, sy = %d\n',sz(1),sz(2));

y = reshape(([F']*raw),sz);                       % Image in spatial domain
n = prod(sz); ntra = numel(sz); nrot = ntra*(ntra-1)/2; T = prod(sz(2:end));
brad = pi/180;                                            % Back to radians

% Initialize motion parameters
tra = zeros(ntra,T); rot = zeros(nrot,T); init = [tra; rot/brad];

figure(1),                                            % Show degraded image
subplot(1,2,1), imagesc(rot90(abs(squeeze(y(:,:))),1)); title('Degraded image'); colormap gray; axis equal; axis tight;

% Set metric - entropy
ep = 1e-8;                                          % stabilisation constant
v = @(u) sqrt(u.*conj(u)+ep^2)/sqrt(u'*u+ep^2);
psi = @(u) -v(u)'*log(v(u));
p = @(u) -(1+log(v(u)));                                % derivative dpsi/dv
dpsi = @(u) u.*v(u).*p(u)./(u.*conj(u)+ep^2) - u*(v(u)'*p(u)) / (u'*u+ep^2);
  
% Create gradient operators
gradx = [1 -1]; grady = [1 -1]';
Gx = matFConv2(gradx,sz,'same'); Gy = matFConv2(grady,sz,'same');

% Gradients + Back to spatial domain
C = cell(2,1); C{1} = [matFFTN(sz)']*(Gx); C{2} = [matFFTN(sz)']*(Gy);

% build arguments
args = {sz,psi,dpsi,y,C};

% Optimize
[z,phi] = minimize(init(:),'phi_matFFTmotion',-nMVM,args{:});

z = reshape(z,[ntra+nrot T]); t_z = z(1:ntra,:); r_z = z(ntra+(1:nrot),:);

% Invert the motion
M = matFastFFTmotion(sz,t_z,brad*r_z,[],'cubicmex');
corr_im = (reshape([matFFTN(sz)']*M*y,sz));

% Plot recovered motion trajectories
figure(2), subplot(3,1,1), plot(t_z(1,:)); title('translation in frequency encoding direction');
subplot(3,1,2), plot(t_z(2,:)); title('translation in phase encoding direction');
subplot(3,1,3), plot(r_z(1,:)); title('in-plane rotation');

figure(1),                                       % Show reconstructed image
subplot(1,2,2), imagesc(rot90(abs(squeeze(corr_im(:,:))),1)); title('Reconstruction'); colormap gray; axis equal; axis tight;

out = rot90(abs(corr_im));
out = (out-min(out(:)))/(max(out(:))-min(out(:)));
% gtmat = load('C:\Users\BCX\Desktop\亚雷文章回复\新建文件夹\gt.mat');
% gtmat = load('C:\Users\BCX\Desktop\亚雷文章回复\新建文件夹\mild_motion_111\gt.mat');
gtmat = load('C:\Users\BCX\Desktop\亚雷文章回复\新建文件夹\结果图\shotnumber\gt.mat');
gt = gtmat.gt;
gt = permute(gt,[2,1,3]);
gtslice = gt(:,:,slicenum);
gtslice = (gtslice-min(gtslice(:)))/(max(gtslice(:))-min(gtslice(:)));
figure;imagesc(gtslice); colormap gray;axis off;axis equal;
out=fliplr(out(:,:));
figure;imagesc(out); colormap gray;axis off;axis equal;
figure;imagesc(abs(out - gtslice));colormap(jet);axis off;caxis([0,0.6]);axis equal;


