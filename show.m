clear all, close all, clc

gtmat = load('gt.mat');
motionmat = load('motion.mat');
cormat = load('corrected.mat');

gt = gtmat.gt;
motion = motionmat.motion;
corrected = cormat.corrected;

gt = permute(gt,[2,3,1]);
motion = permute(motion,[2,3,1]);
corrected = permute(corrected,[2,3,1]);

corrected_bn = zeros(size(corrected));
for j = 1:size(corrected,3)
    tem_our =squeeze(corrected(:,:,j));

    tem_our_max = max(tem_our(:));
    tem_our_min = min(tem_our(:));

    corrected_bn(:,:,j) = (tem_our - tem_our_min)/(tem_our_max - tem_our_min);
end

figure;imshow3Dfull(cat(2,motion,corrected_bn,gt));

% figure;imshow3Dfull(gt);
% figure;imshow3Dfull(motion);
% figure;imshow3Dfull(corrected_bn);

num = 626;
save_motion = motion(:,:,num);
save_gt = gt(:,:,num);
save_our = corrected_bn(:,:,num);

K=wiener2(save_gt,[3 3]);
figure;imagesc(K);colormap(gray);axis off;axis equal;

figure;imagesc(save_motion);colormap(gray);axis off;axis equal;
figure;imagesc(save_gt);colormap(gray);axis off;axis equal;
figure;imagesc(save_our);colormap(gray);axis off;axis equal;