clear;clc;close  all;
mainDir = 'E:\Qingjia_work\NewMatlabCode\matlabcode\Motion_Detection\FSE_motion_simulation\cyl仿真\数据';
sub1dir  = dir(mainDir);
savePath1 = ('E:\Qingjia_work\NewMatlabCode\matlabcode\Motion_Detection\FSE_motion_simulation\cyl仿真\数据\SimulateMotionData\');
savePath2 = ('E:\Qingjia_work\NewMatlabCode\matlabcode\Motion_Detection\FSE_motion_simulation\cyl仿真\数据\GroundTruth\');


x=256;
y=1;
z=x/y;
% b=[-112 -80 -48 -16 16 48 80 112 -111 -79 -47 -15 17 49 81 113 -110 -78 -46 -14  18 50 82 114 -109 -77 -45 -13 19 51 83 115 -108 -76 -44 -12 20 52 84 116 -107 -75 -43 -11 21 53 85 117 -106 -74 -42 -10 22 54 86 118 -105 -73 -41 -9 23 55 87 119 -104 -72 -40 -8 24 56 88 120 -103 -71 -39 -7 25 57 89 121 -102 -70 -38 -6 26 58 90 122 -101 -69 -37 -5 27 59 91 123 -100 -68 -36 -4 28 60 92 124 -99 -67 -35 -3 29 61 93 125 -98 -66 -34 -2 30 62 94 126 -97 -65 -33 -1 31 63 95 127 -96 -64 -32 0 32 64 96 -128 -95 -63 -31 1 33 65 97 -127 -94 -62 -30 2 34 66 98 -126 -93 -61 -29 3 35 67 99 -125 -92 -60 -28 4 36 68 100 -124 -91 -59 -27 5 37 69 101 -123 -90 -58 -26 6 38 70 102 -122 -89 -57 -25 7 39 71 103 -121 -88 -56 -24 8 40 72 104 -120 -87 -55 -23 9 41 73 105 -119 -86 -54 -22 10 42 74 106 -118 -85 -53 -21 11 43 75 107 -117 -84 -52 -20 12 44 76 108 -116 -83 -51 -19 13 45 77 109 -115 -82 -50 -18 14 46 78 110 -114 -81 -49 -17 15 47 79 111 -113];

b=-x/2:x/2-1;
RotationRange=4;
MoveRange=4;
MoveRangeZ=2;


for num = 250

    test11 = load('E:\Qingjia_work\NewMatlabCode\matlabcode\Motion_Detection\FSE_motion_simulation\小鼠数据\RAREImage434.mat');
%     mriVolume = test11.Imagdata;



            FilePath='E:\Qingjia_work\NewMatlabCode\matlabcode\Motion_Detection\FSE_motion_simulation\9\9\';
        %     DataObj=RawDataObject(FilePath);
            DataObj=RawDataObject(FilePath);
            SpatialReso=DataObj.Method.PVM_SpatResol;
            Thickness=DataObj.Method.PVM_SliceThick;
            mriVolumeRaw=double(squeeze(test11.Imagdata));
            mriVolumeRaw=cat(3, zeros(size(mriVolumeRaw(:,:,1,1))), mriVolumeRaw, zeros(size(mriVolumeRaw(:,:,1,1))));
    
            ZeroPadNum=round(size(mriVolumeRaw,3)*Thickness/min(SpatialReso)/4);
            kspacedata=FFTXSpace2KSpace(mriVolumeRaw,3);
            kspacedataZP=zeros(size(mriVolumeRaw,1),size(mriVolumeRaw,2),ZeroPadNum);
            sz=size(kspacedataZP,3);
            mz=size(mriVolumeRaw,3);
            kspacedataZP(:,:,sz/2+1-mz/2 : sz/2+mz/2)=kspacedata;
            mriVolume=abs(FFTXSpace2KSpace(kspacedataZP,3));
    
    
    
    %         GT_File=[savePath2,filesep,'no_blur_gt' num2str(num) '_' num2str(count) '.mat'];
    %         save(GT_File,'mriVolume');
    %
    

    %     b=[];
    %     a = linspace((-x/2+x/y/2),(-x/2+x/y/2)+z*(y-1),y);
    %     nums = zeros(1,256);
    %     for i=1:z
    %         b = cat(2,b,a);
    %         a = linspace((-x/2+x/y/2)+i,(-x/2+x/y/2)+i+z*7,8);
    %     end
    for count=1:2
        GT_File=[savePath2,filesep,'random_shot_8_gt' num2str(num) '_' num2str(count) '.mat'];
        
        ReshapeGTData=imresize3D(mriVolume,size(squeeze(test11.Imagdata)));
        ReshapeGTData=ReshapeGTData(:,:,3:end-3);
        save(GT_File,'mriVolume');
%         tic

        %
        %                 theta1=1*deg2rad(periodicCurve(z,ky0,0.2,pi/4,2.4));
        %                 theta2=1*deg2rad(periodicCurve(z,ky0,0.2,pi/4,2.4));
        %                 theta3=1*deg2rad(periodicCurve(z,ky0,0.2,pi/4,2.4));
        %
        %                 vec1=1*(periodicCurve(z,ky0,0.2,pi/4,2.5));
        %                 vec2=1*(periodicCurve(z,ky0,0.2,pi/4,2.5));
        %                 vec3=1*(periodicCurve(z,ky0,0.2,pi/4,2.5));
        %                                 theta1=1*deg2rad(filter(1,[1 rand(1,1)],rand(1,z))*RotationRange-RotationRange/2).*cat(2,ones(1,7),zeros(1,2),ones(1,7));
        %                 theta2=1*deg2rad(filter(1,[1 rand(1,1)],rand(1,z))*RotationRange-RotationRange/2).*cat(2,ones(1,7),zeros(1,2),ones(1,7));
        %                 theta3=1*deg2rad(filter(1,[1 rand(1,1)],rand(1,z))*RotationRange-RotationRange/2).*cat(2,ones(1,7),zeros(1,2),ones(1,7));
        %
        %                 vec1=1*(filter(1,[1 rand(1,1)],rand(1,z))*MoveRange-MoveRange/2).*cat(2,ones(1,7),zeros(1,2),ones(1,7));
        %                 vec2=1*(filter(1,[1 rand(1,1)],rand(1,z)  )*MoveRange-MoveRange/2).*cat(2,ones(1,7),zeros(1,2),ones(1,7));
        %                 vec3=1*(filter(1,[1 rand(1,1)],rand(1,z))*MoveRangeZ-MoveRangeZ/2).*cat(2,ones(1,7),zeros(1,2),ones(1,7));
        %                 theta1=1*deg2rad(filter(1,[1 rand(1,1)],rand(1,z))*RotationRange-RotationRange/2).*cat(2,ones(1,8),zeros(1,16),ones(1,8));
        %                 theta2=1*deg2rad(filter(1,[1 rand(1,1)],rand(1,z))*RotationRange-RotationRange/2).*cat(2,ones(1,8),zeros(1,16),ones(1,8));
        %                 theta3=1*deg2rad(filter(1,[1 rand(1,1)],rand(1,z))*RotationRange-RotationRange/2).*cat(2,ones(1,8),zeros(1,16),ones(1,8));
        %
        %                 vec1=1*(filter(1,[1 rand(1,1)],rand(1,z))*MoveRange-MoveRange/2).*cat(2,ones(1,8),zeros(1,16),ones(1,8));
        %                 vec2=1*(filter(1,[1 rand(1,1)],rand(1,z))*MoveRange-MoveRange/2).*cat(2,ones(1,8),zeros(1,16),ones(1,8));
        %                 vec3=1*(filter(1,[1 rand(1,1)],rand(1,z))*MoveRangeZ-MoveRangeZ/2).*cat(2,ones(1,8),zeros(1,16),ones(1,8));
        
        theta1=1*deg2rad(rand(1,z)*RotationRange-RotationRange/2);
        theta2=1*deg2rad(rand(1,z)*RotationRange-RotationRange/2);
        theta3=1*deg2rad(rand(1,z)*RotationRange-RotationRange/2);
        
        vec1=1*(rand(1,z)*MoveRange-MoveRange/2);
        vec2=1*(rand(1,z)*MoveRange-MoveRange/2);
        vec3=1*(rand(1,z)*MoveRangeZ-MoveRangeZ/2);
        
        %         TestMotionData=zeros([size(mriVolume) z]);
        v_ksp3dForAll=zeros([size(mriVolume) z]);
        for idx = 1:z
            
            v = threedrotation_v3(mriVolume,theta1(1,idx),theta2(1,idx),theta3(1,idx));   
            v = imtranslate(v,[vec1(1,idx),vec2(1,idx),vec3(1,idx)],'OutputView','same');
            
            %             TestMotionData(:,:,:,idx)=v;
            
            %  kspaceIndex=b((idx-1)*y+1:idx*y);
            v_ksp3dForAll(:,:,:,idx)=FFTXSpace2KSpace(FFTXSpace2KSpace(v,1),2);
            disp(idx)
            %             v_ksp3d0(:,kspaceIndex+x/2+1,:)=v_ksp3dForAll(:,kspaceIndex+x/2+1,:,idx);
            
        end
        v_ksp3d0=zeros(size(mriVolume));
        for ix=1:z
            kspaceIndex=b((ix-1)*y+1:ix*y);
            v_ksp3d0(:,kspaceIndex+x/2+1,:)=v_ksp3dForAll(:,kspaceIndex+x/2+1,:,ix);
        end
        
        motionData=(FFTKSpace2XSpace(FFTKSpace2XSpace(v_ksp3d0,1),2));
        ReshapeMotionData=imresize3D(motionData,size(squeeze(test11.Imagdata)));
        ReshapeMotionData=ReshapeMotionData(:,:,3:end-3);
        
        MotionFile=[savePath1,filesep,'random_shot_8_motion' num2str(num) '_'  num2str(count) '.mat'];
        save(MotionFile,'ReshapeMotionData','theta1','theta2','theta3','vec1','vec2','vec3');
        disp('count')
    end
end


