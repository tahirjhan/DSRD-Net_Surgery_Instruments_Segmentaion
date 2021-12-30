clc
close all
clear all

% Load trained model
load 'trained_sgdm_2'

% Path to test data
dir_path = '_';

% File to save results 
file_name = sprintf('Final_res%d.txt',1);
out = fullfile(dir_path,file_name);
fileID = fopen(out,'w');
fprintf(fileID,'\r\n');

% Test images path
test_image_read_from='_';
result_saved_in='_';
GT_read_from='_'; 

% Read data and get results
aji_individual = []; 

for ff=1:75
    ff
    aji_individual = [];

    % Read single test image
    image=imread(strcat(test_image_read_from,num2str(ff),'.jpg'));
    [r,c,n] = size(image);
    
    im = imresize(image,[512 640],'nearest');
   
    timeR=0;
    tic;
    C = semanticseg(im,net);
    k=C=='instrument';
    timeR = toc;
    
    k1=k;
    
    mask=double(k);
    mask = imresize(mask,[r c],'nearest');

    % Read ground truth
    GT=imread(strcat(GT_read_from,num2str(ff),'.jpg'));  
    GT=double(GT);
    GT=GT(: , : ,1);
    GT = uint8(255 * mat2gray(GT>0));

    imR=image(:,:,1);
    imG=image(:,:,2);
    imB=image(:,:,3);

    [x1,y1]=size(imR);
         
    err=0;
    tp=0;
    fp=0;
    fn=0;
    tn=0;
              
    for jj=1:x1
        for kk=1:y1
            if GT(jj,kk)==255 && mask(jj,kk)==1
                 imR(jj,kk)=0;
                  imG(jj,kk)=0;
                   imB(jj,kk)=255;
                   tp=tp+1;
            elseif GT(jj,kk)==0 && mask(jj,kk)==0
%                imR(jj,kk)=255;
%                imG(jj,kk)=0;
%                imB(jj,kk)=0;
%                err=err+1;
               tn=tn+1;
           elseif GT(jj,kk)==255 && mask(jj,kk)==0
               imR(jj,kk)=255;
               imG(jj,kk)=0;
               imB(jj,kk)=0;
               err=err+1;
               fn=fn+1;
            elseif GT(jj,kk)==0 && mask(jj,kk)==1  %&& k1(jj,kk)==0 && imu(jj,kk)==0 && imd(jj,kk)==0
              imR(jj,kk)=0;
              imG(jj,kk)=255;
              imB(jj,kk)=0;
              err=err+1;
              fp=fp+1;
            end
     
        end
   end

   im1=cat(3,imR,imG,imB);
  
   P = tp/(tp+fp);
   R = tp/(tp+fn);
   F2 = (5*P*R)/((4*P)+R);   
   Jaccard= tp/(tp+fp+fn);
   Dice= 2*tp/((2*tp)+fp+fn);
   
   Accuracy = (tp+tn)/(tp+fp+tn+fn);

   ImgStoreName = sprintf('%dR.bmp',ff);
   save_to=fullfile(result_saved_in,ImgStoreName);
   imwrite(im1,save_to)

   fprintf(fileID,'%s\t%s\r\n',num2str(ff),num2str(Jaccard));
end



