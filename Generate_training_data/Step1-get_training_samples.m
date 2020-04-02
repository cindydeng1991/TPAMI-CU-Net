clear;close all;

folder = 'Training_dataset';

size_input = 64;
size_label = 64;
stride = 33;

%% initialization
data_x = zeros(size_input, size_input, 1, 1);
data_y = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
upscale=4;

%% generate data
filepaths1 = [];filepaths2 = [];
filepaths1 = [filepaths1; dir(fullfile(folder, '*.bmp'))]; % depth
filepaths2 = [filepaths2; dir(fullfile(folder, '*.png'))]; % color



for i = 1 : 500
    
                                               
                    image_HR = im2double(imread(fullfile(folder,filepaths1(i).name)));                    
                    image_LR=imresize(image_HR,1/upscale);
                    image_LR=imresize(image_LR,upscale);
                    image_Guide=im2double(imread(fullfile(folder,filepaths2(i).name)));  
                    
                    image_Guide=  rgb2ycbcr(image_Guide);  % change RGB to YCbCr
                    image_Guide= image_Guide(:,:,1);
                   
                    [hei,wid] = size(image_HR);
                   
                    filepaths2(i).name
                    for x = 1 : stride : hei-size_input+1
                        for y = 1 :stride : wid-size_input+1
                            subim_inputx = image_LR(x : x+size_input-1, y : y+size_input-1);
                            subim_inputy = image_Guide(x : x+size_label-1, y : y+size_label-1); 
                            subim_label = image_HR(x : x+size_label-1, y : y+size_label-1); 
                            count=count+1;
                            
                            data_x(:, :, 1, count) = subim_inputx;
                            label(:, :, 1, count) = subim_label;
                            
                            data_y(:, :, 1, count) = subim_inputy;
                            
                        end
                     end
                   
      
end

order = randperm(count);
data_x = data_x(:, :, 1, order);
save('data_x.mat','data_x','-v7.3'); clear data_x
data_y = data_y(:, :, 1, order);
save('data_y.mat','data_y','-v7.3'); 
clear data_y
label = label(:, :, 1, order);  
save('label.mat','label','-v7.3');
clear label


