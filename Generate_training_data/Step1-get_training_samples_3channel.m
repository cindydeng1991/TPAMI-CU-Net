clear;close all;


folder2 = 'flash';
folder_gt='nonflash_target';

size_input = 64;
size_label = 64;
stride = 92;

%% initialization
data_x = zeros(size_input, size_input, 3, 1);
data_y = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);

count = 0;

noiseSigma=50;
%% generate data
filepaths2 = [];filepaths3 = [];filepathsgt=[]; 
filepaths2 = [filepaths2; dir(fullfile(folder2, '*.png'))]; 
filepathsgt = [filepathsgt; dir(fullfile(folder_gt, '*.png'))]; 

for i = 1 : length(filepaths2)
                                                   
                    image_HR = im2double(imread(fullfile(folder_gt,filepathsgt(i).name)));                    
                     blur_1 = single(image_HR + noiseSigma/255*randn(size(image_HR)));
                    blur_2=im2double(imread(fullfile(folder2,filepaths2(i).name)));                                   
                      
                    [hei,wid,~] = size(image_HR);
                    
                    filepaths2(i).name
                    for x = 1 : stride : hei-size_input+1
                        for y = 1 :stride : wid-size_input+1
                            subim_inputx = blur_1(x : x+size_input-1, y : y+size_input-1,:);
                            subim_inputy = blur_2(x : x+size_label-1, y : y+size_label-1,:);
                            
                            subim_label = image_HR(x : x+size_label-1, y : y+size_label-1,:); 
                            count=count+1;
                            data_x(:, :, :, count) = subim_inputx;
                           data_y(:, :, :, count) = subim_inputy;
                           
                            label(:, :, :, count) = subim_label;
                        end
                    end
                     clear image_HR
                     clear blur_1 blur_2  subim_inputx subim_inputy subim_label
                              
end

order = randperm(count);
data_x = data_x(:, :, :, order);
save('data_x.mat','data_x','-v7.3');
clear data_x
data_y = data_y(:, :, :, order);
save('data_y.mat','data_y','-v7.3');
clear data_y
label = label(:, :, :, order); 
save('label.mat','label','-v7.3');
clear label


