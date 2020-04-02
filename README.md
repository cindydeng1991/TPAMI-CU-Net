# TPAMI-CU-Net
This repository is Tensorflow code for our paper entitled "Deep Convolutional Neural Network for Multi-modal Image Restoration and Fusion
" . [[Paper Download]](https://drive.google.com/file/d/1Nt4VOWNb8LxEt2TXd9OI0nNsFQSeCFeT/view?usp=sharing) [[Project Website]](https://github.com/cindydeng1991/TPAMI-CU-Net/new/master?readme=1)

![](images/framework.PNG)
Network Architecture of the proposed CU-Net. For MIR related tasks, the ﬁnal reconstruction (Point 4) is composed of the common reconstruction (Point 1) and the unique reconstruction (Point 2). For MIF related tasks, the ﬁnal reconstruction is composed of the common reconstruction (Point 1) and the two unique reconstructions (Point 2 and Point 3). 

If you find our work useful in your research or publications, please consider citing:

>@inproceedings{Deng2019deep,  
>    author = {Deng, Xin and Dragotti, Pier Luigi},  
>    title = {Deep Convolutional Neural Network for Multi-modal Image Restoration and Fusion},  
>    booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},  
>    year= {2020}  
>}

## Requirement
- Tensorflow >=1.1.0 
- Matlab>=2017a

## Train
1. Download training dataset.  The RGB/Depth training dataset is from [DPDN](https://github.com/griegler/primal-dual-networks), and can be downloaded from [Googledrive](https://drive.google.com/file/d/14fEIIg7tTxAxz61QsFQqz1ADVDkJG4xW/view?usp=sharing).
The RGB/Multi-spectral dataset is from Columbia multi-spectral [database](https://www.cs.columbia.edu/CAVE/databases/multispectral/).
The Flash/Non-flash dataset is from [Aksoy et al. ECCV2018](http://yaksoy.github.io/flashambient/). The multi-exposure dataset is from [SICE](https://github.com/csjcai/SICE).
2. Generate the training samples following Step 1 to Step 3 in the Generate_training_data file.
3. Put the training data in the root directory, and run the command  
"python training_MIR_task"  
for multi-modal image restoration tasks, including RGB guided depth image SR, RGB guided MS image SR, Flash gudied non-flash image denoising, etc.     
Run the command    
"python training_MIF_task"   
for multi-modal image fusion tasks, including multi-exposure image fusion, multi-focus image fusion, etc.
4. The training model will be saved and later used for testing.

## Test
1. Generate the testing data using the same way the training data.
2. For the MIR related tasks, run the command  
"python testing_MIR_task"
3. For the MIF related tasks, run the command  
"python testing_MIF_task"

The testing datasets and our results in the paper can be downloaded from [Googledrive](https://drive.google.com/file/d/1eYaULXvqNqzHZlK5jVcsz7IvtfmhwrQw/view?usp=sharing).

