# EAST: An Efficient and Accurate Scene Text Detector
### Description:
This version will be updated soon, please pay attention to this work.
The motivation of this version is to build a easy-training model. 
This version can automatically update best_model by comparing current hmean and the former.
At the same time, we can see evaluation info about every sample easily.

+ 1.train
+ 2.predict 
+ 3.compress
+ 4.compute Hmean(if Hmean is higher than before, update best_weight.pkl)
+ 5.visualization(blue, green, red)
+ 6.multi-scale test (update soon)
    multi-scale vis. (vis with score, scales)

### Thanks
The version is ported from [argman/EAST](https://github.com/argman/EAST), from Tensorflow to Pytorch

### Check On Website
If you have no confidence of the result of our program, you could use submit.zip to submit on [website](http://rrc.cvc.uab.es/?ch=2&com=mymethods&task=1),then you can see result of every image.

### Performance
+ right -- green || wrong -- red || miss -- blue
![visualization](https://github.com/songdejia/east-pytorch/blob/master/screenshots/vis01.png)
![visualization](https://github.com/songdejia/east-pytorch/blob/master/screenshots/vis02.png)



+ recall/precision/hmean for every test image
![hmean](https://github.com/songdejia/east-pytorch/blob/master/screenshots/hmean.png)

### Introduction
This is a pytorch re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
The features are summarized blow:

+ Only **RBOX** part is implemented.
+ A fast Locality-Aware NMS in C++ provided by the paper's author.(g++/gcc version 6.0 + will be ok)
+ Evalution see [here](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_samples&task=1&m=29855&gtv=1) for the detailed results.
+ Differences from original paper
	+ Use ResNet-50 rather than PVANET
	+ Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
	+ Use linear learning rate decay rather than staged learning rate decay
	
Thanks for the author's ([@zxytim](https://github.com/zxytim)) help!
Please cite his [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Prepare dataset/pretrain](#dataset)
4. [Test](#train)
5. [Train](#test)
6. [Examples](#examples)


### Installation
1. Any version of pytorch version > 0.4.0 should be ok.

### Download
1. Pretrained model is not provided temporarily. Web site is updating now, please continue to pay attention 

### Prepare dataset/pretrain weight
[1]. dataset(you need to prepare for dataset for train and test)
suggestions: you could do a soft-link to root_to_this_program/dataset/train/img/*.jpg
+ -- train  ./dataset/train/img/img_###.jpg 
	    ./dataset/train/gt/img_###.txt (you need to change name)
+ -- test   ./data/test/img_###.jpg (img only)
+ -- gt.zip ./result/gt.zip(ICDAR15 gt.zip is avaliable on [website](http://rrc.cvc.uab.es/?ch=2&com=mymethods&task=1)

** Note: you can download dataset here
+ -- [ICDAR15](http://rrc.cvc.uab.es/?ch=4&com=downloads)
+ -- [ICDAR13](http://rrc.cvc.uab.es/?ch=2&com=downloads)

[2]. pretrained  
+ In config.py set resume True and set checkpoint path/to/weight/file
+ I will provide pretrianed weight soon

[3]. check GPUs and CPUs 
you can use following to check aviliable gpu, this is for train
```
watch -n 0.1 nvidia-smi
```
then, you will see 2,3 is avaliable, modify config.py
gpu_ids = [0,1], gpu = 2, and modify run.sh - CUDA_VISIBLE_DEVICES=2,3



### Train
If you want to train the model, you should provide the dataset path in config.py and run

```
sh run.py
```
** Note: you should modify run.sh to specify your gpu id

If you have more than one gpu, you can pass gpu ids to gpu_list(like gpu_list=0,1,2,3) in config.py

** Note: you should change the gt text file of icdar2015's filename to img_\*.txt instead of gt_img_\*.txt(or you can change the code in icdar.py), and some extra characters should be removed from the file.
See the examples in training_samples/**

### Test
By default, we set train-eval process into integer.
If you want to use eval independently, you can do it by yourself. Any question can contact me.


### Examples
Here are some test examples on icdar2015, enjoy the beautiful text boxes!
![image_1](demo_images/img_2.jpg)
![image_2](demo_images/img_10.jpg)
![image_3](demo_images/img_14.jpg)
![image_4](demo_images/img_26.jpg)
![image_5](demo_images/img_75.jpg)


