# LGRR: Local and Global Relation Representation for Face Forgery Detection

## Introduction:  
This is the official repository of "LGRR: Local and Global Relation Representation for Face Forgery Detection". 

Although existing DeepFake detection works achieve impressive performance in the intra-domain scenario,
they are weak in learning robust and generalizable feature
representations and fail in coping with unseen domains. To
mitigate this issue, we propose a multi-view dual-branch face
forgery detector, which is capable of extracting robust discriminative
features through two different convolutional pathways, the first
focuses on learning intrinsic local detailed patterns on content-
stripped noise map, while the second integrates spatial attention
module (SAM) and vanilla convolution for global semantic
cognition. The interactive attention module (IAM) is designed
for collaborative learning and complementary representation of
features from two streams. To improve the reliability of the
network, the training process is jointly supervised by annotations
of two scales, i.e. patch and image, to encourage local and global
consistency of feature representations.
The framework of the proposed method is displayed in the img folder.


This paper is currently under review, and we will update the paper status here in time. If you use this repository for your research, please consider citing our paper. 

This repository is currently under maintenance, if you are experiencing any problems, please open an issue.
  
## Download
- git clone https://github.com/yumiaomiao-a/LGRR.git
- cd LGRR

 
## Prerequisites:  
We recommend using the Anaconda to manage the environment.  
- conda create -n lgrr python=3.6  
- conda activate lgrr  
- conda install -c pytorch pytorch=1.7.1 torchvision=0.5.0  


## Dataset Preparation
You need to download the publicly available face forensics datasets. In this work, we conduct experiments on DeepfakeTIMIT, Celeb-DF and FaceForensics++, their official download links are as follows:
- https://www.idiap.ch/en/dataset/deepfaketimit
- https://github.com/yuezunli/celeb-deepfakeforensics
- https://github.com/ondyari/FaceForensics
