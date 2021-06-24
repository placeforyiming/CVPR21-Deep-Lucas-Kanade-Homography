## Abstract

 We develop a deep learning pipeline to accurately align challenging multimodality images. The method is based on traditional Lucas-Kanade algorithm with feature maps extracted by deep neural networks. We test our method on three datasets, MSCOCO with regular images, Google Earth with cross season images, Google Map and Google Satellite with multimodality images.
 


![Alt Text](https://github.com/ProjectTempForReview/Deep-Homography-via-Lifting-Lucas-Kanade-Method/blob/master/demo.gif)

## Dataset
The Google Map and Google Earth data used for both training and tesing can be downloaded here: 
https://drive.google.com/file/d/1-voX6dJtIb1Dbq9_m2Ed9xvPhCjM_FLo/view?usp=sharing
After downloading, extract it. Move ./Dataset/GoogleEarth/* to /CVPR21-Deep-Lucas-Kanade-Homography/Dataset/GoogleEarth/ and ./Dataset/GoogleMap/* to /CVPR21-Deep-Lucas-Kanade-Homography/Dataset/GoogleMap/ 


For MSCOCO, we are using 2014 train and val, which is here:
http://cocodataset.org/#download.

Download 2014 Train images and 2014 Val images, extract them under /CVPR21-Deep-Lucas-Kanade-Homography/Dataset/MSCOCO. Generating the training and testing sample by running:

python3 generate_training_sample.py

python3 generate_testing_sample.py

We also provide the txt file about all 6k images sampled for the tesing used in this paper.  


## How to run? 

Download data into Dataset folder. 
Pull and activate a tensorflow 2 docker container: tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
Install all dependency packages: bash install_dependency.sh

1. To train the single stage regression network for homography, run /model_ours/train_regression_original.py

2. To train the multi-stage regression network for homography, run /model_ours/train_regression_first.py, /model_ours/train_regression_second.py and /model_ours/train_regression_third.py in order.

3. To train our DLKFM for homography, run /model_ours/train_level_one.py, /model_ours/train_level_two.py and /model_ours/train_level_three.py in order.

## Explore more?
Dowanload the pretrained checkpoint from : https://drive.google.com/file/d/1karVQMlh3FOMMa2cTYS35GdehdAt7lW5/view?usp=sharing
Extract and put them into ./model_ours/checkpoints/

Run jupyter.sh to open the jupyter notebook in docker.

Run jupyter notebook in ./model_ours to see the feature map.

You also can explore the traditional SIFT+RANSAC on three datasets by opening /model_SIFT/SIFAT_ransac.ipynb or try our implementated CLKN in the folder model_CLKN.

Our Lucas-Kanade layer is implemented with Tensorflow 2. It can serve as part of the trainable network or can be used independently like the traditional Hierarchy  Lucas-Kanade. If you find part of our code is useful for your project, please cite our paper below:



## Publication ##
Please cite the papers if you use this code:

```
@inproceedings{zhao2021deep,
  title={Deep Lucas-Kanade Homography for Multimodal Image Alignment},
  author={Zhao, Yiming and Huang, Xinming and Zhang, Ziming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15950--15959},
  year={2021}
}

```
