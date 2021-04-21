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


## Code and checkpoints

We submit code with paper for review. If the paper can be accepted, we will open source code for both our model and baseline methods with checkpoints here.

## How to run? 

Download data into Dataset folder.

Put checkpoints into ./model_ours/checkpoints/

Run jupyter notebook in ./model_ours to play the model

For training:
 
 Edite several top lines in train_level_*.py

For testing:

 Edite several top lines in test_level_*.py



