from PIL import Image,ImageDraw
import numpy as np
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
from xml.dom import minidom
import random
import requests
import argparse
import cv2
from io import BytesIO
import os
import json
from skimage.transform import rescale, resize



img_path_list=glob.glob('./map/*')

if not (os.path.exists('./train2014_template/')):
  os.mkdir('./train2014_template/')
if not (os.path.exists('./train2014_template_original/')):
  os.mkdir('./train2014_template_original/')
if not (os.path.exists('./train2014_label/')):
  os.mkdir('./train2014_label/')
if not (os.path.exists('./train2014_input/')):
  os.mkdir('./train2014_input/')


if not (os.path.exists('./val2014_template/')):
  os.mkdir('./val2014_template/')
if not (os.path.exists('./val2014_template_original/')):
  os.mkdir('./val2014_template_original/')

if not (os.path.exists('./val2014_label/')):
  os.mkdir('./val2014_label/')
if not (os.path.exists('./val2014_input/')):
  os.mkdir('./val2014_input/')

output_region=200 # half, size is 400*400

index=0
for i in img_path_list:
  index=index+1
  print (index)
  img_map=plt.imread(i)

  img_name=i.split('/')[-1]
  img_satellite=plt.imread('./satellite/'+img_name)

  for rand_num in range(14):
    
    center_x=random.randint(output_region+50,1280-output_region-50)
    center_y=random.randint(output_region+50,1080-output_region-50)

    squre_img=img_satellite[(center_y-output_region):(center_y+output_region),(center_x-output_region):(center_x+output_region),:3]
    squre_img=resize(squre_img,(192,192))
    if rand_num<11:
      plt.imsave('./train2014_input/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'.jpg',squre_img)
    else:
      plt.imsave('./val2014_input/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'.jpg',squre_img)
    squre_img=img_map[(center_y-output_region):(center_y+output_region),(center_x-output_region):(center_x+output_region),:3]

    squre_img=resize(squre_img,(192,192))
    if rand_num<11:
      plt.imsave('./train2014_template_original/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'.jpg',squre_img)
    else:
      plt.imsave('./val2014_template_original/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'.jpg',squre_img)



    top_left_box_u=random.randint(0,63)
    top_left_box_v=random.randint(0,63)

    top_right_box_u=random.randint(128,191)
    top_right_box_v=random.randint(0,63)

    bottom_left_box_u=random.randint(0,63)
    bottom_left_box_v=random.randint(128,191)

    bottom_right_box_u=random.randint(128,191)
    bottom_right_box_v=random.randint(128,191)

     # prepare source and target four points
    src_points=[[top_left_box_u,top_left_box_v],[top_right_box_u,top_right_box_v],[bottom_left_box_u,bottom_left_box_v],[bottom_right_box_u,bottom_right_box_v]]

    tgt_points=[[32,32],[159,32],[32,159],[159,159]]

    src_points=np.reshape(src_points,[4,1,2])
    tgt_points=np.reshape(tgt_points,[4,1,2])

    # find homography
    h_matrix, status = cv2.findHomography(src_points, tgt_points,0)


    simulated_drone_img = cv2.warpPerspective(squre_img,h_matrix,(192,192))

    if rand_num<11:
      plt.imsave('./train2014_template/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'.jpg',simulated_drone_img[32:160,32:160,:])
    else:
      plt.imsave('./val2014_template/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'.jpg',simulated_drone_img[32:160,32:160,:])


    label = {}
    label['location'] = []

    label['location'].append({
            'top_left_u':top_left_box_u,
            'top_left_v': top_left_box_v
        })
    label['location'].append({
            'top_right_u':top_right_box_u,
            'top_right_v':top_right_box_v
        })
    label['location'].append({
            'bottom_left_u':bottom_left_box_u,
            'bottom_left_v':bottom_left_box_v
        })
    label['location'].append({
            'bottom_right_u':bottom_right_box_u,
            'bottom_right_v':bottom_right_box_v
        })

    if rand_num<11:
        with open('./train2014_label/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'_label.txt', 'w') as outfile:
            json.dump(label, outfile)
    else:
        with open('./val2014_label/'+str(rand_num)+'_'+img_name[:(len(img_name)-4)]+'_label.txt', 'w') as outfile:
            json.dump(label, outfile)







  



