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



img_path_list=glob.glob('./val2014_input/*')
file = open("./val_img_list.txt","w")
for i in range(len(img_path_list)):
  img_name=img_path_list[i].split('/')[-1]
  file.write(img_name)
  file.write('\n')


file.close()


