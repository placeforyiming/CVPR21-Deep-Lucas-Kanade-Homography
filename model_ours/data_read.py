
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import tensorflow as tf
import cv2
import sys
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import json
import glob
import ntpath

import skimage.io as io
import scipy.io as sio

from skimage.io import imsave, imread
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
sys.path.append("../")


sys.path.append("../")

class data_loader_MSCOCO():

  def __init__(self,dataset_name='train'):
    if dataset_name=='train':
      self.img_path=glob.glob('../Dataset/MSCOCO/train2014_input/*')
      self.input_path='../Dataset/MSCOCO/train2014_input/'
      self.label_path='../Dataset/MSCOCO/train2014_label/'
      self.template_path='../Dataset/MSCOCO/train2014_template/'

    elif dataset_name=='val':
      self.img_path=glob.glob('../Dataset/MSCOCO/val2014_input/*')
      self.input_path='../Dataset/MSCOCO/val2014_input/'
      self.label_path='../Dataset/MSCOCO/val2014_label/'
      self.template_path='../Dataset/MSCOCO/val2014_template/'

    else:
      print ("no data load")
    random.shuffle(self.img_path)
    print (len(self.img_path))
    self.start=0



  def data_read_batch(self,batch_size=8):
    input_all=[]
    label_u=[]
    label_v=[]
    template_all=[]
    for i in range(batch_size):
     
      img_name=self.img_path[self.start].split('/')[-1]

      self.start=self.start+1
      if self.start>(len(self.img_path)-1):
        input_all=[0]
        label_u=[1]
        label_v=[2]
        template_all=[3]
        break

      input_img=plt.imread(self.input_path+img_name)/255.0
      template_img=plt.imread(self.template_path+img_name)/255.0


      with open(self.label_path+img_name[:(len(img_name)-4)]+'_label.txt', 'r') as outfile:
        data = json.load(outfile)
      u_list=[data['location'][0]['top_left_u'],data['location'][1]['top_right_u'],data['location'][3]['bottom_right_u'],data['location'][2]['bottom_left_u']]
      v_list=[data['location'][0]['top_left_v'],data['location'][1]['top_right_v'],data['location'][3]['bottom_right_v'],data['location'][2]['bottom_left_v']]

      input_all.append(input_img)
      label_u.append(u_list)
      label_v.append(v_list)
      template_all.append(template_img)

      

    return np.asarray(input_all).astype(np.float32),np.asarray(label_u).astype(np.float32),np.asarray(label_v).astype(np.float32),np.asarray(template_all).astype(np.float32)

class data_loader_GoogleMap():

  def __init__(self,dataset_name='train'):
    self.dataset_name=dataset_name
    if dataset_name=='train':
      self.img_path=glob.glob('../Dataset/GoogleMap/train2014_input/*')
      self.input_path='../Dataset/GoogleMap/train2014_input/'
      self.label_path='../Dataset/GoogleMap/train2014_label/'
      self.template_path='../Dataset/GoogleMap/train2014_template/'

    elif dataset_name=='val':
      self.img_path=glob.glob('../Dataset/GoogleMap/val2014_template/*')
      self.input_path='../Dataset/GoogleMap/val2014_input/'
      self.label_path='../Dataset/GoogleMap/val2014_label/'
      self.template_path='../Dataset/GoogleMap/val2014_template/'

    else:
      print ("no data load")
    random.shuffle(self.img_path)
    print (len(self.img_path))
    self.start=0


  def channel_norm(self,img):
    img=np.squeeze(img)
    for i in range(3):
        temp_max=np.max(img[:,:,i])
        temp_min=np.min(img[:,:,i])
        img[:,:,i] =(img[:,:,i]-temp_min)/(temp_max-temp_min+0.000001)
    return img

  def data_read_batch(self,batch_size=8):
    input_all=[]
    label_u=[]
    label_v=[]
    template_all=[]
    for i in range(batch_size):
     
      img_name=self.img_path[self.start].split('/')[-1]

      self.start=self.start+1
      if self.start>(len(self.img_path)-1):
        input_all=[0]
        label_u=[1]
        label_v=[2]
        template_all=[3]
        break

      input_img=plt.imread(self.input_path+img_name)/255.0
      #input_img=self.channel_norm(input_img)
      template_img=plt.imread(self.template_path+img_name)/255.0
      template_img=self.channel_norm(template_img)

      with open(self.label_path+img_name[:(len(img_name)-4)]+'_label.txt', 'r') as outfile:
          data = json.load(outfile)
         
      u_list=[data['location'][0]['top_left_u'],data['location'][1]['top_right_u'],data['location'][3]['bottom_right_u'],data['location'][2]['bottom_left_u']]
      v_list=[data['location'][0]['top_left_v'],data['location'][1]['top_right_v'],data['location'][3]['bottom_right_v'],data['location'][2]['bottom_left_v']]

      input_all.append(input_img)
      label_u.append(u_list)
      label_v.append(v_list)
      template_all.append(template_img)

      

    return np.asarray(input_all).astype(np.float32),np.asarray(label_u).astype(np.float32),np.asarray(label_v).astype(np.float32),np.asarray(template_all).astype(np.float32)

class data_loader_GoogleEarth():

  def __init__(self,dataset_name='train'):
    if dataset_name=='train':
      self.img_path=glob.glob('../Dataset/GoogleEarth/train2014_input/*')
      self.input_path='../Dataset/GoogleEarth/train2014_input/'
      self.label_path='../Dataset/GoogleEarth/train2014_label/'
      self.template_path='../Dataset/GoogleEarth/train2014_template/'
      random.shuffle(self.img_path)

    elif dataset_name=='val':
      self.img_path=glob.glob('../Dataset/GoogleEarth/val2014_input/*')
      self.input_path='../Dataset/GoogleEarth/val2014_input/'
      self.label_path='../Dataset/GoogleEarth/val2014_label/'
      self.template_path='../Dataset/GoogleEarth/val2014_template/'

    else:
      print ("no data load")
    
    print (len(self.img_path))
    self.start=0



  def data_read_batch(self,batch_size=8):
    input_all=[]
    label_u=[]
    label_v=[]
    template_all=[]
    for i in range(batch_size):
     
      img_name=self.img_path[self.start].split('/')[-1]


      self.start=self.start+1
      if self.start>(len(self.img_path)-1):
        input_all=[0]
        label_u=[1]
        label_v=[2]
        template_all=[3]
        break

      input_img=plt.imread(self.input_path+img_name)/255.0
      template_img=plt.imread(self.template_path+img_name)/255.0


      with open(self.label_path+img_name[:(len(img_name)-4)]+'_label.txt', 'r') as outfile:
        data = json.load(outfile)
      u_list=[data['location'][0]['top_left_u'],data['location'][1]['top_right_u'],data['location'][3]['bottom_right_u'],data['location'][2]['bottom_left_u']]
      v_list=[data['location'][0]['top_left_v'],data['location'][1]['top_right_v'],data['location'][3]['bottom_right_v'],data['location'][2]['bottom_left_v']]

      input_all.append(input_img)
      label_u.append(u_list)
      label_v.append(v_list)
      template_all.append(template_img)

      

    return np.asarray(input_all).astype(np.float32),np.asarray(label_u).astype(np.float32),np.asarray(label_v).astype(np.float32),np.asarray(template_all).astype(np.float32)

class data_loader_DayNight():

  def __init__(self,dataset_name='train'):
    if dataset_name=='train':
      self.img_path=glob.glob('../Dataset/DayNight/train2014_input/*')
      self.input_path='../Dataset/DayNight/train2014_input/'
      self.label_path='../Dataset/DayNight/train2014_label/'
      self.template_path='../Dataset/DayNight/train2014_template/'

    elif dataset_name=='val':
      self.img_path=glob.glob('../Dataset/DayNight/val2014_input/*')
      self.input_path='../Dataset/DayNight/val2014_input/'
      self.label_path='../Dataset/DayNight/val2014_label/'
      self.template_path='../Dataset/DayNight/val2014_template/'

    else:
      print ("no data load")
    random.shuffle(self.img_path)
    print (len(self.img_path))
    self.start=0


  def channel_norm(self,img):
    img=np.squeeze(img)
    for i in range(3):
        temp_max=np.max(img[:,:,i])
        temp_min=np.min(img[:,:,i])
        img[:,:,i] =(img[:,:,i]-temp_min)/(temp_max-temp_min+0.000001)
    return img

  def data_read_batch(self,batch_size=8):
    input_all=[]
    label_u=[]
    label_v=[]
    template_all=[]
    for i in range(batch_size):
     
      img_name=self.img_path[self.start].split('/')[-1]

      self.start=self.start+1
      if self.start>(len(self.img_path)-1):
        input_all=[0]
        label_u=[1]
        label_v=[2]
        template_all=[3]
        break

      input_img=plt.imread(self.input_path+img_name)/255.0
      template_img=plt.imread(self.template_path+img_name)/255.0
      input_img=self.channel_norm(input_img)
      template_img=self.channel_norm(template_img)

      with open(self.label_path+img_name[:(len(img_name)-4)]+'_label.txt', 'r') as outfile:
        data = json.load(outfile)
      u_list=[data['location'][0]['top_left_u'],data['location'][1]['top_right_u'],data['location'][3]['bottom_right_u'],data['location'][2]['bottom_left_u']]
      v_list=[data['location'][0]['top_left_v'],data['location'][1]['top_right_v'],data['location'][3]['bottom_right_v'],data['location'][2]['bottom_left_v']]

      input_all.append(input_img)
      label_u.append(u_list)
      label_v.append(v_list)
      template_all.append(template_img)

      

    return np.asarray(input_all).astype(np.float32),np.asarray(label_u).astype(np.float32),np.asarray(label_v).astype(np.float32),np.asarray(template_all).astype(np.float32)

'''

A=data_loader_GoogleMap('val')
A.data_read_batch(1)



A=data_loader_MSCOCO('val')
A.data_read_batch(1)
'''

































def generate_samples(index,satellite_img_list,cropped_img_in_original_HRO_list,label_list,simulated_drone_img_list):
  satellite_img=plt.imread(satellite_img_list[index])
  cropped_img_in_original_HRO=plt.imread(cropped_img_in_original_HRO_list[index])
  simulated_drone_img=plt.imread(simulated_drone_img_list[index])

  u_list=[]
  v_list=[]
  with open(label_list[index]) as json_file:
      data = json.load(json_file)
      for p in data['pixel_location']:
          #{'top_left_u': 1017, 'top_left_v': 980}
          #{'top_right_u': 799, 'top_right_v': 854}
          #{'bottom_right_u': 894, 'bottom_right_v': 617}
          #{'bottom_left_u': 1088, 'bottom_left_v': 760}

          u_list.append(list(p.values())[0])
          v_list.append(list(p.values())[1])

  u_min=np.min(u_list)
  v_min=np.min(v_list)
  u_max=np.max(u_list)
  v_max=np.max(v_list)
  img_1=satellite_img[v_min:v_max, u_min:u_max,:3]
  new_height=v_max-v_min
  new_width=u_max-u_min

  crop_target_in_new_img=np.hstack((np.asarray(u_list).reshape(4,1),np.asarray(v_list).reshape(4,1)))
  rect = cv2.boundingRect(crop_target_in_new_img)
  x,y,w,h = rect
  croped = satellite_img[y:y+h, x:x+w,:3]
  crop_target_in_new_img = crop_target_in_new_img - crop_target_in_new_img.min(axis=0)
  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [crop_target_in_new_img ], -1, (255, 255, 255), -1, cv2.LINE_AA)
  cropped_img_with_black_region = cv2.bitwise_and(croped, croped, mask=mask)
  img_2=cropped_img_with_black_region

  img_3=resize(simulated_drone_img,(new_height,new_width))

  img_4=resize(cropped_img_in_original_HRO,(new_height,new_width))

  return img_1,img_2,img_3,img_4
  


def generate_samples_for_evaluate(index,satellite_img_list,cropped_img_in_original_HRO_list,label_list,simulated_drone_img_list):
  satellite_img=plt.imread(satellite_img_list[index])
  cropped_img_in_original_HRO=plt.imread(cropped_img_in_original_HRO_list[index])
  simulated_drone_img=plt.imread(simulated_drone_img_list[index])

  u_list=[]
  v_list=[]
  with open(label_list[index]) as json_file:
      data = json.load(json_file)
      for p in data['pixel_location']:
          #{'top_left_u': 1017, 'top_left_v': 980}
          #{'top_right_u': 799, 'top_right_v': 854}
          #{'bottom_right_u': 894, 'bottom_right_v': 617}
          #{'bottom_left_u': 1088, 'bottom_left_v': 760}

          u_list.append(list(p.values())[0])
          v_list.append(list(p.values())[1])

  u_min=np.min(u_list)
  v_min=np.min(v_list)
  u_max=np.max(u_list)
  v_max=np.max(v_list)
  img_1=satellite_img[v_min:v_max, u_min:u_max,:3]
  new_height=v_max-v_min
  new_width=u_max-u_min

  crop_target_in_new_img=np.hstack((np.asarray(u_list).reshape(4,1),np.asarray(v_list).reshape(4,1)))
  rect = cv2.boundingRect(crop_target_in_new_img)
  x,y,w,h = rect
  croped = satellite_img[y:y+h, x:x+w,:3]
  crop_target_in_new_img = crop_target_in_new_img - crop_target_in_new_img.min(axis=0)
  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [crop_target_in_new_img ], -1, (255, 255, 255), -1, cv2.LINE_AA)
  cropped_img_with_black_region = cv2.bitwise_and(croped, croped, mask=mask)
  img_2=cropped_img_with_black_region

  img_3=resize(simulated_drone_img,(new_height,new_width))

  img_4=resize(cropped_img_in_original_HRO,(new_height,new_width))

  return img_1,img_2,img_3,img_4,u_list-u_min,v_list-v_min
  



class data_loader_remote_sensing():
  def __init__(self,dataset_name='train', template_size=128*2,input_size=192*2):

      self.dataset_name=dataset_name
      self.template_size=template_size
      self.input_size=input_size
      self.satellite_img_list,self.cropped_img_in_original_HRO_list,self.label_list,self.simulated_drone_img_list=self.load_path()
      self.total_sample=len(self.satellite_img_list)
      self.total_list=[i for i in range(self.total_sample)]
      random.shuffle(self.total_list)
      self.start_index=0

      

      

  def load_path(self):

      satellite_img_list=glob.glob('../Dataset/remote_sensing/'+self.dataset_name+'/*/*/satellite_img.png')
      cropped_img_in_original_HRO_list=[]
      label_list=[]
      simulated_drone_img_list=[]

      for i in satellite_img_list:
          path_list=i.split('/')
          #print (path_list)
          cropped_img_in_original_HRO_list.append(path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+path_list[3]+'/'+path_list[4]+'/'+path_list[5]+'/'+'cropped_img_in_original_HRO.png')
          simulated_drone_img_list.append(path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+path_list[3]+'/'+path_list[4]+'/'+path_list[5]+'/'+'simulated_drone_img.png')
          label_list.append(path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+path_list[3]+'/'+path_list[4]+'/'+path_list[5]+'/'+'label.txt')

      return satellite_img_list,cropped_img_in_original_HRO_list,label_list,simulated_drone_img_list
      

  def data_read_batch(self,batch_size):

    img_input_list=[]
    img_template_list=[]
    img_u_list=[]
    img_v_list=[]

    for i in range(batch_size):
      index=self.start_index
      self.start_index=self.start_index+1
      
      if self.start_index>(self.total_sample-1):
        return [0],[1],[2],[3]
      
      satellite_img=plt.imread(self.satellite_img_list[index])
      cropped_img_in_original_HRO=plt.imread(self.cropped_img_in_original_HRO_list[index])
      simulated_drone_img=plt.imread(self.simulated_drone_img_list[index])

      u_list=[]
      v_list=[]
      with open(self.label_list[index]) as json_file:
          data = json.load(json_file)
          for p in data['pixel_location']:
              #{'top_left_u': 1017, 'top_left_v': 980}
              #{'top_right_u': 799, 'top_right_v': 854}
              #{'bottom_right_u': 894, 'bottom_right_v': 617}
              #{'bottom_left_u': 1088, 'bottom_left_v': 760}

              u_list.append(list(p.values())[0])
              v_list.append(list(p.values())[1])

      u_min=np.min(u_list)
      v_min=np.min(v_list)
      u_max=np.max(u_list)
      v_max=np.max(v_list)
      img_1=satellite_img[v_min:v_max, u_min:u_max,:3]
      new_height=v_max-v_min
      new_width=u_max-u_min

      

      img_3=resize(simulated_drone_img,(self.template_size,self.template_size))

      img_1=resize(img_1,(self.input_size,self.input_size))

      


      img_input_list.append(img_1[:,:,:3]*2.0-1.0)
      img_u_list.append((u_list-u_min)/float(new_width)*self.input_size)
      img_v_list.append((v_list-v_min)/float(new_height)*self.input_size)
      img_template_list.append(img_3[:,:,:3]*2.0-1.0)
    return np.asarray(img_input_list).astype(np.float32),np.asarray(img_u_list).astype(np.float32),np.asarray(img_v_list).astype(np.float32),np.asarray(img_template_list).astype(np.float32)
      
class data_loader_remote_sensing_new():
  def __init__(self,dataset_name='train'):

      self.dataset_name=dataset_name

      self.input_img_list,self.label_u_list,self.label_v_list,self.template_img_list=self.load_path()
      self.total_sample=len(self.input_img_list)
      self.total_list=[i for i in range(self.total_sample)]
      random.shuffle(self.total_list)
      print (self.total_sample)
      self.start_index=0

      

      

  def load_path(self):

      input_img_list=glob.glob('./prepared_img_remote_sensing/'+self.dataset_name+'/input_img/*.jpg')
      label_u_list=[]
      label_v_list=[]
      template_img_list=[]

      for i in input_img_list:
          path_list=i.split('/')[-1]
          template_img_list.append('./prepared_img_remote_sensing/'+self.dataset_name+'/template_img/'+path_list)

          label_u_list.append('./prepared_img_remote_sensing/'+self.dataset_name+'/label/'+path_list[:(len(path_list)-4)]+'u_list_gt.txt')

          label_v_list.append('./prepared_img_remote_sensing/'+self.dataset_name+'/label/'+path_list[:(len(path_list)-4)]+'v_list_gt.txt')
          

      return input_img_list,label_u_list,label_v_list,template_img_list
      

  def data_read_batch(self,batch_size):

    img_input_list=[]
    img_template_list=[]
    img_u_list=[]
    img_v_list=[]

    for i in range(batch_size):
      index=self.start_index
      self.start_index=self.start_index+1
      
      if self.start_index>(self.total_sample-1):
        return [0],[1],[2],[3]
      
      input_img=plt.imread(self.input_img_list[index])/255.0
      template_img=plt.imread(self.template_img_list[index])/255.0
      
      u_list=[]
      v_list=[]

      with open(self.label_u_list[index]) as json_file:
        for mm in json_file:
          u_list_string=mm.split(',')
          for nn in range(4):
            u_list.append(float(u_list_string[nn]))
      with open(self.label_v_list[index]) as json_file:
        for mm in json_file:
          v_list_string=mm.split(',')
          for nn in range(4):
            v_list.append(float(v_list_string[nn]))

      img_u_list.append(u_list)
      img_v_list.append(v_list)
      img_input_list.append(input_img*2.0-1.0)
      img_template_list.append(template_img*2.0-1.0)

    return np.asarray(img_input_list).astype(np.float32),np.asarray(img_u_list).astype(np.float32),np.asarray(img_v_list).astype(np.float32),np.asarray(img_template_list).astype(np.float32)



class data_loader_remote_sensing_to_north():
  def __init__(self,dataset_name='train', template_size=128,input_size=192):

      self.dataset_name=dataset_name
      self.template_size=template_size
      self.input_size=input_size
      self.satellite_img_list,self.label_list,self.simulated_drone_img_list=self.load_path()
      self.total_sample=len(self.satellite_img_list)
      self.total_list=[i for i in range(self.total_sample)]
      random.shuffle(self.total_list)
      self.start_index=0

      

      

  def load_path(self):

      satellite_img_list=glob.glob('../Dataset/remote_sensing/'+self.dataset_name+'/*/*/satellite_img_no_rotation.png')
      cropped_img_in_original_HRO_list=[]
      label_list=[]
      simulated_drone_img_list=[]

      for i in satellite_img_list:
          path_list=i.split('/')
          #print (path_list)
          #cropped_img_in_original_HRO_list.append(path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+path_list[3]+'/'+path_list[4]+'/'+path_list[5]+'/'+'cropped_img_in_original_HRO.png')
          simulated_drone_img_list.append(path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+path_list[3]+'/'+path_list[4]+'/'+path_list[5]+'/'+'simulated_drone_img_no_rotation.png')
          label_list.append(path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+path_list[3]+'/'+path_list[4]+'/'+path_list[5]+'/'+'label_no_rotation.txt')

      return satellite_img_list,label_list,simulated_drone_img_list


  def data_read_batch(self,batch_size):

    img_input_list=[]
    img_template_list=[]
    img_u_list=[]
    img_v_list=[]

    for i in range(batch_size):
      index=self.start_index
      self.start_index=self.start_index+1
      
      if self.start_index>(self.total_sample-1):
        return [0],[1],[2],[3]
      
      satellite_img=plt.imread(self.satellite_img_list[index])
      #cropped_img_in_original_HRO=plt.imread(self.cropped_img_in_original_HRO_list[index])
      simulated_drone_img=plt.imread(self.simulated_drone_img_list[index])

      u_list=[]
      v_list=[]

      with open(self.label_list[index]) as json_file:
          data = json.load(json_file)
          
          for p in data['pixel_location']:
              #{'top_left_u': 1017, 'top_left_v': 980}
              #{'top_right_u': 799, 'top_right_v': 854}
              #{'bottom_right_u': 894, 'bottom_right_v': 617}
              #{'bottom_left_u': 1088, 'bottom_left_v': 760}

              u_list.append(list(p.values())[0])
              v_list.append(list(p.values())[1])



      u_min=np.min(u_list)
      v_min=np.min(v_list)
      u_max=np.max(u_list)
      v_max=np.max(v_list)
      img_1=satellite_img[v_min:v_max, u_min:u_max,:3]
      new_height=v_max-v_min
      new_width=u_max-u_min

      

      img_3=resize(simulated_drone_img,(self.template_size,self.template_size))

      img_1=resize(img_1,(self.input_size,self.input_size))

      


      img_input_list.append(img_1[:,:,:3]*2.0-1.0)
      img_u_list.append((u_list-u_min)/float(new_width)*self.input_size)
      img_v_list.append((v_list-v_min)/float(new_height)*self.input_size)
      img_template_list.append(img_3[:,:,:3]*2.0-1.0)
    return np.asarray(img_input_list).astype(np.float32),np.asarray(img_u_list).astype(np.float32),np.asarray(img_v_list).astype(np.float32),np.asarray(img_template_list).astype(np.float32)
      

     
class data_loader_remote_sensing_to_north_new():
  def __init__(self,dataset_name='train'):

      self.dataset_name=dataset_name

      self.input_img_list,self.label_u_list,self.label_v_list,self.template_img_list=self.load_path()
      self.total_sample=len(self.input_img_list)
      self.total_list=[i for i in range(self.total_sample)]
      random.shuffle(self.total_list)
      print (self.total_sample)
      self.start_index=0

      

      

  def load_path(self):

      input_img_list=glob.glob('./prepared_img_remote_sensing_no_rotation/'+self.dataset_name+'/input_img/*.jpg')
      label_u_list=[]
      label_v_list=[]
      template_img_list=[]

      for i in input_img_list:
          path_list=i.split('/')[-1]
          template_img_list.append('./prepared_img_remote_sensing_no_rotation/'+self.dataset_name+'/template_img/'+path_list)

          label_u_list.append('./prepared_img_remote_sensing_no_rotation/'+self.dataset_name+'/label/'+path_list[:(len(path_list)-4)]+'u_list_gt.txt')

          label_v_list.append('./prepared_img_remote_sensing_no_rotation/'+self.dataset_name+'/label/'+path_list[:(len(path_list)-4)]+'v_list_gt.txt')
          

      return input_img_list,label_u_list,label_v_list,template_img_list
      

  def data_read_batch(self,batch_size):

    img_input_list=[]
    img_template_list=[]
    img_u_list=[]
    img_v_list=[]

    for i in range(batch_size):
      index=self.start_index
      self.start_index=self.start_index+1
      
      if self.start_index>(self.total_sample-1):
        return [0],[1],[2],[3]
      
      input_img=plt.imread(self.input_img_list[index])/255.0
      template_img=plt.imread(self.template_img_list[index])/255.0
      
      u_list=[]
      v_list=[]

      with open(self.label_u_list[index]) as json_file:
        for mm in json_file:
          u_list_string=mm.split(',')
          for nn in range(4):
            u_list.append(float(u_list_string[nn]))
      with open(self.label_v_list[index]) as json_file:
        for mm in json_file:
          v_list_string=mm.split(',')
          for nn in range(4):
            v_list.append(float(v_list_string[nn]))

      img_u_list.append(u_list)
      img_v_list.append(v_list)
      img_input_list.append(input_img*2.0-1.0)
      img_template_list.append(template_img*2.0-1.0)

    return np.asarray(img_input_list).astype(np.float32),np.asarray(img_u_list).astype(np.float32),np.asarray(img_v_list).astype(np.float32),np.asarray(img_template_list).astype(np.float32)

