
import sys
sys.path.append('../')
from data_read import *
from net import *
import matplotlib.pyplot as plt
import numpy as np

import argparse

import os







parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="GoogleMap",help='MSCOCO,GoogleMap,GoogleEarth,DayNight')

parser.add_argument('--epoch_eval', action="store", dest="epoch_eval", type=int, default=100,help='eval from which epoch')


input_parameters = parser.parse_args()



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)







save_path='./checkpoints/'+input_parameters.dataset_name+'/regression_original/'


if not(os.path.exists('./checkpoints')):
    os.mkdir('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name)):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name)
if not(os.path.exists(save_path)):
    os.mkdir(save_path)









def loss_func(batch_size,network_output,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127):
    four_conner=[[top_left_u,top_left_v,1],[bottom_right_u,top_left_v,1],[bottom_right_u,bottom_right_v,1],[top_left_u,bottom_right_v,1]]
    four_conner=np.asarray(four_conner)
    four_conner=np.transpose(four_conner)
    four_conner=np.expand_dims(four_conner,axis=0)
    four_conner=np.tile(four_conner,[batch_size,1,1]).astype(np.float32)
    four_conner=tf.dtypes.cast(four_conner,tf.float32)
    
    extra=tf.ones((batch_size,1))
    predicted_matrix=tf.concat([network_output,extra],axis=-1)

    predicted_matrix=tf.reshape(predicted_matrix,[batch_size,3,3])

    new_four_points=tf.matmul(predicted_matrix,four_conner)

    new_four_points_scale=new_four_points[:,2:,:]
    new_four_points= new_four_points/new_four_points_scale
    
    
    u_predict=new_four_points[:,0,:]
    v_predict=new_four_points[:,1,:]
    average_conner=tf.math.pow(u_predict-u_list,2)+tf.math.pow(v_predict-v_list,2)
    #print (np.shape(average_conner))
    average_conner=tf.reduce_mean(tf.math.sqrt(average_conner))
    
    
    return average_conner

    

def gt_motion_rs(u_list,v_list,batch_size=1):
    # prepare source and target four points
    matrix_list=[]
    for i in range(batch_size):
       
        src_points=[[0,0],[127,0],[127,127],[0,127]]

        #tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

        tgt_points=np.concatenate([u_list[i:(i+1),:],v_list[i:(i+1),:]],axis=0)
        tgt_points=np.transpose(tgt_points)
        tgt_points=np.expand_dims(tgt_points,axis=1)
       
        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)

        matrix_list.append(np.squeeze(np.reshape(h_matrix,(1,9))[:,:8]))
    return np.asarray(matrix_list).astype(np.float32)


regression_network=Net_first()





regression_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_eval))



    
#LK_layer_one=Lucas_Kanade_layer(batch_size=input_parameters.batch_size,height_template=128,width_template=128,num_channels=1)





if input_parameters.dataset_name=='MSCOCO':
    data_loader_caller=data_loader_MSCOCO('val')

if input_parameters.dataset_name=='GoogleMap':
    data_loader_caller=data_loader_GoogleMap('val')

if input_parameters.dataset_name=='GoogleEarth':
    data_loader_caller=data_loader_GoogleEarth('val')

if input_parameters.dataset_name=='DayNight':
    data_loader_caller=data_loader_DayNight('val')
        



total_loss=0.0
for iters in range(10000000):
    input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=1)
    if len(np.shape(input_img))<2:
        break

    '''
    input_img_new=tf.image.resize(input_img, (128,128))
    input_img_grey=tf.image.rgb_to_grayscale(input_img_new)
    template_img_grey=tf.image.rgb_to_grayscale(template_img)
    network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
    '''
    input_img_grey=tf.image.rgb_to_grayscale(input_img)
        
    template_img_new=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)
        
    template_img_grey=tf.image.rgb_to_grayscale(template_img_new)


        
    network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)

    homography_vector=regression_network.call(network_input,training=False)
    gt_vector=gt_motion_rs(u_list,v_list,batch_size=1)
    loss= tf.reduce_mean(tf.math.sqrt((homography_vector-gt_vector)**2))       

    loss_1=loss_func(1,homography_vector,u_list,v_list)


    
    total_loss+=loss_1


              
    print(iters)
   
    print (loss_1)
    print (total_loss/iters)

