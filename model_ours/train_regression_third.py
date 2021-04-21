
import sys
sys.path.append('../')
from data_read import *
from net import *
import matplotlib.pyplot as plt
import numpy as np

import argparse

import os



# goolge earth
# 0.000004  16   10   80


# MSCOCO
# 0.000004  32   5   40

parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="GoogleEarth",help='MSCOCO,GoogleMap,GoogleEarth,DayNight')



parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.000004,help='learning_rate')

parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=16,help='batch_size')


parser.add_argument('--save_eval_f', action="store", dest="save_eval_f", type=int, default=400000,help='save and eval after how many iterations')

parser.add_argument('--epoch_load_one', action="store", dest="epoch_load_one", type=int, default=100,help='epoch_load_one which epoch to load')

parser.add_argument('--epoch_load_two', action="store", dest="epoch_load_two", type=int, default=100,help='epoch_load_two which epoch to load')

parser.add_argument('--epoch_start', action="store", dest="epoch_start", type=int, default=1,help='train from which epoch')

parser.add_argument('--epoch_decay', action="store", dest="epoch_decay", type=int, default=10,help='how many epoch to over lr by 2')

parser.add_argument('--epoch_num', action="store", dest="epoch_num", type=int, default=80,help='how many epochs to train')




input_parameters = parser.parse_args()



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)







save_path='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_3/'


if not(os.path.exists('./checkpoints')):
    os.mkdir('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name)):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name)
if not(os.path.exists(save_path)):
    os.mkdir(save_path)

save_path_one='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_1/'

save_path_two='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_2/'


lr=input_parameters.learning_rate







def loss_func(batch_size,four_cornners,network_output,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127):

    four_conner=tf.dtypes.cast(four_cornners,tf.float32)
    
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

    

def gt_motion_rs(network_output_1,network_output_2,u_list,v_list,batch_size=1):

    # prepare source and target four points
    matrix_list=[]
    four_cornners=[]
    extra=tf.ones((batch_size,1))
    predicted_matrix_1=tf.concat([network_output_1,extra],axis=-1)
    predicted_matrix_1=tf.reshape(predicted_matrix_1,[batch_size,3,3])

    predicted_matrix_2=tf.concat([network_output_2,extra],axis=-1)
    predicted_matrix_2=tf.reshape(predicted_matrix_2,[batch_size,3,3])

    for i in range(batch_size):
       
        src_points=[[0,0,1],[127,0,1],[127,127,1],[0,127,1]]
        src_points=np.transpose(src_points)
        src_points=np.dot(predicted_matrix_1[i,:,:],src_points)
        src_points=np.transpose(src_points)
        src_points=src_points/src_points[:,2:]

        src_points=np.transpose(src_points)
        src_points=np.dot(predicted_matrix_2[i,:,:],src_points)
        src_points=np.transpose(src_points)
        src_points=src_points/src_points[:,2:]

        four_cornners.append(np.transpose(src_points))
        src_points=src_points[:,:2]

        #tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

        tgt_points=np.concatenate([u_list[i:(i+1),:],v_list[i:(i+1),:]],axis=0)
        tgt_points=np.transpose(tgt_points)
        tgt_points=np.expand_dims(tgt_points,axis=1)
       
        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)


        matrix_list.append(np.squeeze(np.reshape(h_matrix,(1,9))[:,:8]))
    return np.asarray(matrix_list).astype(np.float32),np.asarray(four_cornners)

def construct_matrix(batch_size,network_output,network_output_2=[0]):
    extra=tf.ones((batch_size,1))
    predicted_matrix=tf.concat([network_output,extra],axis=-1)
    predicted_matrix=tf.reshape(predicted_matrix,[batch_size,3,3])
    if len(np.shape(network_output_2))>1:
        predicted_matrix_2=tf.concat([network_output_2,extra],axis=-1)
        predicted_matrix_2=tf.reshape(predicted_matrix_2,[batch_size,3,3])
    hh_matrix=[]
    for i in range(batch_size):
        if len(np.shape(network_output_2))>1:
            hh_matrix.append(np.linalg.inv(np.dot(predicted_matrix_2[i,:,:],predicted_matrix[i,:,:])))
        else:
            hh_matrix.append(np.linalg.inv(predicted_matrix[i,:,:]))
        #hh_matrix.append(predicted_matrix[i,:,:])
    
    #return tf.linalg.inv(predicted_matrix+0.0001)
    return np.asarray(hh_matrix)



regression_network_one=Net_first()

regression_network_one.load_weights(save_path_one + 'epoch_'+str(input_parameters.epoch_load_one))

regression_network_two=Net_second()

regression_network_two.load_weights(save_path_two + 'epoch_'+str(input_parameters.epoch_load_two))

regression_network_three=Net_third()


LK_layer_one=Lucas_Kanade_layer(batch_size=input_parameters.batch_size,height_template=192,width_template=192,num_channels=3)


if input_parameters.epoch_start>1:
    #load weights
    regression_network_three.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1))

else:

    regression_network_three.load_weights(save_path_two + 'epoch_'+str(input_parameters.epoch_load_two))
    
#LK_layer_one=Lucas_Kanade_layer(batch_size=input_parameters.batch_size,height_template=128,width_template=128,num_channels=1)



for current_epoch in range(input_parameters.epoch_num):


    if input_parameters.dataset_name=='MSCOCO':
        data_loader_caller=data_loader_MSCOCO('train')

    if input_parameters.dataset_name=='GoogleMap':
        data_loader_caller=data_loader_GoogleMap('train')

    if input_parameters.dataset_name=='GoogleEarth':
        data_loader_caller=data_loader_GoogleEarth('train')

    if input_parameters.dataset_name=='DayNight':
        data_loader_caller=data_loader_DayNight('train')
        
    if current_epoch>0 and current_epoch%input_parameters.epoch_decay==0:
      lr=lr*0.5

    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

    print("Starting epoch " + str(current_epoch+input_parameters.epoch_start))
    print("Learning rate is " + str(lr)) 


    total_loss=0.0
    for iters in range(10000000):
        input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=input_parameters.batch_size)
        

        if len(np.shape(input_img))<2:
          regression_network_three.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch))
          break
        

        input_img_grey=tf.image.rgb_to_grayscale(input_img)
        template_img_new=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)
            
        template_img_grey=tf.image.rgb_to_grayscale(template_img_new)
            
        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)

        homography_vector_one=regression_network_one.call(network_input,training=False)

        matrix_one=construct_matrix(input_parameters.batch_size,homography_vector_one)

        template_img_new=LK_layer_one.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_one)
        template_img_grey=tf.image.rgb_to_grayscale(template_img_new)
            
        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)

        homography_vector_two=regression_network_two.call(network_input,training=False)

        matrix_two=construct_matrix(input_parameters.batch_size,homography_vector_one,homography_vector_two)

        template_img_new=LK_layer_one.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_two)
        template_img_grey=tf.image.rgb_to_grayscale(template_img_new)
            


        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)

            
        gt_vector,four_cornners=gt_motion_rs(homography_vector_one,homography_vector_two,u_list,v_list,batch_size=input_parameters.batch_size)


        with tf.GradientTape() as tape:
            homography_vector_three=regression_network_three.call(network_input)

            
            loss= tf.reduce_mean(tf.math.sqrt((homography_vector_three-gt_vector)**2))       

            loss_1=loss_func(input_parameters.batch_size,four_cornners,homography_vector_three,u_list,v_list)



        all_parameters=regression_network_three.trainable_variables
           
        grads = tape.gradient(loss, all_parameters)
        grads=[tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))

        total_loss+=loss



        if iters%100==0 and iters>0:
            
            
            print(iters)
            print (save_path)

            print (total_loss/100)
            print (loss_1)

            total_loss=0.0


        if iters%input_parameters.save_eval_f==0 and iters>0:

            regression_network_three.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+str(iters))
            