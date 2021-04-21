from data_read import *
from net import *
import matplotlib.pyplot as plt
import numpy as np

import argparse

import os





parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="GoogleMap",help='MSCOCO,GoogleEarth,GoogleMap')

parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.000005,help='learning_rate')

parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=16,help='batch_size')


parser.add_argument('--save_eval_f', action="store", dest="save_eval_f", type=int, default=400000,help='save and eval after how many iterations')

parser.add_argument('--epoch_start', action="store", dest="epoch_start", type=int, default=20,help='train from which epoch')


parser.add_argument('--alpha', action="store", dest="alpha", type=int, default=4.0,help='alpha')

parser.add_argument('--epoch_num', action="store", dest="epoch_num", type=int, default=1,help='how many epochs to train')


input_parameters = parser.parse_args()



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2800)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)







save_path='./checkpoints/'+input_parameters.dataset_name+'/level_one/'


if not(os.path.exists('./checkpoints')):
    os.mkdir('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name)):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name)
if not(os.path.exists(save_path)):
    os.mkdir(save_path)


lr=input_parameters.learning_rate





level_one_network=Level_one()
if input_parameters.epoch_start>1:
    #load weights
    level_one_network.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1)+"_full")

def initial_motion_COCO():
    # prepare source and target four points
    matrix_list=[]
    for i in range(input_parameters.batch_size):
       
        src_points=[[0,0],[127,0],[127,127],[0,127]]

        tgt_points=[[32,32],[160,32],[160,160],[32,160]]

    
        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)
        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)

def initial_motion_rs():
    # prepare source and target four points
    matrix_list=[]
    for i in range(input_parameters.batch_size):
       
        src_points=[[0,0],[127*2+1,0],[127*2+1,127*2+1],[0,127*2+1]]

        tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

    
        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)
        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)

def construct_matrix(initial_matrix,scale_factor,batch_size):
    #scale_factor size_now/(size to get matrix)
    initial_matrix=tf.cast(initial_matrix,dtype=tf.float32)
    
    scale_matrix=np.eye(3)*scale_factor
    scale_matrix[2,2]=1.0
    scale_matrix=tf.cast(scale_matrix,dtype=tf.float32)
    scale_matrix_inverse=tf.linalg.inv(scale_matrix)

    scale_matrix=tf.expand_dims(scale_matrix,axis=0)
    scale_matrix=tf.tile(scale_matrix,[batch_size,1,1])

    scale_matrix_inverse=tf.expand_dims(scale_matrix_inverse,axis=0)
    scale_matrix_inverse=tf.tile(scale_matrix_inverse,[batch_size,1,1])

    final_matrix=tf.matmul(tf.matmul(scale_matrix,initial_matrix),scale_matrix_inverse)
    return final_matrix



def average_cornner_error(batch_size,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127):
    
    four_conner=[[top_left_u,top_left_v,1],[bottom_right_u,top_left_v,1],[bottom_right_u,bottom_right_v,1],[top_left_u,bottom_right_v,1]]
    four_conner=np.asarray(four_conner)
    four_conner=np.transpose(four_conner)
    four_conner=np.expand_dims(four_conner,axis=0)
    four_conner=np.tile(four_conner,[batch_size,1,1]).astype(np.float32)
    
    new_four_points=tf.matmul(predicted_matrix,four_conner)
    
    new_four_points_scale=new_four_points[:,2:,:]
    new_four_points= new_four_points/new_four_points_scale
    
    
    u_predict=new_four_points[:,0,:]
    v_predict=new_four_points[:,1,:]
    
    average_conner=tf.math.pow(u_predict-u_list,2)+tf.math.pow(v_predict-v_list,2)
    #print (np.shape(average_conner))
    average_conner=tf.reduce_sum(average_conner)/batch_size
    
    
    return average_conner
    


def loss_function(batch_size,initial_matrix,predicted_matrix,u_list,v_list,alpha,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127):
    d_cornner=average_cornner_error(batch_size,predicted_matrix,u_list,v_list,top_left_u,top_left_v,bottom_right_u,bottom_right_v)
    
    d_cornner_initial=average_cornner_error(batch_size,initial_matrix,u_list,v_list,top_left_u,top_left_v,bottom_right_u,bottom_right_v)

    delta=d_cornner_initial-tf.math.pow(tf.math.maximum(0,tf.math.sqrt(d_cornner_initial)-2*alpha),2)

    #print (delta)
    total_loss=tf.math.maximum(0,1+delta+d_cornner-d_cornner_initial)
    #print (d_cornner)
    #print (d_cornner_initial)

    #print (total_loss)
    return total_loss




initial_matrix=initial_motion_COCO()
LK_layer=Lucas_Kanade_layer(batch_size=input_parameters.batch_size,height_template=16,width_template=16,num_channels=4)






initial_matrix_scaled=construct_matrix(initial_matrix,scale_factor=0.125,batch_size=input_parameters.batch_size)

for current_epoch in range(input_parameters.epoch_num):


    if input_parameters.dataset_name=='MSCOCO':
        data_loader_caller=data_loader_MSCOCO('train')
    if input_parameters.dataset_name=='GoogleMap':
        data_loader_caller=data_loader_GoogleMap('train')
    if input_parameters.dataset_name=='GoogleEarth':
        data_loader_caller=data_loader_GoogleEarth('train')




    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

    print("Starting epoch " + str(current_epoch+input_parameters.epoch_start))
    print("Learning rate is " + str(lr)) 

    error_ave_1000=0.0
    cornner_error=0.0
    initial_cornner_error=0.0

    for iters in range(10000000):
        input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=input_parameters.batch_size)

        if len(np.shape(input_img))<2:
          level_one_network.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"_full")
          break


        with tf.GradientTape() as tape:
            input_feature=level_one_network.call(input_img)
            template_feature=level_one_network.call(template_img)



            updated_matrix=LK_layer.update_matrix(template_feature,input_feature,initial_matrix_scaled)
            #print (updated_matrix)

            updated_matrix=construct_matrix(updated_matrix,scale_factor=8,batch_size=input_parameters.batch_size)


            total_loss=loss_function(input_parameters.batch_size,initial_matrix,updated_matrix,u_list,v_list, input_parameters.alpha,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)

        grads = tape.gradient(total_loss, level_one_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, level_one_network.trainable_variables))

        error_ave_1000=error_ave_1000+total_loss
        cornner_error+=np.sqrt(average_cornner_error(input_parameters.batch_size,updated_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)/4.0)
        initial_cornner_error+=np.sqrt(average_cornner_error(input_parameters.batch_size,initial_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)/4.0)
             
        #print (np.sqrt(average_cornner_error(input_parameters.batch_size,updated_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)/4.0))
        #print (np.sqrt(average_cornner_error(input_parameters.batch_size,unity_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)/4.0))
        if iters%100==0 and iters>0:
            
            
            print(iters)
            print (save_path)
            print (error_ave_1000/100)
            print (cornner_error/100)
            print (initial_cornner_error/100)
            error_ave_1000=0.0
            cornner_error=0.0
            initial_cornner_error=0.0

        if iters%input_parameters.save_eval_f==0 and iters>0:

            level_one_network.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"_"+str(iters))

           
        input_img = None
        u_list = None
        v_list = None
        template_img = None

        template_feature = None
        updated_matrix = None

       




