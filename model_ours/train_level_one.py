
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



parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.00001,help='learning_rate')

parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=4,help='batch_size')

parser.add_argument('--feature_map_type', action="store", dest="feature_map_type", default='special',help='regular or special')

parser.add_argument('--save_eval_f', action="store", dest="save_eval_f", type=int, default=400000,help='save and eval after how many iterations')

parser.add_argument('--epoch_start', action="store", dest="epoch_start", type=int, default=1,help='train from which epoch')

parser.add_argument('--sample_noise', action="store", dest="sample_noise", type=int, default=4,help='samples noise number, 4 for google, 2 for MSCOCO')

parser.add_argument('--lambda_loss', action="store", dest="lambda_loss", type=int, default=0.2,help='0.2 for Google, 0.1 for MSCOCO')


parser.add_argument('--epoch_num', action="store", dest="epoch_num", type=int, default=10,help='how many epochs to train')


input_parameters = parser.parse_args()



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)





if input_parameters.feature_map_type=='regular':
  save_path='./checkpoints/'+input_parameters.dataset_name+'/level_one_regular/'

elif input_parameters.feature_map_type=='special':
  save_path='./checkpoints/'+input_parameters.dataset_name+'/level_one/'


if not(os.path.exists('./checkpoints')):
    os.mkdir('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name)):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name)
if not(os.path.exists(save_path)):
    os.mkdir(save_path)


lr=input_parameters.learning_rate






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
    


'''
def compute_ssim(img_1,img_2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    paddings = tf.constant([[0,0],[1, 1,], [1, 1],[0,0]])
        
    img_1=tf.pad(img_1, paddings, "REFLECT")
    img_2=tf.pad(img_2, paddings, "REFLECT")
        
    mu_1=tf.nn.avg_pool2d(img_1,ksize=3,strides=1,padding='VALID')
    mu_2=tf.nn.avg_pool2d(img_2,ksize=3,strides=1,padding="VALID")

    sigma_1=tf.nn.avg_pool2d(img_1**2,ksize=3,strides=1,padding='VALID')-mu_1**2
    sigma_2=tf.nn.avg_pool2d(img_2**2,ksize=3,strides=1,padding='VALID')-mu_2**2
    sigma_1_2=tf.nn.avg_pool2d(img_1*img_2,ksize=3,strides=1,padding='VALID')-mu_1*mu_2

    SSIM_n=(2 * mu_1 * mu_2 + C1) * (2 * sigma_1_2 + C2)
    SSIM_d = (mu_1 ** 2 + mu_2 ** 2 + C1) * (sigma_1 + sigma_2 + C2)

    #return (1 - SSIM_n / SSIM_d) / 2

    return tf.clip_by_value((1 - SSIM_n / SSIM_d) / 2, 0, 1)
'''

def compute_ssim(img_1,img_2):

    return tf.math.pow((img_1-img_2),2)


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

        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)

def gt_motion_rs_random_noisy(u_list,v_list,batch_size,lambda_noisy):
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
        element_h_matrix=np.reshape(h_matrix,(9,1))
        noisy_matrix=np.zeros((9,1))
        for jj in range(8):
            #if jj!=0 and jj!=4: 
            noisy_matrix[jj,0]=element_h_matrix[jj,0]*lambda_noisy[jj]  
        noisy_matrix=np.reshape(noisy_matrix,(3,3))    
        matrix_list.append(noisy_matrix)
    return np.asarray(matrix_list).astype(np.float32)
'''
def calculate_feature_map(input_tensor):
    bs,height,width,channel=tf.shape(input_tensor)
    path_extracted=tf.image.extract_patches(input_tensor, sizes=(1,3,3,1), strides=(1,1,1,1), rates=(1,1,1,1), padding='SAME')
    path_extracted=tf.reshape(path_extracted,(bs,height,width,channel,9))
    path_extracted_mean=tf.math.reduce_mean(path_extracted,axis=3,keepdims=True)

    #path_extracted_mean=tf.tile(path_extracted_mean,[1,1,1,channel,1])
    path_extracted=path_extracted-path_extracted_mean
    path_extracted_transpose=tf.transpose(path_extracted,(0,1,2,4,3))
    variance_matrix=tf.matmul(path_extracted_transpose,path_extracted)
    eigenvalue=tf.linalg.eigh(variance_matrix)[0]
    return  tf.math.reduce_max(eigenvalue,axis=-1,keepdims=True)/tf.math.reduce_sum(eigenvalue,axis=-1,keepdims=True)
'''

def calculate_feature_map(input_tensor):
    bs,height,width,channel=tf.shape(input_tensor)
    path_extracted=tf.image.extract_patches(input_tensor, sizes=(1,3,3,1), strides=(1,1,1,1), rates=(1,1,1,1), padding='SAME')
    path_extracted=tf.reshape(path_extracted,(bs,height,width,channel,9))
    path_extracted_mean=tf.math.reduce_mean(path_extracted,axis=3,keepdims=True)

    #path_extracted_mean=tf.tile(path_extracted_mean,[1,1,1,channel,1])
    path_extracted=path_extracted-path_extracted_mean
    path_extracted_transpose=tf.transpose(path_extracted,(0,1,2,4,3))
    variance_matrix=tf.matmul(path_extracted_transpose,path_extracted)
    
    tracevalue=tf.linalg.trace(variance_matrix)
    row_sum=tf.reduce_sum(variance_matrix,axis=-1)
    max_row_sum=tf.math.reduce_max(row_sum,axis=-1)
    min_row_sum=tf.math.reduce_min(row_sum,axis=-1)
    mimic_ratio=(max_row_sum+min_row_sum)/2.0/tracevalue
    
    return  tf.expand_dims(mimic_ratio,axis=-1)

#extract feature tensor 
#apply 

#level_one_template=ResNet_first_template()

if input_parameters.feature_map_type=='regular':
    level_one_input=ResNet_first_input(if_regular=True)
    level_one_template=ResNet_first_template(if_regular=True)

elif input_parameters.feature_map_type=='special':
    level_one_input=ResNet_first_input()
    level_one_template=ResNet_first_template()




if input_parameters.epoch_start>1:
    #load weights
    level_one_input.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1)+"input_full")

    level_one_template.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1)+"template_full")


initial_matrix=initial_motion_COCO()
    
LK_layer_one=Lucas_Kanade_layer(batch_size=input_parameters.batch_size,height_template=128,width_template=128,num_channels=1)






for current_epoch in range(input_parameters.epoch_num):


    if input_parameters.dataset_name=='MSCOCO':
        data_loader_caller=data_loader_MSCOCO('train')

    if input_parameters.dataset_name=='GoogleMap':
        data_loader_caller=data_loader_GoogleMap('train')

    if input_parameters.dataset_name=='GoogleEarth':
        data_loader_caller=data_loader_GoogleEarth('train')

    if input_parameters.dataset_name=='DayNight':
        data_loader_caller=data_loader_DayNight('train')
        

    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

    print("Starting epoch " + str(current_epoch+input_parameters.epoch_start))
    print("Learning rate is " + str(lr)) 

    error_ave_1000=0.0
    convex_loss_total=0.0
    ssim_loss_total=0.0

    for iters in range(10000000):
        input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=input_parameters.batch_size)
        

        if len(np.shape(input_img))<2:
          level_one_input.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"input_full")
          level_one_template.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"template_full")
          break

        gt_matrix_one=gt_motion_rs(u_list,v_list,batch_size=input_parameters.batch_size)
        with tf.GradientTape() as tape:
            input_feature=level_one_input.call(input_img)
            template_feature=level_one_template.call(template_img)

            if input_parameters.feature_map_type=='regular':
                input_feature_map_one=input_feature
                template_feature_map_one=template_feature

            elif input_parameters.feature_map_type=='special':
                input_feature_map_one=calculate_feature_map(input_feature)
                template_feature_map_one=calculate_feature_map(template_feature)

            input_warped_to_template=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one)

            ssim_middle_one=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template))
            ssim_middle=ssim_middle_one

            

            #print ('!!!!!!!!!')

            for nn in range(input_parameters.sample_noise):
                lambda_one=(np.random.rand(8)-0.5)/6


                for mm in range(len(lambda_one)):
                  if lambda_one[mm]>0 and lambda_one[mm]<0.02:
                    lambda_one[mm]=0.02
                  if lambda_one[mm]<0 and lambda_one[mm]>-0.02:
                    lambda_one[mm]=-0.02
              
           
                gt_matrix_noisy_one=gt_motion_rs_random_noisy(u_list,v_list,batch_size=input_parameters.batch_size,lambda_noisy=lambda_one)

                
                input_warped_to_template_shift_one_left=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one+gt_matrix_noisy_one)
                ssim_shift_one_left=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_left))
                ssim_convex_one_left= -tf.math.minimum((ssim_shift_one_left-ssim_middle_one)-np.sum(lambda_one**2),0)



                '''
                input_warped_to_template_shift_one_left_left=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one+1.5*gt_matrix_noisy_one)
                ssim_shift_one_left_left=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_left_left))

                input_warped_to_template_shift_one_left_right=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one+0.5*gt_matrix_noisy_one)
                ssim_shift_one_left_right=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_left_right))

                ssim_convex_one_left_further= -tf.math.minimum((ssim_shift_one_left_left+ssim_shift_one_left_right-2*ssim_shift_one_left)-(np.sum((0.5*lambda_one)**2)+np.sum((1.5*lambda_one)**2)-2*np.sum(lambda_one**2)),0)

                '''

                input_warped_to_template_shift_one_left_left=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one+2.0*gt_matrix_noisy_one)
                ssim_shift_one_left_left=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_left_left))

                 
                ssim_convex_one_left_further= -tf.math.minimum((ssim_shift_one_left_left-ssim_shift_one_left)-(np.sum((2*lambda_one)**2)-np.sum(lambda_one**2)),0)




                input_warped_to_template_shift_one_right=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one-gt_matrix_noisy_one)
                ssim_shift_one_right=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_right))
                ssim_convex_one_right= -tf.math.minimum((ssim_shift_one_right-ssim_middle_one)-np.sum(lambda_one**2),0)

                '''
                input_warped_to_template_shift_one_right_left=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one-1.5*gt_matrix_noisy_one)
                ssim_shift_one_right_left=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_right_left))

                input_warped_to_template_shift_one_right_right=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one-0.5*gt_matrix_noisy_one)
                ssim_shift_one_right_right=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_right_right))

                ssim_convex_one_right_further= -tf.math.minimum((ssim_shift_one_right_left+ssim_shift_one_right_right-2*ssim_shift_one_right)-(np.sum((0.5*lambda_one)**2)+np.sum((1.5*lambda_one)**2)-2*np.sum(lambda_one**2)),0)
                '''

                input_warped_to_template_shift_one_right_right=LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one-2.0*gt_matrix_noisy_one)
                ssim_shift_one_right_right=tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template_shift_one_right_right))

               
                ssim_convex_one_right_further= -tf.math.minimum((ssim_shift_one_right_right-ssim_shift_one_right)-(np.sum((2*lambda_one)**2)-np.sum(lambda_one**2)),0)





         
                if nn==0:
                    convex_loss=ssim_convex_one_left+ssim_convex_one_right+ssim_convex_one_left_further+ssim_convex_one_right_further
                else:
                    convex_loss=convex_loss+ssim_convex_one_left+ssim_convex_one_right+ssim_convex_one_left_further+ssim_convex_one_right_further

            convex_loss=convex_loss
            total_loss=ssim_middle+input_parameters.lambda_loss*convex_loss

            #print (ssim_middle)
            #print (convex_loss)

            convex_loss_total+=convex_loss
            ssim_loss_total+=ssim_middle
            error_ave_1000+=total_loss
            #print ('!!!!!!!!!!!!!!!!!')



        all_parameters=level_one_template.trainable_variables+level_one_input.trainable_variables
           
        grads = tape.gradient(total_loss, all_parameters)
        grads=[tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))

      

 
        if iters%100==0 and iters>0:
            
            
            print(iters)
            print (save_path)

            print (error_ave_1000/100)
            print (ssim_loss_total/100)
            print (convex_loss_total/100)
            error_ave_1000=0.0
            convex_loss_total=0.0
            ssim_loss_total=0.0

        if iters%input_parameters.save_eval_f==0 and iters>0:

            level_one_input.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"input_"+str(iters))
            level_one_template.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"template_"+str(iters))


           
        input_img = None
        u_list = None
        v_list = None
        template_img = None

        input_feature_map = None
        template_feature_map = None
        input_warped_to_template=None
        input_warped_to_template_left_1=None
        input_warped_to_template_left_2=None
        input_warped_to_template_right_1=None
        input_warped_to_template_right_2=None

               




