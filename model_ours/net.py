import tensorflow as tf
import numpy as np

class Level_one(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Level_one") as scope:
            super(Level_one,self).__init__()
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1d = tf.keras.layers.BatchNormalization()


            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1e = tf.keras.layers.BatchNormalization()

            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1f = tf.keras.layers.BatchNormalization()
            
            self.conv_output = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.00))



    def call(self, image,training=True):
       

        with tf.name_scope(name="Level_one") as scope:
            
    
            tensor_1=self.conv1a(image)
            tensor_1=self.bn_1a(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)
           
            tensor_1=self.conv1b(tensor_1)
            tensor_1=self.bn_1b(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1c(tensor_1)
            tensor_1=self.bn_1c(tensor_1,training=training) 
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            tensor_1=self.bn_1d(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1e(tensor_1)
            tensor_1=self.bn_1e(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1f(tensor_1)
            tensor_1=self.bn_1f(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            output=self.conv_output(tensor_1)


        return output

class Level_two(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Level_two") as scope:
            super(Level_two,self).__init__()
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1d = tf.keras.layers.BatchNormalization()


            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1e = tf.keras.layers.BatchNormalization()

            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1f = tf.keras.layers.BatchNormalization()
            
            self.conv_output = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.00))



    def call(self, image,training=True):
       

        with tf.name_scope(name="Level_two") as scope:
            
    
            tensor_1=self.conv1a(image)
            tensor_1=self.bn_1a(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)
           
            tensor_1=self.conv1b(tensor_1)
            tensor_1=self.bn_1b(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1c(tensor_1)
            tensor_1=self.bn_1c(tensor_1,training=training) 
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            tensor_1=self.bn_1d(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1e(tensor_1)
            tensor_1=self.bn_1e(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1f(tensor_1)
            tensor_1=self.bn_1f(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            output=self.conv_output(tensor_1)


        return output

class Level_three(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Level_three") as scope:
            super(Level_three,self).__init__()
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1d = tf.keras.layers.BatchNormalization()


            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1e = tf.keras.layers.BatchNormalization()

            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.bn_1f = tf.keras.layers.BatchNormalization()
            
            self.conv_output = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.00))



    def call(self, image,training=True):
       

        with tf.name_scope(name="Level_three") as scope:
            
    
            tensor_1=self.conv1a(image)
            tensor_1=self.bn_1a(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)
           
            tensor_1=self.conv1b(tensor_1)
            tensor_1=self.bn_1b(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1c(tensor_1)
            tensor_1=self.bn_1c(tensor_1,training=training) 
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            tensor_1=self.bn_1d(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1e(tensor_1)
            tensor_1=self.bn_1e(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1f(tensor_1)
            tensor_1=self.bn_1f(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            output=self.conv_output(tensor_1)


        return output

class Level_four(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Level_four") as scope:
            super(Level_four,self).__init__()
            
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.00))
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.00))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()

            #self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            #self.bn_1c = tf.keras.layers.BatchNormalization()
            
            #self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            #self.bn_1d = tf.keras.layers.BatchNormalization()

           
            self.conv_output = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.00))



    def call(self, image,training=True):
       

        with tf.name_scope(name="Level_four") as scope:
            
    
            tensor_1=self.conv1a(image)
            tensor_1=self.bn_1a(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)
           
            tensor_1=self.conv1b(tensor_1)
            tensor_1=self.bn_1b(tensor_1,training=training)
            tensor_1=tf.nn.leaky_relu(tensor_1)

            #tensor_1=self.conv1c(tensor_1)
            #tensor_1=self.bn_1c(tensor_1,training=training) 
            #tensor_1=tf.nn.leaky_relu(tensor_1)
            
            #tensor_1=self.conv1d(tensor_1)
            #tensor_1=self.bn_1d(tensor_1,training=training)
            #tensor_1=tf.nn.leaky_relu(tensor_1)
            

            output=self.conv_output(tensor_1)


        return output




class Lucas_Kanade_layer():
  def __init__(self,batch_size,height_template,width_template,num_channels,mask_region_size=0):
    #self.batch_size=batch_size
    self.batch_size=1
    self.height_template=height_template-mask_region_size
    self.width_template=width_template-mask_region_size
    self.num_channels=num_channels
    
    self.coords=self.meshgrid(self.batch_size,self.height_template,self.width_template)

    self.p_W_p_p=self.partial_W_partial_p(self.batch_size,self.height_template,self.width_template,self.num_channels)


    self.mask_region_size=mask_region_size    




  def form_unity_matrix(self,matrix_size=3):
    unity_matrix=tf.eye(matrix_size)
    unity_matrix=tf.expand_dims(unity_matrix,axis=0)
    unity_matrix=tf.tile(unity_matrix,[self.batch_size,1,1])
    return unity_matrix
     

    
  def meshgrid(self, batch, height, width):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      matrix: warp matrix explained in projective_inverse_warp
    Returns:
      x,y grid coordinates [batch, 2 , height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)

    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)

    return coords

    

  def meshgrid_after(self,coords, matrix):
    coords=tf.tensordot(matrix,coords,axes = 1)
   
    coords=coords/coords[:,2:,:,:]
    coords=coords[:,:2,:,:]
    
    coords=tf.transpose(coords,[0,2,3,1])
    return coords


  def bilinear_sampler(self,imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """
    def _repeat(x, n_repeats):
      rep = tf.transpose(
          tf.expand_dims(tf.ones(shape=tf.stack([
              n_repeats,
          ])), 1), [1, 0])
      rep = tf.cast(rep, 'float32')
      x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
      return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
      coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
      inp_size = imgs.get_shape()
      coord_size = coords.get_shape()
      out_size = coords.get_shape().as_list()
      out_size[3] = imgs.get_shape().as_list()[3]

      coords_x = tf.cast(coords_x, 'float32')
      coords_y = tf.cast(coords_y, 'float32')

      x0 = tf.floor(coords_x)
      x1 = x0 + 1
      y0 = tf.floor(coords_y)
      y1 = y0 + 1

      y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
      x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
      zero = tf.zeros([1], dtype='float32')

      x0_safe = tf.clip_by_value(x0, zero, x_max)
      y0_safe = tf.clip_by_value(y0, zero, y_max)
      x1_safe = tf.clip_by_value(x1, zero, x_max)
      y1_safe = tf.clip_by_value(y1, zero, y_max)

      ## bilinear interp weights, with points outside the grid having weight 0
      # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
      # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
      # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
      # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

      wt_x0 = x1_safe - coords_x
      wt_x1 = coords_x - x0_safe
      wt_y0 = y1_safe - coords_y
      wt_y1 = coords_y - y0_safe

      ## indices in the flat image to sample from
      dim2 = tf.cast(inp_size[2], 'float32')
      dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
      base = tf.reshape(
          _repeat(
              tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
              coord_size[1] * coord_size[2]),
          [out_size[0], out_size[1], out_size[2], 1])

      base_y0 = base + y0_safe * dim2
      base_y1 = base + y1_safe * dim2
      idx00 = tf.reshape(x0_safe + base_y0, [-1])
      idx01 = x0_safe + base_y1
      idx10 = x1_safe + base_y0
      idx11 = x1_safe + base_y1

      ## sample from imgs
      imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
      imgs_flat = tf.cast(imgs_flat, 'float32')
      im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
      im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
      im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
      im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

      w00 = wt_x0 * wt_y0
      w01 = wt_x0 * wt_y1
      w10 = wt_x1 * wt_y0
      w11 = wt_x1 * wt_y1

      output = tf.add_n([
          w00 * im00, w01 * im01,
          w10 * im10, w11 * im11
      ])
      return output




  def projective_inverse_warp(self,input_feature, matrix):
    """Inverse warp a source image to the target image plane based on projection.

    img: input_img with shape(batch_size, height, width, channel), or the feature map of input_img
    matrix:  warp matrix h_matrix= cv2.findHomography(src_points, tgt_points,0), src_points are four vertex of template
    height_tgt, width_tgt: the height and width of template_img, since template_img is smaller than input_img

    return warped result from input_img to template_img


    """

    pixel_coords = self.meshgrid_after(self.coords, matrix)
   
    output_img = self.bilinear_sampler(input_feature, pixel_coords )
    return output_img



  def partial_W_partial_p(self,batch_size,height,width,num_channel):
    
    #height,width,num_channel: all are based on template_features
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)




    ones = tf.ones_like(x_t)
    zeros = tf.zeros_like(x_t)
    
    x_t=tf.expand_dims(x_t,axis=0)
    x_t=tf.expand_dims(x_t,axis=-1)
    x_t=tf.tile(x_t,[batch_size,1,1,num_channel])
    x_t=tf.reshape(x_t,[batch_size,-1])
    
    y_t=tf.expand_dims(y_t,axis=0)
    y_t=tf.expand_dims(y_t,axis=-1)
    y_t=tf.tile(y_t,[batch_size,1,1,num_channel])
    y_t=tf.reshape(y_t,[batch_size,-1])
    
    
    ones=tf.expand_dims(ones,axis=0)
    ones=tf.expand_dims(ones,axis=-1)
    ones=tf.tile(ones,[batch_size,1,1,num_channel])
    ones=tf.reshape(ones,[batch_size,-1])
    
    
    zeros=tf.expand_dims(zeros,axis=0)
    zeros=tf.expand_dims(zeros,axis=-1)
    zeros=tf.tile(zeros,[batch_size,1,1,num_channel])
    zeros=tf.reshape(zeros,[batch_size,-1])

    x_t=tf.expand_dims(x_t,axis=-1)
    y_t=tf.expand_dims(y_t,axis=-1)
    ones=tf.expand_dims(ones,axis=-1)
    zeros=tf.expand_dims(zeros,axis=-1)
    
    
    first_row=tf.concat([x_t,y_t,ones,zeros,zeros,zeros,-x_t*x_t,-x_t*y_t],axis=-1)
    second_row=tf.concat([zeros,zeros,zeros,x_t,y_t,ones,-y_t*x_t,-y_t*y_t],axis=-1)

    
    first_row=tf.expand_dims(first_row,axis=-1)
    second_row=tf.expand_dims(second_row,axis=-1)

    
    
    partial_W_partial_p=tf.concat([first_row,second_row],axis=-1)

    partial_W_partial_p=tf.transpose(partial_W_partial_p,[0,1,3,2])    
    return partial_W_partial_p
    
  def Gradient_F(self, template_feature):
    # template_feature
    assert len(np.shape(template_feature))==4

    bs,height,width,channel=np.shape(template_feature)
    paddings = tf.constant([[0,0,],[1, 1,], [1, 1,],[0,0]])
    template_feature=tf.pad(template_feature, paddings, "SYMMETRIC")

    template_feature_partial_u=(template_feature[:,1:height+1,2:,:]-template_feature[:,1:height+1,:width,:])/1.0
    template_feature_partial_v=(template_feature[:,2:,1:width+1,:]-template_feature[:,:height,1:width+1,:])/1.0
    
    template_feature_partial_u=tf.reshape(template_feature_partial_u,[bs,-1])
    template_feature_partial_v=tf.reshape(template_feature_partial_v,[bs,-1])
    template_feature_partial_u=tf.expand_dims(template_feature_partial_u,axis=-1)
    template_feature_partial_v=tf.expand_dims(template_feature_partial_v,axis=-1)
    Gradient_F=tf.concat([template_feature_partial_u,template_feature_partial_v],axis=2)

    return Gradient_F



  def calculate_J(self,template_feature):
    assert len(np.shape(template_feature))==4

    Gradient_F=self.Gradient_F(template_feature)


    Gradient_F=tf.expand_dims(Gradient_F,axis=2)

    J=tf.matmul(Gradient_F,self.p_W_p_p)


    return J[:,:,0,:]

  def compute_ssim(self,img_1,img_2):
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

  def calculate_r(self,template_feature,input_feature,matrix):
    warped_template=self.projective_inverse_warp(input_feature, matrix)
    #photoness_loss=self.compute_ssim(warped_template,template_feature)
    photoness_loss=warped_template-template_feature
    photoness_loss=tf.reshape(photoness_loss,[self.batch_size,-1])
    return photoness_loss

  def calculate_del_p(self,template_feature,input_feature,matrix):
    J=self.calculate_J(template_feature)
    r=self.calculate_r(template_feature,input_feature,matrix)

    temp_1=tf.matmul(tf.transpose(J,[0,2,1]),J)
    r=tf.expand_dims(r,axis=-1)
    temp_2=tf.matmul(tf.transpose(J,[0,2,1]),r)

    delta_p=tf.matmul(tf.linalg.inv(temp_1+self.form_unity_matrix(8)*0.001),temp_2)


    return delta_p[:,:,0]

  def update_matrix(self,template_feature_real,input_feature_real,matrix_real): 

    real_bs=tf.shape(template_feature_real)[0]


    for i in range(real_bs):
      template_feature=template_feature_real[i,int(self.mask_region_size/2):(int(self.mask_region_size/2)+self.height_template),int(self.mask_region_size/2):(int(self.mask_region_size/2)+self.width_template),:]

      

      template_feature=tf.expand_dims(template_feature,axis=0)
      input_feature=input_feature_real[i,:,:,:]
      input_feature=tf.expand_dims(input_feature,axis=0)

      matrix=matrix_real[i,:,:]
      matrix=tf.expand_dims(matrix,axis=0)


      delta_p=self.calculate_del_p(template_feature,input_feature,matrix)
      zeros=tf.zeros((self.batch_size,1))
      delta_p=tf.concat([delta_p,zeros],axis=1)

      
      delta_p=tf.reshape(delta_p,[self.batch_size,3,3])
      unity_matrix=self.form_unity_matrix(3)

      delta_p=delta_p+unity_matrix

      #final_p=tf.matmul(matrix,tf.linalg.inv(delta_p))

      final_p=tf.matmul(matrix,tf.linalg.inv(delta_p))

      normalize_value=final_p[:,2:,2:]
      normalize_value=tf.tile(normalize_value,[1,3,3])

      final_p=final_p/normalize_value
      #final_p_list.append(tf.squeeze(final_p))
      if i==0:
        final_p_list=final_p
        
      else:
        
        final_p_list=tf.concat([final_p_list,final_p],axis=0)


    

    return final_p_list
    

class ResNet_first_input(tf.keras.Model):
    def __init__(self, if_regular=False):
        self.if_regular=if_regular

        with tf.name_scope(name="ResNet_first_input") as scope:
            super(ResNet_first_input, self).__init__()

    
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()

            if self.if_regular:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            else:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            

            
          

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_first_input") as scope:



            
                        
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
            
            

            return (tensor_1_total)


class ResNet_first_template(tf.keras.Model):
    def __init__(self,if_regular=False):
        self.if_regular=if_regular

        with tf.name_scope(name="ResNet_first_template") as scope:
            super(ResNet_first_template, self).__init__()

    
           
                
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()

            if self.if_regular:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            else:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
       
            
            

            
 

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_first_template") as scope:
            
                      
            
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
                      

            return (tensor_1_total)


class ResNet_second_input(tf.keras.Model):
    def __init__(self,if_regular=False):
        self.if_regular=if_regular
        with tf.name_scope(name="ResNet_second_input") as scope:
            super(ResNet_second_input, self).__init__()

    
           
                
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()


            if self.if_regular:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            else:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
       
          
            
            

            
 

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_second_template") as scope:
            
                      
            
                        
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
                      

            return (tensor_1_total)


class ResNet_second_template(tf.keras.Model):
    def __init__(self,if_regular=False):
        self.if_regular=if_regular

        with tf.name_scope(name="ResNet_second_template") as scope:
            super(ResNet_second_template, self).__init__()

    
           
                
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()


            if self.if_regular:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            else:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
       
      
            

            
 

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_second_template") as scope:
            
                      
            
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
                      

            return (tensor_1_total)



class ResNet_third_input(tf.keras.Model):
    def __init__(self,if_regular=False):
        self.if_regular=if_regular

        with tf.name_scope(name="ResNet_third_input") as scope:
            super(ResNet_third_input, self).__init__()


                
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()


            if self.if_regular:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            else:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
       
      
            

            
 

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_third_template") as scope:
            
                      
            
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
                      

            return (tensor_1_total)


class ResNet_third_template(tf.keras.Model):
    def __init__(self,if_regular=False):
        self.if_regular=if_regular
        with tf.name_scope(name="ResNet_third_template") as scope:
            super(ResNet_third_template, self).__init__()

    
           
                
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()


            if self.if_regular:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            else:
                self.conv1_forge = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
       
            
            
            

            
 

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_third_template") as scope:
            
                      
            
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
                      

            return (tensor_1_total)





class Net_first(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Net_first") as scope:
            super(Net_first, self).__init__()

    
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()



            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2a = tf.keras.layers.BatchNormalization()
            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2b = tf.keras.layers.BatchNormalization()

            
            self.conv3a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3a = tf.keras.layers.BatchNormalization()
            
            self.conv3b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3b = tf.keras.layers.BatchNormalization()


            
            self.conv4a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4a = tf.keras.layers.BatchNormalization()
            
            self.conv4b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv5a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5a = tf.keras.layers.BatchNormalization()
            
            self.conv5b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5b = tf.keras.layers.BatchNormalization()


            self.conv6a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_6a = tf.keras.layers.BatchNormalization()
            
            self.conv6b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_6b = tf.keras.layers.BatchNormalization()


            self.flatten=tf.keras.layers.Flatten()


            #self.glob_average=tf.keras.layers.GlobalAveragePooling2D()

            self.conv_final_pre = tf.keras.layers.Dense(1024)

            self.conv_final = tf.keras.layers.Dense(8)

          
            
          

    def call(self, image,training=True):
       

        with tf.name_scope(name="Net_first") as scope:


            
            tensor_1=self.conv1a(image)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')


            tensor_1=self.conv2a(tensor_1)
            
            tensor_1=self.bn_2a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv2b(tensor_1)
            
            tensor_1=self.bn_2b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')


            tensor_1=self.conv3a(tensor_1)
            
            tensor_1=self.bn_3a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv3b(tensor_1)
            
            tensor_1=self.bn_3b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv4a(tensor_1)
            
            tensor_1=self.bn_4a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv4b(tensor_1)
            
            tensor_1=self.bn_4b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv5a(tensor_1)
            
            tensor_1=self.bn_5a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv5b(tensor_1)
            
            tensor_1=self.bn_5b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv6a(tensor_1)
            
            tensor_1=self.bn_6a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv6b(tensor_1)
            
            tensor_1=self.bn_6b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            #tensor_1=tf.nn.avg_pool(tensor_1, ksize=2, strides=1,padding='SAME')

            tensor_1=self.flatten(tensor_1)


            tensor_1=self.conv_final_pre(tensor_1)

            if training:
            
                tensor_1=tf.nn.dropout(tensor_1, rate=0.2)


            tensor_1=self.conv_final(tensor_1)


            return (tensor_1)


class Net_second(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Net_second") as scope:
            super(Net_second, self).__init__()

    
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()



            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2a = tf.keras.layers.BatchNormalization()
            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2b = tf.keras.layers.BatchNormalization()

            
            self.conv3a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3a = tf.keras.layers.BatchNormalization()
            
            self.conv3b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3b = tf.keras.layers.BatchNormalization()


            
            self.conv4a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4a = tf.keras.layers.BatchNormalization()
            
            self.conv4b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv5a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5a = tf.keras.layers.BatchNormalization()
            
            self.conv5b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5b = tf.keras.layers.BatchNormalization()


            self.conv6a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_6a = tf.keras.layers.BatchNormalization()
            
            self.conv6b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_6b = tf.keras.layers.BatchNormalization()


            self.flatten=tf.keras.layers.Flatten()


            #self.glob_average=tf.keras.layers.GlobalAveragePooling2D()

            self.conv_final_pre = tf.keras.layers.Dense(1024)

            self.conv_final = tf.keras.layers.Dense(8)

          
            
          

    def call(self, image,training=True):
       

        with tf.name_scope(name="Net_second") as scope:


            
            tensor_1=self.conv1a(image)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')


            tensor_1=self.conv2a(tensor_1)
            
            tensor_1=self.bn_2a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv2b(tensor_1)
            
            tensor_1=self.bn_2b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')


            tensor_1=self.conv3a(tensor_1)
            
            tensor_1=self.bn_3a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv3b(tensor_1)
            
            tensor_1=self.bn_3b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv4a(tensor_1)
            
            tensor_1=self.bn_4a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv4b(tensor_1)
            
            tensor_1=self.bn_4b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv5a(tensor_1)
            
            tensor_1=self.bn_5a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv5b(tensor_1)
            
            tensor_1=self.bn_5b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv6a(tensor_1)
            
            tensor_1=self.bn_6a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv6b(tensor_1)
            
            tensor_1=self.bn_6b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            #tensor_1=tf.nn.avg_pool(tensor_1, ksize=2, strides=1,padding='SAME')

            tensor_1=self.flatten(tensor_1)

            tensor_1=self.conv_final_pre(tensor_1)

            if training:
                #tensor_1=tf.nn.dropout(tensor_1, rate=0.5)

                tensor_1=tf.nn.dropout(tensor_1, rate=0.2)
           
            #self.glob_average=tf.keras.layers.GlobalAveragePooling2D()

            #tensor_1=self.glob_average(tensor_1)

            tensor_1=self.conv_final(tensor_1)


            return (tensor_1)


class Net_third(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Net_third") as scope:
            super(Net_third, self).__init__()

    
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()



            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2a = tf.keras.layers.BatchNormalization()
            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2b = tf.keras.layers.BatchNormalization()

            
            self.conv3a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3a = tf.keras.layers.BatchNormalization()
            
            self.conv3b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3b = tf.keras.layers.BatchNormalization()


            
            self.conv4a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4a = tf.keras.layers.BatchNormalization()
            
            self.conv4b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv5a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5a = tf.keras.layers.BatchNormalization()
            
            self.conv5b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5b = tf.keras.layers.BatchNormalization()


            self.conv6a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_6a = tf.keras.layers.BatchNormalization()
            
            self.conv6b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_6b = tf.keras.layers.BatchNormalization()


            self.flatten=tf.keras.layers.Flatten()


            #self.glob_average=tf.keras.layers.GlobalAveragePooling2D()

            self.conv_final_pre = tf.keras.layers.Dense(1024)

            self.conv_final = tf.keras.layers.Dense(8)


    def call(self, image,training=True):
       

        with tf.name_scope(name="Net_third") as scope:


            
            tensor_1=self.conv1a(image)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')


            tensor_1=self.conv2a(tensor_1)
            
            tensor_1=self.bn_2a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv2b(tensor_1)
            
            tensor_1=self.bn_2b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')


            tensor_1=self.conv3a(tensor_1)
            
            tensor_1=self.bn_3a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv3b(tensor_1)
            
            tensor_1=self.bn_3b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv4a(tensor_1)
            
            tensor_1=self.bn_4a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv4b(tensor_1)
            
            tensor_1=self.bn_4b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv5a(tensor_1)
            
            tensor_1=self.bn_5a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv5b(tensor_1)
            
            tensor_1=self.bn_5b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')



            tensor_1=self.conv6a(tensor_1)
            
            tensor_1=self.bn_6a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv6b(tensor_1)
            
            tensor_1=self.bn_6b(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            #tensor_1=tf.nn.avg_pool(tensor_1, ksize=2, strides=1,padding='SAME')

            tensor_1=self.flatten(tensor_1)

            tensor_1=self.conv_final_pre(tensor_1)

            if training:
                #tensor_1=tf.nn.dropout(tensor_1, rate=0.5)
                tensor_1=tf.nn.dropout(tensor_1, rate=0.2)
           
            #self.glob_average=tf.keras.layers.GlobalAveragePooling2D()

            #tensor_1=self.glob_average(tensor_1)

            tensor_1=self.conv_final(tensor_1)


            return (tensor_1)


'''
class Net_first_Res(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="Net_first_Res") as scope:
            super(Net_first_Res, self).__init__()

    
            
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()


                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()


            self.conv1a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2a = tf.keras.layers.BatchNormalization()
            
            self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2b = tf.keras.layers.BatchNormalization()


            self.conv2a_extra = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.conv2c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2c = tf.keras.layers.BatchNormalization()
            
            self.conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_2d = tf.keras.layers.BatchNormalization()





            self.conv3a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3a = tf.keras.layers.BatchNormalization()
            
            self.conv3b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3b = tf.keras.layers.BatchNormalization()


            self.conv3a_extra = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.conv3c = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3c = tf.keras.layers.BatchNormalization()
            
            self.conv3d = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_3d = tf.keras.layers.BatchNormalization()





            
            self.conv4a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4a = tf.keras.layers.BatchNormalization()
            
            self.conv4b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4b = tf.keras.layers.BatchNormalization()


            self.conv4a_extra = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))


            self.conv4c = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4c = tf.keras.layers.BatchNormalization()
            
            self.conv4d = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_4d = tf.keras.layers.BatchNormalization()


            self.conv5a = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5a = tf.keras.layers.BatchNormalization()
            
            self.conv5b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_5b = tf.keras.layers.BatchNormalization()

            self.flatten=tf.keras.layers.Flatten()


            #self.glob_average=tf.keras.layers.GlobalAveragePooling2D()

            self.conv_final_pre = tf.keras.layers.Dense(1024)

            self.conv_final = tf.keras.layers.Dense(8)

          
            
          

    def call(self, image,training=True):
       

        with tf.name_scope(name="Net_first_Res") as scope:


            tensor_pre=self.conv1_pre(image)
            
            tensor_pre=self.bn_1_pre(tensor_pre,training=training)


            
            tensor_1=self.conv1a(tensor_pre)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)

            tensor_1_add=self.conv1a_extra(tensor_pre)

            tensor_1=tensor_1+tensor_1_add

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=self.conv1c(tensor_1)
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)

            tensor_1=tf.nn.leaky_relu(tensor_1)

            tensor_1=tf.nn.max_pool(tensor_1, ksize=2, strides=2,padding='SAME')





            tensor_2=self.conv2a(tensor_1)
            
            tensor_2=self.bn_2a(tensor_2,training=training)
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2b(tensor_2)
            
            tensor_2=self.bn_2b(tensor_2,training=training)

            tensor_2_add=self.conv2a_extra(tensor_1)

            tensor_2=tensor_2+tensor_2_add

            tensor_2=tf.nn.leaky_relu(tensor_2)

            tensor_2=self.conv2c(tensor_2)
            
            tensor_2=self.bn_2c(tensor_2,training=training)
            
            tensor_2=tf.nn.leaky_relu(tensor_2)
            
            tensor_2=self.conv2d(tensor_2)
            
            tensor_2=self.bn_2d(tensor_2,training=training)

            tensor_2=tf.nn.leaky_relu(tensor_2)
      
            tensor_2=tf.nn.max_pool(tensor_2, ksize=2, strides=2,padding='SAME')




            tensor_3=self.conv3a(tensor_2)
            
            tensor_3=self.bn_3a(tensor_3,training=training)
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3b(tensor_3)
            
            tensor_3=self.bn_3b(tensor_3,training=training)

            tensor_3_add=self.conv3a_extra(tensor_2)

            tensor_3=tensor_3+tensor_3_add

            tensor_3=tf.nn.leaky_relu(tensor_3)

            tensor_3=self.conv3c(tensor_3)
            
            tensor_3=self.bn_3c(tensor_3,training=training)
            
            tensor_3=tf.nn.leaky_relu(tensor_3)
            
            tensor_3=self.conv3d(tensor_3)
            
            tensor_3=self.bn_3d(tensor_3,training=training)

            tensor_3=tf.nn.leaky_relu(tensor_3)
      
            tensor_3=tf.nn.max_pool(tensor_3, ksize=2, strides=2,padding='SAME')



            tensor_4=self.conv4a(tensor_3)
            
            tensor_4=self.bn_4a(tensor_4,training=training)
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4b(tensor_4)
            
            tensor_4=self.bn_4b(tensor_4,training=training)

            tensor_4_add=self.conv4a_extra(tensor_3)

            tensor_4=tensor_4+tensor_4_add

            tensor_4=tf.nn.leaky_relu(tensor_4)

            tensor_4=self.conv4c(tensor_4)
            
            tensor_4=self.bn_4c(tensor_4,training=training)
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv4d(tensor_4)
            
            tensor_4=self.bn_4d(tensor_4,training=training)

            tensor_4=tf.nn.leaky_relu(tensor_4)
      
            tensor_4=tf.nn.max_pool(tensor_4, ksize=2, strides=2,padding='SAME')

 

            tensor_4=self.conv5a(tensor_4)
            
            tensor_4=self.bn_5a(tensor_4,training=training)
            
            tensor_4=tf.nn.leaky_relu(tensor_4)
            
            tensor_4=self.conv5b(tensor_4)

            tensor_4=self.bn_5b(tensor_4,training=training)
            
            tensor_4=tf.nn.leaky_relu(tensor_4)


            tensor_4=self.flatten(tensor_4)

            tensor_4=self.conv_final_pre(tensor_4)

            tensor_4=tf.nn.dropout(tensor_4, rate=0.5)

            #self.glob_average=tf.keras.layers.GlobalAveragePooling2D()

            #tensor_1=self.glob_average(tensor_1)

            tensor_4=self.conv_final(tensor_4)


            return (tensor_4)









class ResNet_fourth_input(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="ResNet_fourth_input") as scope:
            super(ResNet_fourth_input, self).__init__()

              
                
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()


            self.conv1_forge = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            

            
 

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_fourth_template") as scope:
            
                      
            
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
                      

            return (tensor_1_total)


class ResNet_fourth_template(tf.keras.Model):
    def __init__(self):

        with tf.name_scope(name="ResNet_fourth_template") as scope:
            super(ResNet_fourth_template, self).__init__()

    
           
                
            self.conv1_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_pre = tf.keras.layers.BatchNormalization()
            
                
            self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1a = tf.keras.layers.BatchNormalization()
            
            self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1b = tf.keras.layers.BatchNormalization()
            
            
            
            self.conv1c = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1c = tf.keras.layers.BatchNormalization()
            
            self.conv1d = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1d = tf.keras.layers.BatchNormalization()



            self.conv1e = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1e = tf.keras.layers.BatchNormalization()
            
            self.conv1f = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1f = tf.keras.layers.BatchNormalization()



            self.conv1_aft = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))
            
            self.bn_1_aft = tf.keras.layers.BatchNormalization()


            self.conv1_forge = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1),padding="same",kernel_regularizer=tf.keras.regularizers.l2(l=0.05))

            
            
            

            
 

    def call(self, image,training=True):
       

        with tf.name_scope(name="ResNet_fourth_template") as scope:
            
                      
            
            tensor_1=self.conv1_pre(image)

            tensor_1_total=self.bn_1_pre(tensor_1) 


            
            tensor_1=self.conv1a(tensor_1_total)
            
            tensor_1=self.bn_1a(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1b(tensor_1)
            
            tensor_1=self.bn_1b(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)
            
            
            tensor_1=self.conv1c(tensor_1_total) 
            
            tensor_1=self.bn_1c(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1d(tensor_1)
            
            tensor_1=self.bn_1d(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)


            tensor_1=self.conv1e(tensor_1_total) 
            
            tensor_1=self.bn_1e(tensor_1,training=training)
            
            tensor_1=tf.nn.leaky_relu(tensor_1)
            
            tensor_1=self.conv1f(tensor_1)
            
            tensor_1=self.bn_1f(tensor_1,training=training)
            
            tensor_1_total=tensor_1+tensor_1_total
            
            tensor_1_total=tf.nn.leaky_relu(tensor_1_total)




            tensor_1_total=self.conv1_aft(tensor_1_total)

            tensor_1_total=self.bn_1_aft(tensor_1_total,training=training) 

            tensor_1_total=self.conv1_forge(tensor_1_total)
                      

            return (tensor_1_total)
'''






