# DCGAN-Keras-Eager

from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf
tf.enable_eager_execution()

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

# Get dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# Reshape images
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

# We are normalizing the images to the range of [-1, 1]
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000 # N of elements
BATCH_SIZE = 256 

# Set dataset (all elementents in RAM memory)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Define Generator model
class Generator(tf.keras.Model):

  def __init__(self):
    super(Generator, self).__init__()
    # Define weights / learning parameters

    self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False) # Bias to false because the batch norm ignores it
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    self.conv1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    
    self.conv2 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm3 = tf.keras.layers.BatchNormalization()
    
    self.conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

  # Define call function (forward, or execution function)
  def call(self, x, training=True):
    x = self.fc1(x) # Using the model parameters
    x = self.batchnorm1(x, training=training)
    x = tf.nn.relu(x) # using other not learnable functions

    x = tf.reshape(x, shape=(-1, 7, 7, 64))

    x = self.conv1(x)
    x = self.batchnorm2(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.batchnorm3(x, training=training)
    x = tf.nn.relu(x)

    x = tf.nn.tanh(self.conv3(x))  
    return x

# Define the discriminator
class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
    self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
    self.dropout = tf.keras.layers.Dropout(0.3)
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(1)

  def call(self, x, training=True):
    x = tf.nn.leaky_relu(self.conv1(x))
    x = self.dropout(x, training=training)
    x = tf.nn.leaky_relu(self.conv2(x))
    x = self.dropout(x, training=training)
    x = self.flatten(x)
    x = self.fc1(x)
    return x


generator = Generator()
discriminator = Discriminator()


'''

The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones (since these are the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros (since these are the fake images)
Then the total_loss is the sum of real_loss and the generated_loss

It is a sigmoid cross entropy loss of the generated images and an array of ones

The discriminator and the generator optimizers are different since we will train them separately.
'''

def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

# Optimizers
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
generator_optimizer = tf.train.AdamOptimizer(1e-4)


EPOCHS = 150
noise_dim = 100
num_examples_to_generate = 100

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement of the gan.
random_vector_for_generation = tf.random_normal([num_examples_to_generate, noise_dim])

def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(10,10))
  
  for i in range(predictions.shape[0]):
      plt.subplot(10, 10, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
        
  # tight_layout minimizes the overlap between 2 sub-plots
  plt.tight_layout()
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

def train(dataset, epochs, noise_dim):  
  for epoch in range(epochs): # for every epoch
    start = time.time()
    
    for images in dataset: # for every batch

      # generating noise from a uniform distribution
      noise = tf.random_normal([BATCH_SIZE, noise_dim])
      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: # get two different gradientTape
        generated_images = generator(noise, training=True) #get the generate images 
      
        real_output = discriminator(images, training=True) # get outputs from real images
        generated_output = discriminator(generated_images, training=True) # get outputs from generated images
        
        # get losses
        gen_loss = generator_loss(generated_output) 
        disc_loss = discriminator_loss(real_output, generated_output)
        
      # get gradients
      gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
      
      # appy gradients to some learning variables
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

      
    if epoch % 10 == 0:
      generate_and_save_images(generator,
                               epoch + 1,
                               random_vector_for_generation)

    print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                      time.time()-start))
  # generating after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           random_vector_for_generation)




train(train_dataset, EPOCHS, noise_dim)


'''
with imageio.get_writer('dcgan.gif', mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  # this is a hack to display the gif inside the notebook
  os.system('mv dcgan.gif dcgan.gif.png')

  display.Image(filename="dcgan.gif.png")
'''