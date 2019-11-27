from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DataSet(object):
  def __init__(self, images_list_path, num_epoch, batch_size):
    # filling the record_list
    #Read from the list of file names to images
    input_file = open(images_list_path, 'r')
    self.record_list = []
    #making a list of filenames
    for line in input_file:
      line = line.strip()
      self.record_list.append(line)
    #generates a queue of image finapaths for the number of epochs mentioned
    filename_queue = tf.train.string_input_producer(self.record_list, num_epochs=num_epoch)
    image_reader = tf.WholeFileReader()
    #reads each file from the queue
    _, image_file = image_reader.read(filename_queue)
    #decode the jpeg image with 3 channels (RGB)
    image = tf.image.decode_jpeg(image_file, 3)
    #preprocess
    #resize to high resolution image for ground truth
    hr_image = tf.image.resize_images(image, [32, 32])
    #resize to low resolution image for input to the network
    lr_image = tf.image.resize_images(image, [8, 8])
    #convert the images into float tensors
    hr_image = tf.cast(hr_image, tf.float32)
    lr_image = tf.cast(lr_image, tf.float32)
    #minimum number of images in the queue after 1 dequeue operation
    min_after_dequeue = 1000
    #max capacity of the queue
    capacity = min_after_dequeue + 400 * batch_size
    self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
