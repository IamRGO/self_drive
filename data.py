import tensorflow as tf
import cv2
import numpy as np

def read_output(file_path):
  file = open(file_path, 'r')
  data = file.read()
  file.close()

  steering, throttle = data.split(":")
  steering = int(steering) # range 40, 131
  steering = np.interp(steering, [40, 130], [-1.0, 1.0])

  # throttle = int(throttle) # range 90, 180
  # throttle = np.interp(throttle, [90, 150], [0.0, 1.0])

  return [steering]

def read_input(file_path):
  image = cv2.imread(file_path)
  img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
  img_gray = tf.image.rgb_to_grayscale(img_tensor)
  img_resized = tf.image.resize_with_pad(img_gray, 80, 60)
  return img_resized

def parse_image(image):
  rgb_image = mask_image(image)
  img_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.float32)
  img_gray = tf.image.rgb_to_grayscale(img_tensor)
  img_resized = tf.image.resize_with_pad(img_gray, 80, 60)
  return img_resized

def mask_image(image):
  image = cv2.resize(image, (80, 60))

  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  light_yellow = np.array([20, 50, 140])
  dark_yellow = np.array([100, 200, 250])

  mask = cv2.inRange(hsv, light_yellow, dark_yellow)
  result = cv2.bitwise_or(image, image, mask = mask)

  result[mask == 255] = [255, 255, 255]
  rgb_image = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

  return rgb_image