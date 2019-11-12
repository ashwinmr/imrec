import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.models import load_model
import os
import pickle

def sqr(n):
  """ Find square of a number
  """
  return n*n

def parse_args():
  """ Parse arguments for program
  """
  parser = argparse.ArgumentParser(description="Program for recommending images")

  subparsers = parser.add_subparsers(required=True, dest='sub_command')

  # Score parser
  score_parser = subparsers.add_parser('score', help='obtain score of image')
  score_parser.add_argument('image_path', help='path to image file')
  score_parser.add_argument('model_path', help='path to trained model')
  score_parser.set_defaults(func=score)

  return parser.parse_args()

def resize_image(img, size):
  """ Function to resize smaller dimension of image to the input size.
  Keep aspect ratio.
  """

  height, width = img.shape[:2]

  # Resize smaller dimension to input size and keep aspect ratio
  if height < width:
    img_res = cv2.resize(img,(math.floor(size*(width/height)),size))
  else:
    img_res = cv2.resize(img,(size,math.floor(size*(height/width))))

  return img_res

def crop_center_square(img):
  """ Function to crop an image from center into a square of smaller dimension
  """

  height, width = img.shape[:2]

  if height < width:
    start = math.floor((width-height)/2)
    end = start + height
    img_crp = img[:,start:end,:]
  else:
    start = math.floor((height-width)/2)
    end = start + width
    img_crp = img[start:end,:,:]

  return img_crp

def get_paths_from_dir(directory):
  """ Function to get file paths from a directory
  """
  paths = []
  for file in os.listdir(directory):
    paths.append(os.path.join(directory,file))

  return paths

def create_img_array(img_paths, size = 100):
  """ Function to create an array of images of same size
  """
  imgs = []
  for img_path in img_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = resize_image(img,size)
    img = crop_center_square(img)
    imgs.append(img)

  imgs = np.array(imgs)

  return imgs

def get_score_for_image(img):
  """ Function to obtain the score for an image from user input
  """
  plt.imshow(img)
  plt.show()
  score = int(input("Enter score out of 10 for image:\n"))
  return score

def create_data_set(directory, save_path = 'temp/dataset.p'):
  """ Function to create a dataset from images in a directory
  """

  # Get images
  img_paths = get_paths_from_dir(directory)
  imgs = create_img_array(img_paths,100)

  # Get scores
  scores = []
  for img in imgs:
    scores.append(get_score_for_image(img))

  scores = np.array(scores)

  # Create dictionary of images and scores
  data = {'images':imgs,'scores':scores}

  # Save the data
  pickle.dump(data,open(save_path,"wb"))

  return

def get_dataset_indexes(size, split = [60,20,20]):
  """ get random indexes corresponding to training, validation and test sets
  for a given dataset size
  """
  idx = np.arange(size)
  np.random.shuffle(idx)

  train_end = math.floor(split[0]*size/100)
  val_end = math.floor((split[0]+split[1])*size/100)

  idx_train = idx[0:train_end]
  idx_val = idx[train_end:val_end]
  idx_test = idx[val_end::]

  return idx_train, idx_val, idx_test

def get_datasets(dataset_path):
  """ get trainging, validation and test datasets from the dataset file
  """

  # Load dataset
  data = pickle.load(open(dataset_path,"rb"))

  imgs = data['images']
  scores = data['scores']

  # Create datasets
  size = len(imgs)
  idx_train, idx_val, idx_test = get_dataset_indexes(size)

  x_train = imgs[idx_train] / 255.0 - 0.5
  y_train = scores[idx_train] / 10.0

  x_val = imgs[idx_val] / 255.0 - 0.5
  y_val = scores[idx_val] / 10.0

  x_test = imgs[idx_test] / 255.0 - 0.5
  y_test = scores[idx_test] / 10.0

  return x_train, y_train, x_val, y_val, x_test, y_test

def train_model(x_train,y_train, save_path = 'temp/model.h5'):

  # Create ML model
  model = Sequential()
  model.add(Flatten(input_shape=x_train[0].shape))
  model.add(Dense(1,activation = 'sigmoid'))

  # Train the model
  model.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
  model.fit(x_train, y_train)

  # Save the model
  model.save(save_path)

def normalize_images(imgs):
  """ Normalize images for learning
  """

  imgs_norm = imgs/255.0 - 0.5

  return imgs_norm

def denormalize_images(imgs_norm):
  """ De normalize images for plotting
  """

  imgs = (imgs_norm + 0.5) * 255.0

  return imgs

def scale_scores(scores):
  """ Scale the scores for learning
  """

  scores_scl = scores / 10.0

  return scores_scl

def descale_scores(scores_scl):
  """ De scale the scores for output
  """

  scores = scores_scl * 10.0

  return scores

def score(args):
  """ Function to determine score of an image
  """

  image_path = args.image_path
  model_path = args.model_path

  # Load the trained model
  model = load_model(model_path)

  # Load the image for ML model
  imgs = create_img_array([image_path])

  # Normalize the image
  imgs = normalize_images(imgs)

  # Predict
  pred = model.predict(imgs)

  # Descale output
  scores = descale_scores(pred)

  return scores[0][0]

def main():

  args = parse_args()
  print(args.func(args))

if __name__ == "__main__":
  main()
