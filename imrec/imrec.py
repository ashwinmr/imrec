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

  # Train parser
  train_parser = subparsers.add_parser('train', help='train a model using a dataset')
  train_parser.add_argument('dataset_path', help='path to dataset')
  train_parser.add_argument('model_path', help='path to save trained model')
  train_parser.set_defaults(func=train)

  # Load parser
  load_parser = subparsers.add_parser('load', help='load data into a dataset')
  load_parser.add_argument('image_dir', help='directory of images')
  load_parser.add_argument('dataset_path', help='path to save dataset')
  load_parser.set_defaults(func=load)

  # View parser
  view_parser = subparsers.add_parser('view', help='view data in a dataset')
  view_parser.add_argument('dataset_path', help='path to dataset')
  view_parser.set_defaults(func=view)

  # Eval parser
  eval_parser = subparsers.add_parser('eval', help='eval a model using a dataset')
  eval_parser.add_argument('dataset_path', help='path to dataset')
  eval_parser.add_argument('model_path', help='path to saved model')
  eval_parser.set_defaults(func=eval)

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

    if img is None:
      continue

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

def get_training_set(dataset_path):
  """ get training set from the dataset file
  """

  # Load dataset
  data = pickle.load(open(dataset_path,"rb"))

  imgs = data['images']
  scores = data['scores']

  # Normalize images
  x_train = normalize_images(imgs)

  # Scale scores
  y_train = scale_scores(scores)

  return x_train, y_train

def train_model(x_train,y_train, save_path):
  """ Create a model and train it
  """

  # Create ML model
  model = Sequential()
  model.add(Flatten(input_shape=x_train[0].shape))
  model.add(Dense(1,activation = 'sigmoid'))

  # Train the model
  model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=3, validation_split = 0.2)

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

  print(scores[0][0])

  return

def train(args):
  """ Function to train a model with data
  """

  dataset_path = args.dataset_path
  model_path = args.model_path

  # Get training set
  x_train, y_train = get_training_set(dataset_path)

  # create and train the model
  train_model(x_train, y_train, model_path)

  return

def eval(args):
  """ evaluate the model in test mode using a dataset
  """

  dataset_path = args.dataset_path
  model_path = args.model_path

  # Load the trained model
  model = load_model(model_path)

  # Get data set
  x_train, y_train = get_training_set(dataset_path)

  # Evluate the model
  metrics = model.evaluate(x_train, y_train)

  # Display results
  for metric_i in range(len(model.metrics_names)):
      metric_name = model.metrics_names[metric_i]
      metric_value = metrics[metric_i]
      print('{}: {}'.format(metric_name, metric_value))

  return

def load(args):
  """ Load images into a dataset
  """

  image_dir = args.image_dir
  dataset_path = args.dataset_path

  # Get images
  img_paths = get_paths_from_dir(image_dir)
  imgs = create_img_array(img_paths,100)

  # Get scores
  scores = []
  for img in imgs:
    try:
      scores.append(get_score_for_image(img))
    except ValueError:
      imgs = imgs[0:len(scores)]
      break

  scores = np.array(scores)

  # Create dictionary of images and scores
  data = {'images':imgs,'scores':scores}

  # Save the data
  pickle.dump(data,open(dataset_path,"wb"))

  return

def view(args):
  """ View data in a dataset
  """

  dataset_path = args.dataset_path

  # Load dataset
  data = pickle.load(open(dataset_path,"rb"))

  imgs = data['images']
  scores = data['scores']

  # Create grid
  size = len(imgs)
  cols = math.ceil(np.sqrt(size))
  rows = math.ceil(size/cols)

  # Plot all images
  f, axarr = plt.subplots(rows,cols)

  for row in range(rows):
    for col in range(cols):
      idx = cols*row + col
      ax = axarr[row,col]
      ax.axis('off')
      if idx < size:
        ax.imshow(imgs[idx])
        ax.set_title(scores[idx],fontsize='x-small')

  plt.show()

  return

def main():

  args = parse_args()
  args.func(args)

if __name__ == "__main__":
  main()
