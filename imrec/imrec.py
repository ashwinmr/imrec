import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

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

def create_img_array(img_paths, size):
  """ Function to create an array of images of same size
  """
  imgs = []
  for img_path in img_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = resize_image(img,100)
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

def score(args):
  """ Function to determine score of an image
  """

  return 10

def main():

  args = parse_args()
  print(args.func(args))

if __name__ == "__main__":
  main()
