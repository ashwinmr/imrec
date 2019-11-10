import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

def score(args):
  """ Function to determine score of an image
  """

  return 10

def main():

  args = parse_args()
  print(args.func(args))

if __name__ == "__main__":
  main()
