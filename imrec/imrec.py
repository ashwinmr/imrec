import argparse

def sqr(n):
  """ Find square of a number
  """
  return n*n

def parse_args():
  """ Parse arguments for program
  """
  parser = argparse.ArgumentParser(description="Program for recommending images")

  subparsers = parser.add_subparsers()

  # Score parser
  score_parser = subparsers.add_parser('score', help='obtain score of image')
  score_parser.add_argument('image_path', help='path to image file')
  score_parser.set_defaults(func=score)

  return parser.parse_args()

def score(args):
  """ Function to determine score of an image
  """

  return 10

def main():

  args = parse_args()
  print(args.func(args))

if __name__ == "__main__":
  main()