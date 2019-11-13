# imrec
A command line tool for recommending images

## Features
-   Score an image based on a learned model
-   Train a model using a dataset of scored images
-   Create a dataset of scored images
-   View a dataset of scored images

# Install

```
git clone git@github.com:ashwinmr/imrec.git
cd imrec
sudo -H pip install -r requirements.txt
sudo -H python setup.py install
```

# Usage

imrec allows subcommands for performing different functions

## Create a dataset using images

You can provide imrec a directory of images.  
It will then plot each image in the directory and ask for a score from 0 to 10.  
The dataset will then be saved to a provided path.  
```
imrec load <path/to/images/dir> <path/to/save/dataset.p>
```

You can view the resulting dataset
```
imrec view <path/to/saved/dataset.p>
```

## Train a model using a dataset

You can use the created dataset to train an ML model.  
The model will then be saved to a provided h5 file.  
```
imrec train <path/to/dataset.p> <path/to/save/model.h5>
```

You can evaluate the model against a dataset
```
imrec eval <path/to/dataset.p> <path/to/model.h5>
```

## Score an image

You can use the trained model to score an input image.  
```
imrec score <path/to/image> <path/to/model.h5>
```

# Testing
```
pytest
```
