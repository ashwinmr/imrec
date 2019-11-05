# Description

imrec takes an image file as input and gives a score out of 10 for the image.
It uses machine learning to figure out what features of an image the user likes.
The learning happens using a database of images and scores.

# Features

-   score a single image
-   load a directory of images and obtain user score for them for learning
-   use pickle to store the learned model

# Usage

-   imrec score <path/to/image/file>

-   imrec load <path/to/directory/of/images>
