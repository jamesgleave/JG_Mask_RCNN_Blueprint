# Blueprint for Mask R-CNN for Object Detection and Segmentation

This is an implementation to matterport's [Mask_RCNN repository](https://github.com/matterport/Mask_RCNN). Using python3, Keras, and tensorflow, this model detects, segments and builds bounding boxes around objects. 

![Instance Segmentation Sample](assets/street.png)

The repository contains: 
  - All files contained in the [parent repository](https://github.com/matterport/Mask_RCNN)
  - An added "CoinCounter" class, which counts the total value of coins in a photo
  - A "Blueprint" directory, which contains a blackbox blueprint_class and a blackbox blueprint_inspect_data .py file.
 
# About the blueprint

The goal of this repository is to create system for generalized and easily configurable object detection and segmentation. The blueprint class streamlines the processes of training, testing and inference.


# Getting started
-0. Clone this repository.

-1. Download [the COCO weights](https://www.dropbox.com/s/acoxck93wmuq151/mask_rcnn_coco.h5?dl=0). You can optionally download the [coin and balloon dataset](https://www.dropbox.com/sh/5ywnb1788fhlrps/AAAoj0S2gjoKf6am077DQ_UAa?dl=0) as well. Place the coco weights in your repository and create a directory called "datasets" for the datasets.

-2. Create a "logs" file to save training progress (weights after every epoch and scalers for tensorboard). 

-3. Install all of the dependencies > pip3 install -r requirements

-4. Run the setup.py file > python3 setup.py install

![Instance Segmentation Sample](assets/balloon_color_splash.gif)

