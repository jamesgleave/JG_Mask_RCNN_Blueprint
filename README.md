# Blueprint for Mask R-CNN for Object Detection and Segmentation

This is an implementation to matterport's [Mask_RCNN repository](https://github.com/matterport/Mask_RCNN). Using python3, Keras, and tensorflow, this model detects, segments and builds bounding boxes around objects. 

![Instance Segmentation Sample](assets/street.png)

The repository contains: 
  - All files contained in the [parent repository](https://github.com/matterport/Mask_RCNN)
  - An added "CoinCounter" class, which counts the total value of coins in a photo
  - A "Blueprint" directory, which contains a blackbox blueprint_class, a blackbox blueprint_inspect_model and a blackbox           blueprint_inspect_data .py file.
 
# About the blueprint

The goal of this repository is to create system for generalized and easily configurable object detection and segmentation. The blueprint class streamlines the processes of training, testing and inference.

![Instance Segmentation Sample](assets/balloon_color_splash.gif)

