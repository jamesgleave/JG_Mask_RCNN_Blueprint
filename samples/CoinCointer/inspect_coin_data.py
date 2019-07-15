"""
Running this will show samples of the data set.
The samples will have bounding boxes.
"""

import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.CoinCointer import CoinCounter


# Configuration:
config = CoinCounter.CoinConfig
COIN_DIR = os.path.join(ROOT_DIR, "datasets/coin")


# **************************************************************************** #
# / // // // // // // // // // // // // // // // // // // // // // // // // /
# Data set.
# \ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \
# **************************************************************************** #


# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = CoinCounter.CoinDataset()
dataset.load_coin(COIN_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# **************************************************************************** #
# / // // // // // // // // // // // // // // // // // // // // // // // // /
# Displaying the samples and bounding boxes
# \ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \
# **************************************************************************** #


def display_dataset(num_of_random_samples):
    # Load and display random samples
    if num_of_random_samples >= len(dataset.image_ids):
        print("The number of samples cannot be larger than the amount of samples available")
        print("\nSetting the amount of equal to the amount of samples")
        num_of_random_samples = len(dataset.image_ids) - 1

    image_ids = np.random.choice(dataset.image_ids, num_of_random_samples)

    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

    # Load random image and mask.
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


display_dataset(3)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Inspecting Data...')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'inspect'")
    parser.add_argument('nsi', required=True,
                        metavar="N", type=int,
                        help='Number of samples to inspect (int)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "inspect":
        assert args.dataset, "Argument --nsi (number of samples to inspect) is required for inspecting data"

    print("Dataset: ", args.dataset)

    # Configurations
    if args.command == "inspect":
        display_dataset(args.nsi)
