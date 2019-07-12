import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.color
import skimage.io
import skimage.transform as skit

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# *Make sure you create a "logs" directory*!
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

"                       It is easiest to run these commands within the project folder                         "
# *********************************************************************************************************** #
#                                 HOW TO TRAIN THE MODEL WITH COCO WEIGHTS                                    #
#        sudo python3 samples/Blueprint/(Name).py train --dataset=datasets/(YourDataset)/ --model=coco
# *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                 HOW TO TRAIN THE MODEL WITH ImageNet WEIGHTS                                #
#        sudo python3 samples/Blueprint/(Name).py train --dataset=datasets/(YourDataset)/ --model=imagenet
# *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                      CONTINUE TRAINING THE LAST MODEL                                       #
#       sudo python3 samples/Blueprint/(Name).py train --dataset=datasets/(YourDataset)/ --weights=last
# *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                                TO LOG PROGRESS                                              #
#                                            --logs=logs/(NameOfLogs)
# *********************************************************************************************************** #


############################################################
#  Configurationsf
############################################################


class BlueprintConfig(Config):
    """Configuration for training on a dataset.
        Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Your Name"
#   .
#   .
#   .

    """Look at the parent class to see all possible hyper parameters"""


############################################################
#  Data-set
############################################################

class BlueprintDataset(utils.Dataset):

    def load_blueprint(self, dataset_dir, subset):
        """Load a subset of the blueprint dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val


        # Add classes. Include all classes you want to detect.
        self.add_class("Source", "ID (this should be an int)", "Name")

        self.add_image(
            "Source",
            image_id=a['filename'],  # use file name as a unique image id
            path=image_path,
            width=width, height=height,
            polygons=polygons)
        """

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

    def debug_polygons(self, p, rr, cc, i, mask, info):
        """ If there is an issue of with one or more photos in a dataset,
        this will ensure it does not crash the program"""


    def image_reference(self, image_id):
        """Return the path of the image."""

    # Checks which polygon was used in tagging the photo and returns the appropriate points
    def check_shape_of_annotation(self, p=None):
        if p['name'] == 'polygon':
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        if p['name'] == 'polyline':
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        if p['name'] == 'circle':
            rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
        if p['name'] == 'ellipse':
            rr, cc = skimage.draw.ellipse(p['cx'], p['cy'], r_radius=p['rx'], c_radius=p['ry'], rotation=p['theta'])
    #   .
    #   .
    #   . etc

        return rr, cc

    def get_class_names(self):
        """Returns a list of the names of the classes were created for detection"""

    def load_inference_classes(self):
        # This loads in the classes for inference only
        # Copy your class names here.

        # self.add_class("Source", 1, "name")
        # self.add_class("Source", 2, "name")
        # self.add_class("Source", 3, "name")
        # etc...
        return


def train(model):
    """Train the model.
    # Training dataset.
    dataset_train = BlueprintDataset()
    dataset_train.load_blueprint(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BlueprintDataset()
    dataset_val.load_blueprint(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Training all network nodes")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all") """


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """


def detect_and_color_splash(model, image_path=None, video_path=None):
    """Applies colour splash and detects objects in a video or image"""

############################################################
#  Inference
############################################################


# Load a random image from the images folder
def define_path(file_path, model_inf, class_names):
    if os.path.isdir(file_path):
        inference_dir(file_path, model_inf, class_names)
    else:
        inference_image(file_path, model_inf, class_names)



def inference_image(file_path, model_inf, class_names):
    """
    If an image is sent for inference this function is used.
    It reads a single image."""


def inference_dir(file_path, model_inf, class_names):
    """If a directory's path is sent for inference, this function will be called.
        It will preform inference on every image inside the given directory."""


def inference(path, model_inf):
    """"""



############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Your description')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/your/dataset/",
                        help='Directory of the your dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BlueprintConfig()
    else:
        class InferenceConfig(BlueprintConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
