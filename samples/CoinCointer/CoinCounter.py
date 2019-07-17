import os
import sys
import json
import datetime
import numpy as np
import tensorflow as tf
import skimage.draw
import skimage.color
import skimage.io
import glob





print("OS.name:", os.name)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print("Root dir:", ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
print("Found local version of the library\n")

print("Importing...\n")
print("Importing tensorflow client...")
from tensorflow.python.client import device_lib
print("Successfully imported tensorflow client...")
print("Importing mrcnn config")
from mrcnn.config import Config
print("Successfully imported mrcnn config...")
print("Importing mrcnn model")
from mrcnn import model as modellib, utils
print("Successfully imported mrcnn model...")
print("Importing mrcnn visualize")
from mrcnn import visualize
print("Successfully imported mrcnn visualize...")


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print("the coco weights:", COCO_WEIGHTS_PATH)

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
print("default log path:", DEFAULT_LOGS_DIR)

############################################################
#  Configurations
############################################################


# *********************************************************************************************************** #
#                                 HOW TO TRAIN THE MODEL WITH COCO WEIGHTS                                    #
# python3 samples/CoinCointer/CoinCounter.py train --dataset=datasets/coin/ --weights=coco --logs=logs/CoinCounterLogs
# *********************************************************************************************************** #
# /groups/hachgrp/projects/dev-image-segmentation/scripts/JG_Mask_RCNN_Blueprint/samples/CoinCointer/CoinCounter.py train --dataset=/groups/hachgrp/projects/dev-image-segmentation/scripts/JG_Mask_RCNN_Blueprint/datasets/coin/ --weights=coco --logs=/groups/hachgrp/projects/dev-image-segmentation/scripts/JG_Mask_RCNN_Blueprint/logs/CoinCounterLogs
# # *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                      HOW TO CONTINUE TRAINING THE MODEL                                     #
#                               sudo python3 samples/CoinCointer/CoinCounter.py train
#                     --dataset=datasets/coin/ --weights=last --logs=logs/CoinCounterLogs
# *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                      HOW TO RUN INFERENCE ON A TRAINED MODEL                                #
# python3 samples/CoinCointer/CoinCounter.py inference --image=datasets/coin/val --weights=mask_rcnn_coin_0030.h5
# *********************************************************************************************************** #


class CoinConfig(Config):
    """Configuration for training on the coin dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "coin"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + penny + nickle + dime + quarter + loonie + toonie

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    LEARNING_RATE = 0.0009
    LEARNING_MOMENTUM = 0.9

    DETECTION_MIN_CONFIDENCE = 0.7

    BACKBONE = "resnet101"


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print("The available devices are", get_available_devices())


############################################################
#  Data-set
############################################################


class CoinDataset(utils.Dataset):

    def load_coin(self, dataset_dir, subset):
        """Load a subset of the coin dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Add classes. We have 6 classes to add.
        self.add_class("Coin", 1, "penny")
        self.add_class("Coin", 2, "nickel")
        self.add_class("Coin", 3, "dime")
        self.add_class("Coin", 4, "quarter")
        self.add_class("Coin", 5, "loonie")
        self.add_class("Coin", 6, "toonie")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #         #   'regions': {
        #         #       '0': {
        #         #           'region_attributes': {},
        #         #           'shape_attributes': {
        #         #               'all_points_x': [...],
        #         #               'all_points_y': [...],
        #         #               'name': 'polygon'}},
        #         #       ... more regions ...
        #         #   },
        #         #   'size': 100202
        #         # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            coin_variant = []
            region_annotations = a['regions']
            for i in range(len(region_annotations)):
                coin_code = region_annotations[i]['region_attributes']['Coin']

                if coin_code == "Toonie":
                    coin_variant.append(6)
                elif coin_code == "Loonie":
                    coin_variant.append(5)
                elif coin_code == "Quarter":
                    coin_variant.append(4)
                elif coin_code == "Dime":
                    coin_variant.append(3)
                elif coin_code == "Nickel":
                    coin_variant.append(2)
                elif coin_code == "Penny":
                    coin_variant.append(1)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the data set is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)

            height, width = image.shape[:2]
            self.add_image(
                "Coin",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                coinVariant=coin_variant)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a coin dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Coin":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        # *************************************************************** #
        #    IMPLEMENT A METHOD TO CHECK IF ANNOTATION IS OF ANY SHAPE    #
        # *************************************************************** #

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1

            rr, cc = self.check_shape_of_annotation(p)
            if self.debug_polygons(p, rr, cc, i, mask, info):
                mask[rr, cc, i] = 1

        # *************************************************************** #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # *************************************************************** #

        # Return mask, and array of class IDs of each instance. We have six
        # possible instances, so we give the class ID's the list of coin variants in each image
        return mask.astype(np.bool), np.array(image_info['coinVariant'])

    # If there is an issue of with one or more photos in a dataset, this will ensure it does not crash the program
    def debug_polygons(self, p, rr, cc, i, mask, info):
        try:
            mask[rr, cc]
            return True
        except:
            if i == 0:
                print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                      "\n!!!!!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!!!!!"
                      "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("The mask length was", len(mask))
                print("The iteration was:", i)
                print("The length of rr was:", len(rr), "and the length of cc was:", len(cc))
                print("The image info was:", info)
            return False

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Coin":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

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

        return rr, cc

    # Calculates the total value of the coins found in the photo
    def calculate_total_value(self, coins):
        total_value = 0

        for coinCode in coins:
            if coinCode == 1:
                total_value += 0.01
            elif coinCode == 2:
                total_value += 0.05

            elif coinCode == 3:
                total_value += 0.1

            elif coinCode == 4:
                total_value += 0.25

            elif coinCode == 5:
                total_value += 1

            elif coinCode == 6:
                total_value += 2

        return total_value

    def get_class_names(self):
        class_name_list = []
        print("Classes:")
        for obj in self.class_info:
            print("- ", obj["name"])
            name = obj["name"]
            class_name_list.append(name)

        return class_name_list

    def load_inference_classes(self):

        # This loads in the classes for inference only
        self.add_class("Coin", 1, "penny")
        self.add_class("Coin", 2, "nickel")
        self.add_class("Coin", 3, "dime")
        self.add_class("Coin", 4, "quarter")
        self.add_class("Coin", 5, "loonie")
        self.add_class("Coin", 6, "toonie")


class OptimizeHyperparametersConfig(Config):

    """Creates a config for the process of hyperparameter optimization"""

    STEPS_PER_EPOCH = 2
    VALIDATION_STEPS = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + penny + nickle + dime + quarter + loonie + toonie

    def set_params(self, hyperparameters, index):

        """Sets the tunable hyperparameters"""
        self.NAME = "V-" + str(index)

        # Get the learning rate
        lr_min, lr_max = hyperparameters["lr"][0], hyperparameters["lr"][1]
        self.LEARNING_RATE = np.random.uniform(lr_min, lr_max)

        # Get the learning momentum
        lm_min, lm_max = hyperparameters["lm"][0], hyperparameters["lm"][1]
        self.LEARNING_MOMENTUM = np.random.uniform(lm_min, lm_max)

        # Get the weight decay
        wd_min, wd_max = hyperparameters["wd"][0], hyperparameters["wd"][1]
        self.WEIGHT_DECAY = np.random.uniform(wd_min, wd_max)


def optimize_hyperparameters(log_path, benchmark_model, num_of_cylces=30, epochs=1):
    """Giving a range of values, this function uses random search to approximate the optimal
        hyperparameters for a giving RCNN. The benchmark model is the initial config.
        Therefor; the first model tested will be using the hyperparameters specified
        by the user. The epochs and steps, however; will be normalized.
    """

    config_list = []

    learning_rate_range = [0.0005, 0.002]
    learning_momentum_range = [0.5, 0.99]
    weight_decay_range = [0.00007, 0.00014]

    hyperparameter_dict = {"lr": learning_rate_range, "lm": learning_momentum_range, "wd": weight_decay_range}

    # Sets the certain values to the specified config's values.
    config_hpo = OptimizeHyperparametersConfig()
    config_hpo.IMAGES_PER_GPU = benchmark_model.config.IMAGES_PER_GPU
    config_hpo.NUM_CLASSES = benchmark_model.config.NUM_CLASSES

    benchmark_model.config.STEPS_PER_EPOCH = config_hpo.STEPS_PER_EPOCH
    benchmark_model.config.NAME = "Benchmark"

    model_hpo = benchmark_model

    for index in range(num_of_cylces):

        """Train the model."""
        # Training dataset.
        dataset_train = CoinDataset()
        dataset_train.load_coin(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CoinDataset()
        dataset_val.load_coin(args.dataset, "val")
        dataset_val.prepare()

        print("************")
        print("Name:", model_hpo.config.NAME)
        print("lr:", model_hpo.config.LEARNING_RATE)
        print("lm:", model_hpo.config.LEARNING_MOMENTUM)
        print("wd:", model_hpo.config.WEIGHT_DECAY)
        print("")

        """ np print options
        {'edgeitems': 3, 'threshold': 1000, 'floatmode': 'maxprec', 'precision': 8, 'suppress': False, 
        'linewidth': 75, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-', 'formatter': None, 'legacy': False   """

        print("Training network heads of", index)
        model_hpo.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=epochs,
                        layers='heads')

        loss = model_hpo.keras_model.total_loss

        print("\n\n\n\n******************************************************************")
        print("loss type", type(loss))
        session = tf.Session()
        print(session.run(loss))
        print("\n\n\n\n")

        x = 5/0

        loss_config_name = (loss, model_hpo.config, model_hpo.config.NAME)
        config_list.append(loss_config_name)

        config_hpo.set_params(hyperparameter_dict, index)

        model_hpo = modellib.MaskRCNN(mode="training", config=config_hpo,
                                      model_dir=log_path)

    opt_hyperparameters = config_list[0]
    for c in config_list:
        loss, con, name = c
        if loss < opt_hyperparameters[0]:
            opt_hyperparameters = c
    print("The optimal hyperparameters are approximately", opt_hyperparameters[2])
    print("With a loss of", opt_hyperparameters[0])
    print("The config was", opt_hyperparameters[3])


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CoinDataset()
    dataset_train.load_coin(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CoinDataset()
    dataset_val.load_coin(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
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
                layers="all")


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Inference
############################################################


# Import my config
sys.path.append(os.path.join(ROOT_DIR, "samples/CoinCoiter/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
CoinCounter_MODEL_PATH = os.path.join(ROOT_DIR, "logs/CoinCounterLogsTwo__LowerLR_and_LowerMMNTM/coin20190710T2036"
                                                "mask_rcnn_coin_0030.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/coin/val")

# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


def define_path(file_path, model_inf, class_names):
    if os.path.isdir(file_path):
        inference_dir(file_path, model_inf, class_names)
    else:
        inference_image(file_path, model_inf, class_names)


def inference_image(file_path, model_inf, class_names):
    im = skimage.io.imread(file_path)
    results = model_inf.detect([im], verbose=1)
    r = results[0]
    visualize.display_instances(im, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], figsize=(8, 8))


def inference_dir(file_path, model_inf, class_names):
    im_list = []
    r_list = []
    for path in glob.iglob(pathname=file_path + "/*.jpg"):
        im = skimage.io.imread(path)
        im_list.append(im)

        results = model_inf.detect([im], verbose=1)
        r_list.append(results[0])

    i = 0
    for r in r_list:
        visualize.display_instances(im_list[i], r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], figsize=(8, 8))
        i += 1


def inference(path, model_inf):

    # My Class names
    inference_dataset = CoinDataset()
    inference_dataset.load_inference_classes()
    class_names = inference_dataset.get_class_names()

    define_path(path, model_inf, class_names)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect coins.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'inference' or optimizeHP")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/coin/dataset/",
                        help='Directory of the coin dataset')
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
    parser.add_argument('--epochs', required=False,
                        metavar="amount of epochs for hyperparameter optimization",
                        help='Video to apply the color splash effect on')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "inference":
        assert args.image,\
               "Provide --image=(image path) or (directory path)"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CoinConfig()
    else:
        class InferenceConfig(CoinConfig):
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
    elif args.command == "optimizeHP":
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
            utils.download_trained_weights(weights_path+"/raw-data")
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
    elif args.command == "inference":
        inference(model_inf=model, path=args.image)
    elif args.command == "optimizeHP":
        optimize_hyperparameters(log_path=args.logs, benchmark_model=model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
