import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.color
import skimage.io
import keras
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

"""     It is easiest to run these commands within the project folder. Replace "private" with your own name        """
"""                          For an easier time using this code, use VGG image annotator                             """

# *********************************************************************************************************** #
#                                 HOW TO TRAIN THE MODEL WITH COCO WEIGHTS                                    #
#        sudo python3 private.py train --dataset=datasets/private_dataset/ --weights=coco
# *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                 HOW TO TRAIN THE MODEL WITH ImageNet WEIGHTS                                #
#        sudo python3 samples/ImageDetection/(Name).py train --dataset=datasets/(YourDataset)/ --weights=imagenet
# *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                      CONTINUE TRAINING THE LAST MODEL                                       #
#       sudo python3 samples/ImageDetection/(Name).py train --dataset=datasets/(YourDataset)/ --weights=last
# *********************************************************************************************************** #

# *********************************************************************************************************** #
#                                                TO LOG PROGRESS                                              #
#                                            --logs=logs/(NameOfLogs)
# *********************************************************************************************************** #

########################################################################################################################
#  Configurations
########################################################################################################################


class ImageDetectionConfig(Config):
    """Configuration for training on the private dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "ImageDetection"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # + (number of your classes)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.0009

    # Check the config class for more information

############################################################
#  Data-set
############################################################


class ImageDetectionDataset(utils.Dataset):

    def load_private(self, dataset_dir, subset):
        """Load a subset your dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Add classes.
        self.add_class("private", 1, "private")
        # self.add_class("Source", 2, "name")
        # self.add_class("Source", 3, "name")
        # etc...

        # Training or validation dataset?
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

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the data set is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)

            height, width = image.shape[:2]
            self.add_image(
                "private",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a private dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        if image_info["source"] != "private":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1

            rr, cc = self.check_shape_of_annotation(p)
            if self.debug_polygons(rr, cc, i, mask, info):
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. We have six
        # possible instances, so we give the class ID's the list of private variants in each image
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def debug_polygons(self, rr, cc, i, mask, info):
        """ If there is an issue of with one or more photos in a dataset,
        this will ensure it does not crash the program"""

        try:
            mask[rr, cc]
            return True
        except ArithmeticError:
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
        if info["source"] == "private":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def check_shape_of_annotation(self, p=None):
        """Checks which polygon was used in tagging the photo and returns the appropriate points"""
        rr, cc = None, None
        if p['name'] == 'polygon':
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        if p['name'] == 'polyline':
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        if p['name'] == 'circle':
            rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
        if p['name'] == 'ellipse':
            rr, cc = skimage.draw.ellipse(p['cx'], p['cy'], r_radius=p['rx'], c_radius=p['ry'], rotation=p['theta'])

        if rr is None:
            print("The annotation shape was not recognized")

        return rr, cc

    def get_class_names(self):
        """Returns a list of classes that were previously defined"""

        class_name_list = []
        print("Classes:")
        for obj in self.class_info:
            print("- ", obj["name"])
            name = obj["name"]
            class_name_list.append(name)

        return class_name_list

    def load_inference_classes(self):
        """This loads in the classes for inference only.
        Copy your class names here."""

        self.add_class("private", 1, "private")
        # self.add_class("Source", 1, "name")
        # self.add_class("Source", 2, "name")
        # self.add_class("Source", 3, "name")
        # etc...
        return


# *************************************************************** #
# Hyperparameter Optimization                                     #
# *************************************************************** #

class MaskRCNN(modellib.MaskRCNN):
    def __init__(self, mode, model_dir, config):
        super().__init__(mode=mode, config=config, model_dir=model_dir)
        self.history = keras.callbacks.History()

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):

        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = modellib.data_generator(train_dataset, self.config, shuffle=True,
                                                  augmentation=augmentation,
                                                  batch_size=self.config.BATCH_SIZE,
                                                  no_augmentation_sources=no_augmentation_sources)
        val_generator = modellib.data_generator(val_dataset, self.config, shuffle=True,
                                                batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        modellib.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        modellib.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = modellib.multiprocessing.cpu_count()

        self.history = self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)


class OptimizeHyperparametersConfig(Config):

    """Creates a config for the process of hyperparameter optimization"""

    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

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


def optimize_hyperparameters(benchmark_model, num_of_cylces=10, epochs=1):
    """Giving a range of values, this function uses random search to approximate the optimal
        hyperparameters for a giving RCNN. The benchmark model is the initial config.
        Therefor; the first model tested will be using the hyperparameters specified
        by the user. The epochs and steps, however; will be normalized.
    """

    config_list = []

    log_path = benchmark_model.model_dir

    learning_rate_range = [0.0005, 0.0015]
    learning_momentum_range = [0.81, 0.99]
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
        dataset_train = ImageDetectionDataset()
        dataset_train.load_private(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = ImageDetectionDataset()
        dataset_val.load_private(args.dataset, "val")
        dataset_val.prepare()

        print("************")
        print("Name:", model_hpo.config.NAME)
        print("lr:", model_hpo.config.LEARNING_RATE)
        print("lm:", model_hpo.config.LEARNING_MOMENTUM)
        print("wd:", model_hpo.config.WEIGHT_DECAY)
        print("")

        print("Training network heads of", index)

        model_hpo.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=epochs,
                        layers='heads')

        history = model_hpo.history
        loss = history.history['loss']
        loss_config_name = (loss, model_hpo.config, model_hpo.config.NAME)
        config_list.append(loss_config_name)

        config_hpo.set_params(hyperparameter_dict, index)

        print("Training", model_hpo.config.NAME, "Successful\n********************************************************"
              , "\nThe total loss was:", loss)

        model_hpo = MaskRCNN(mode="training", config=config_hpo,
                             model_dir=log_path)

        print("Now training", model_hpo.config.NAME, "\n\n")

    opt_hyperparameters = config_list[0]
    for c in config_list:
        loss, con, name = c
        print("Name:", name, "\nLoss:", loss, "\nConfig:", con, "\n***********************************************\n\n")
        if loss < opt_hyperparameters[0]:
            opt_hyperparameters = c
    print("The optimal hyperparameters are approximately", opt_hyperparameters[1])
    print("With a loss of", opt_hyperparameters[0])
    print("The config was", opt_hyperparameters[2])


# *************************************************************** #
# *************************************************************** #


def train(model):
    import imgaug
    """Train the model."""
    # Training dataset.
    dataset_train = ImageDetectionDataset()
    dataset_train.load_private(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ImageDetectionDataset()
    dataset_val.load_private(args.dataset, "val")
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 3 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=45,
                layers='3+',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 5 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='5+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='all',
                augmentation=augmentation)


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


def remove_background(image, mask):
    """Blacks out all objects that are not masked.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a blacked out copy of the image. The blacked out copy still
    # has 3 RGB channels, though.
    blackout = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 0

    blackout = np.array(blackout)
    image = np.array(image)

    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        augmented_image = np.where(mask, image, blackout).astype(np.uint8)
    else:
        augmented_image = blackout.astype(np.uint8)

    return augmented_image


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

# Load a random image from the images folder
def define_path(file_path, model_inf, class_names):
    if os.path.isdir(file_path):
        inference_dir(file_path, model_inf, class_names)
    else:
        inference_image(file_path, model_inf, class_names)


# If an image is sent for inference this function is used.
# It reads a single image.
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
    """Begins the inference process. First, the class names are loaded,
    then the specified path is checked to see if it is a directory or an
    image file."""
    inference_dataset = ImageDetectionDataset()
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
        description='Train Mask R-CNN to detect privates.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'inference' or 'optimizeHP' or 'removeBG'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/YOUR/dataset/",
                        help='Directory of the YOUR dataset')
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
    elif args.command == "inference":
        assert args.image,\
               "Provide --image=(image path) or (directory path)"
    elif args.command == "removeBG":
        assert args.image,\
               "Provide --image=(image path)"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ImageDetectionConfig()
    else:
        class InferenceConfig(ImageDetectionConfig):
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
        model = MaskRCNN(mode="training", config=config,
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
    elif args.command == "removeBG":
        detect_and_remove_background(model, image_path=args.image,
                                     video_path=args.video)
    elif args.command == "inference":
        inference(model_inf=model, path=args.image)
    elif args.command == "optimizeHP":
        optimize_hyperparameters(benchmark_model=model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))