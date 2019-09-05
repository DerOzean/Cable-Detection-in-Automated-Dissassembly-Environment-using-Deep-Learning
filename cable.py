"""
Usage: import the module, or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cable.py train --dataset=/path/to/cable/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 cable.py train --dataset=/path/to/cable/dataset --weights=/path/to/cable/weights

    # Train a new model starting from ImageNet weights
    python3 cable.py train --dataset=/path/to/cable/dataset --weights=imagenet

    # Apply color splash to video using the last weights you trained
    python3 cable.py splash --weights=/path/to/weights/file.h5 --video=<URL or path to file>
"""

import cv2

import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from matplotlib import pyplot as plt

# Path to Root directory
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # Find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

import imgaug as ia
import imgaug.augmenters as iaa

# Path to COCO_weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs

#Path to logs, to save model checkpoints
#command line --logs, if not provided
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#Set seed for randomness in Image Augumentation

ia.seed(4207645)

############################################################
#################  Configurations  #########################
############################################################











class CableConfig(Config):

    """
    Configuration for training on the cable-dataset.
    Derives from the base Config class and overrides some values.
    """
    # Create a suitable name 
    NAME = "cable"

    # Adjust the numer of images per GPU, for bigger GPU 2 or more
    IMAGES_PER_GPU = 1

    # Adjust how many classes you distinguish (including background)
    NUM_CLASSES = 1 + 1  # Background + cable

    # Training steps per epoch
    STEPS_PER_EPOCH = 20#333#519#598

    # Adjust the minmum confidence for detection in %
    DETECTION_MIN_CONFIDENCE = 0.5#0.875#0.83#0.7


############################################################
######################  Dataset  ###########################
############################################################

class CableDataset(utils.Dataset):

    def load_cable(self, dataset_dir, subset):
        """
        Load a subset of the cable-dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("cable", 1, "cable")

        # Choose datasets to load
        assert subset in ["train", "val", "predict"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotation for data
	#x and y coordinates are most importent  
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())

        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points that make up the outline.
            # Only polgons are recogniced
            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [r['region_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path) #Read image
            height, width = image.shape[:2]  #Save image height and width

            self.add_image(
                "cable",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):								
        """
	Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cable-dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cable":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Set every pixel inside the polygon to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """ Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cable":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask_hc(self, image_id):								
        """
        Generat instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cable-dataset image, delegate to phttps://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencvarent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cable":
            return super(self.__class__, self).load_mask(image_id)
	
	
	
        # Convert polygons to a bitmap mask.
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Set every pixel inside the polygon to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        # Pictures are labeled with name 'cable'.
        for i, p in enumerate(class_names):
            if p['name'] == 'cable':
                class_ids[i] = 14
            elif p['name'] == 'error':
                pass
            else:
                class_ids[i] = int(p['name'])
                # assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids
"""
#Old version from Mask_rcnn
def train(model, *dic):
        # Training dataset.
    dataset_train = CableDataset()
    dataset_train.load_cable(args.dataset, "train")
    dataset_train.prepare()
    # Validation dataset
    dataset_val = CableDataset()
    dataset_val.load_cable(args.dataset, "val")
    dataset_val.prepare()
    # No need to train long, since we're using small dataset.
    # Also just training the heads should be enough.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=333,	#Decrease this number for testing. 
                layers='heads')
"""
def train(model, *dic):
    """Train the model."""

    # Training dataset.
    dataset_train = CableDataset()
    dataset_train.load_cable(args.dataset, "train")
    dataset_train.prepare()
    print(dataset_train)
    # Validation dataset
    dataset_val = CableDataset()
    dataset_val.load_cable(args.dataset, "val")
    dataset_val.prepare()
    print(dataset_val)
    augmentation = iaa.Sometimes(.967, iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5), # vertical flip 50% of the time eren: (0)
        iaa.Sometimes(0.333,#later will be rescaled and translated with affine
            iaa.OneOf([
        
                iaa.Crop(percent=(0, 0.1)), # crops each side by a random value from the range 0 percent to 10 percent### Eren:percent=(0, 0.1)
                iaa.CropAndPad(percent=(0, 0.1),sample_independently=False)#crops and pad afterwards same parameters as early
                ])
        ),
        # Small gaussian blur with random sigma between 0 and 0.25.
        # But we only blur about 50 of all images.
        iaa.Sometimes(0.0533,
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 0.25)),####Eren: (0,0.25) and only GaussianBlur
        #        #iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
        #        #iaa.MedianBlur(k=(2, 3)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025*255)),
        #        iaa.AdditiveGaussianNoise(per_channel=True, scale=(0.0,0.028*255)),
        #        iaa.AdditiveLaplaceNoise(scale=(0.0, 0.025*255)),
                #iaa.ElasticTransformation(alpha=(0, 1), sigma=(0.8,1))#Distort images locally by moving individual pixels around following a distortions field with strength 
        #        
            ])
        ),
        # Strengthen or weaken the contrast in each image.
        #iaa.Sometimes(0.145,
         #   iaa.OneOf([
          #      iaa.ContrastNormalization((0.825, 1.5), per_channel=0.3), #Eren: (0.5, 1.5) without sometimes
           #     iaa.GammaContrast(gamma=(0.825, 1.5), per_channel= 0.3),#Eren: Just had Contrast Normalization
            #    iaa.SigmoidContrast(gain=(2,8), cutoff=(0.333,0.667)),
             #   iaa.Sharpen(alpha=(0.250, 0.75), lightness=(0.75, 1.25))#Sharpen an image, then overlay the results with the original using an alpha between 0.0 and 1.0:
            #])
        #),
        #Have to find a fix, because the mask can't be color changed.
        #iaa.Sometimes(0.25,# change the colorspace from RGB to HSV, then add 50-100 to the first channel, then convert back to RGB. This increases the hue value of each image
        #       iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        #      iaa.WithChannels(0, iaa.Add((50, 100))),
        #     iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        #),
        # Add gaussian noise.
        # For 50 of all images, we sample the noise once per pixel.
        # For the other 50 of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)), #moved it upwards to the blur
        # Make some images brighter and some darker.
        # In 20 of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Sometimes(0.1667,
            iaa.OneOf([
                iaa.Multiply((0.8, 1.2)), #Eren:(0.8,1.2)
                iaa.Multiply((0.8, 1.2), per_channel=0.833),###Sometimes focus a colour channel 
                iaa.Add((-15, 15), per_channel=0.75), # add a random value to a colour channel
                #iaa.AddToHueAndSaturation((-15, 15))
            ])
        ),
        
        #iaa.Sometimes(0.125,
        #    iaa.OneOf([
        #        iaa.Invert(0.25, per_channel=0.5),
        #        iaa.Grayscale(alpha=(0.0, 0.3)),
        #        iaa.ContrastNormalization((0.85, 1.15), per_channel=0.75),
        #        iaa.Sharpen(alpha=(0.0, 0.750), lightness=(0.75, 2.0))
        #    ])
        #),
        
        # Either drop randomly 1 to 10 of all pixels (i.e. set
        # them to black) or drop them on an image with 2-5% percent
        # of the original size, leading to large dropped
        # rectangles.
        #Module for landscapes but here uesed to make the imperession of a dirty cable or a cable were parts are hidden.
        #iaa.Sometimes(0.0027,
         #   iaa.OneOf([
          #      iaa.Clouds(),
                #iaa.Clouds(),
                #iaa.FastSnowyLandscape(),
                #iaa.FrequencyNoiseAlpha(
                 #       exponent=(-4, 0),
                  #      first=iaa.Multiply((0.5, 1.5), per_channel=True),
                   #     second=iaa.ContrastNormalization((0.5, 2.0))
                   # ),
                #iaa.FrequencyNoiseAlpha(
                #    first=iaa.EdgeDetect(1.0),
                 #   per_channel=True),
                #iaa.SimplexNoiseAlpha(iaa.OneOf([
                 #   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                  #  iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                #]))
            #])
        #),
        #wasn't supposed by eren
        #iaa.Sometimes(0.012,
        #    iaa.OneOf([
        #                iaa.Dropout((0.01, 0.1), per_channel=0),
                        #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.03, 0.12),per_channel=True),
        #            ])),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Sometimes(0.8,
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #wasn't supposed by eren
            rotate=(-180, 180),#We already flip the images at the begining therefor a rotation of 90% should be enough. ###Eren: rotate=(-180, 180)
            #shear=(-8, 8),
            mode=["constant"], # Eren: had only black new pixels, now pixels are const in diff colors.
            cval=(0, 128)#Collor spectrum for the additinal pixels
            )
        )], random_order=True))# apply augmenters in random order
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training heads")

    model.train(dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs= 86,#333,#100,
    layers='heads', augmentation=augmentation)
    
    print("Training all layers")
    model.train(dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs= 1000,#333,#100,
    layers='all', augmentation=augmentation)

def color_splash(image, mask):
    """
    Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Create a grayscale copy of the image.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255			
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Taking colors from original image, in area of mask.
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None, out_dir=''):
    assert image_path or video_path

    class_names = ['BG', 'cable']

    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], making_image=True)
        file_name = 'splash.png'
        #Save output
        #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #save_file_name = os.path.join(out_dir, file_name)
        #skimage.io.imsave(save_file_name, splash)
    elif video_path:
        VIDEO_SAVE_DIR= '../../../Videos/save/'
        import cv2
        #import glob
        #batch_size=1
        count=0
        capture = cv2.VideoCapture(video_path)  
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = capture.read()
            #cv2.imshow('Hallo',frame)
            #frame = frame.astype(np.uint8)
            #
            # Bail out when the video file ends
            if not ret:
                break        
            # Save each frame of the video to a list
            #frames = []
            count += 1
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)#cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
            #cv2.imshow('Hallo',frame)
            #print(count)
            #frames.append(frame)
            #if len(frames) == batch_size:
            r = model.detect([frame], verbose=1)[0]
            #for i, item in enumerate(zip(frames, results)):
            #frame = item[0]
            #r = item[1]
            colors = visualize.random_colors(len(class_names))
            frame = visualize.display_instances_video(
                                    count, frame, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], colors=colors)
            #(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            # Clear the frames array to start the next batch
            #frames = []
        # Get all image file paths to a list.
        #frames = []
        #images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
        # Sort the images by name index.
        #images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))
        #video = cv2.VideoCapture(os.path.join(VIDEO_SAVE_DIR, 'trailer1.mp4'));
        # Find OpenCV version
        """
        images = []
        for img in glob.glob(VIDEO_SAVE_DIR+"images/*.jpg"):
            n= cv2.imread(img)
            images.append(n)
        video = cv2.VideoWriter('video.avi',-1,1,(1200,900))
        for q in range(1,count):
            video.write(images[q])
        cv2.destroyAllWindows()
        video.release()
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = video.get(cv2.CAP_PROP_FPS)
            #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        video.release()
        """

        images = [img for img in sorted(os.listdir(VIDEO_SAVE_DIR)) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(VIDEO_SAVE_DIR, images[0]))
        height, width, layers = frame.shape
        video_name = 'Detection.avi'
        video = cv2.VideoWriter(video_name, 0, fps, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(VIDEO_SAVE_DIR, image)))

        cv2.destroyAllWindows()
        video.release()

    """elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = 1600
        # height = 1600
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer "splash_{:%Y%m%dT%H%M%S}.wmv"
        fname = "splash_{:%Y%m%dT}.wmv".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(fname,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        #For video, we wish classes keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(class_names))
        while success:
            print("frame: ", count)
            # Read next image
            plt.clf()
            plt.close()
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                #image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                #splash = color_splash(image, r['masks'])

                splash = visualize.display_instances_video(image, r['rois'], r['masks'], r['class_ids'],
                                                     class_names, r['scores'])#, colors=None)#, making_video=True)

                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
"""
############################################################
####################  RLE Encoding  ########################
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def detect(model, dataset_dir, subset):
    """ Run detection on images in the given directory. """
    print("Running on {}".format(dataset_dir))

    os.makedirs('RESULTS')
    submit_dir = os.path.join(os.getcwd(), "RESULTS/")
    # Read dataset
    dataset = CableDataset()
    dataset.load_cable(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)





        # Save image with masks
        canvas = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'], detect=True)
            # show_bbox=False, show_mask=False,
            # title="Predictions",
            # detect=True)
        canvas.print_figure("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"][:-4]))
    # Save to csv file


    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect hdd cable.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/home/eren/workspace/mask_rcnn/datasets/cable",
                        help='Directory of the cable dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/home/eren/workspace/logs/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    elif args.command == "splash":
        assert args.video#args.image or args.video,\
               #"Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)


    # Configurations
    if args.command == "train":
        config = CableConfig()
    else:
        class InferenceConfig(CableConfig):
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
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
