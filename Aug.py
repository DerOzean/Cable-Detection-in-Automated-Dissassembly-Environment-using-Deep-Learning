import cable
#from mrcnn import visualize
#from mrcnn import model as modellib, utils
#from mrcnn.config import Config
import cv2
import os
import sys
import glob
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../Aug/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
PRETRAINED_MODEL_PATH = '../../logs/mask_rcnn_cable_0100.h5'
PREDICTION_DATA_PATH = '../../datasets/cable/predict_ac/'
OUTPUT_PATH = "../../output_images"

#import glob
from scipy import ndimage

fps = glob.glob("/home/michael/Bachelor/important/Cable-detection/Mask_RCNN-2.1/datasets/cable/predict_ac/Predict_*.jpg")
images = [ndimage.imread(fp, mode="RGB") for fp in fps]

ia.seed(3455)

augmentation = iaa.Sometimes(.967, iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5), # vertical flip 50% of the time eren: (0)
        #iaa.Sometimes(0.333,#later will be rescaled and translated with affine
         #   iaa.OneOf([
        #
          #      iaa.Crop(percent=(0, 0.1)), # crops each side by a random value from the range 0 percent to 10 percent### Eren:percent=(0, 0.1)
            #    iaa.CropAndPad(percent=(0, 0.1),sample_independently=False)#crops and pad afterwards same parameters as early
           #     ])
        #),
        # Small gaussian blur with random sigma between 0 and 0.25.
        # But we only blur about 50 of all images.
        #iaa.Sometimes(0.0533,
         #   iaa.OneOf([
                #iaa.GaussianBlur(sigma=(0, 0.25)),####Eren: (0,0.25) and only GaussianBlur
        #        #iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
        #        #iaa.MedianBlur(k=(2, 3)),
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025*255)),
        #        iaa.AdditiveGaussianNoise(per_channel=True, scale=(0.0,0.028*255)),
        #        iaa.AdditiveLaplaceNoise(scale=(0.0, 0.025*255)),
                #iaa.ElasticTransformation(alpha=(0, 1), sigma=(0.8,1))#Distort images locally by moving individual pixels around following a distortions field with strength 
        #        
           # ])
        #),
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
        #iaa.Sometimes(0.5667,
         #   iaa.OneOf([
          #      iaa.Multiply((0.8, 1.2)), #Eren:(0.8,1.2)
           #     iaa.Multiply((0.8, 1.2), per_channel=0.833),###Sometimes focus a colour channel 
            #    iaa.Add((-15, 15), per_channel=0.75), # add a random value to a colour channel
                #iaa.AddToHueAndSaturation((-15, 15))
            #])
        #),
        
        #iaa.Sometimes(0.125,
         #   iaa.OneOf([
        #        iaa.Invert(0.25, per_channel=0.5),
          #      iaa.Grayscale(alpha=(0.0, 0.3)),
           #     iaa.ContrastNormalization((0.85, 1.15), per_channel=0.75),
            #    iaa.Sharpen(alpha=(0.0, 0.750), lightness=(0.75, 2.0))
           # ])
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
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #wasn't supposed by eren
            #rotate=(-180, 180),#We already flip the images at the begining therefor a rotation of 90% should be enough. ###Eren: rotate=(-180, 180)
            #shear=(-8, 8),
            #mode=["constant"], # Eren: had only black new pixels, now pixels are const in diff colors.
            #cval=(0, 128)#Collor spectrum for the additinal pixels
            )
        )], random_order=True))# apply augmenters in random order
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
print(len(fps))
for i in range(len(fps)):
    images_aug = augmentation.augment_image(images[i])
    new_img = cv2.cvtColor(images_aug , cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(ROOT_DIR,str('Augumentation')+str(i)+'.jpg'), new_img)
    cv2.waitKey(20)
    #print(images_aug)
#cv2.imshow("Original", images_aug)#[0, ..., ::-1])
cv2.waitKey(0)
