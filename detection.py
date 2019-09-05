import matplotlib
matplotlib.use('Agg')
import cv2
import os
import sys
import glob
import h5py


# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import cable

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
PRETRAINED_MODEL_PATH = '../../logs/mask_rcnn_cable_0100.h5'
PREDICTION_DATA_PATH = '../../datasets/cable/predict/'
OUTPUT_PATH = "../../output_images"

class InferenceConfig(cable.CableConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "inference"
    NUM_CLASSES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if __name__ == '__main__':
    class_names = ['BG', 'cable']
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='../../logs/')
    model_path = PRETRAINED_MODEL_PATH
    model.load_weights(model_path, by_name=True)
    colors = visualize.random_colors(len(class_names))

    images_path = os.path.join(PREDICTION_DATA_PATH, '*.jpg')
    files = glob.glob(images_path)
    images = []
    counter = 0
    for f1 in files:
        img = cv2.cvtColor(cv2.imread(f1), cv2.COLOR_BGR2RGB)
        predictions = model.detect([img], verbose=1)  # We are replicating the same image to fill up the batch_size
        p = predictions[0]
        output = visualize.display_instances(img, p['rois'], p['masks'], p['class_ids'],
                                    class_names, p['scores'], colors=colors, title=counter)#, real_time=True)
        cv2.imwrite(os.path.join(OUTPUT_PATH , 'second'+str(counter) + '.png'), output)
        counter += 1
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
