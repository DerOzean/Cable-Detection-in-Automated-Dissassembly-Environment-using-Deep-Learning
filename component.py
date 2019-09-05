import os
from os.path import expanduser
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob, cv2
import shutil
import operator
import tensorflow as tf

# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Get home
home = expanduser('~')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# trained model path
TRAINED_MODEL_PATH = '../../logs/mask_rcnn_cable_0100.h5'
PREDICTION_DATA_PATH = '../../datasets/cable/predict_old/'
OUTPUT_PATH = "../../output_images"

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# all the necessary classes for the device of interest: HDD
class_names = ['BG','cable']

# epsilon value for the evaluation calculations
EPS = 1e-12

def numpy2Mat(arrayImg):
    """
    Convert a numpy array into an RGB Image of OpenCV
    """
    return cv2.cvtColor(arrayImg*255, cv2.COLOR_GRAY2BGR)

# a custom class for the device of interest
class ComponentDataset(utils.Dataset):

    def load_data(self, dataset_dir, annotation):

        self.encode = {'cable':1}#'fpc':2, 'bearing':3, 'rw_head':4, 'spindle_hub':5, 'platters_clamp':6, 'platter':7, 'magnet':1, 'bay':8, 'lid':9, 'pcb':10, ' spindle_hub':5, 'head_contacts':11, 'top_dumper':12, 'spindle__hub':5}
        self.add_class('images', 1, "cable")
        #self.add_class('images', 2, "fpc")
        #self.add_class('images', 3, "bearing")
        #self.add_class('images', 4, "rw_head")
        #self.add_class('images', 5, "spindle_hub")
        #self.add_class('images', 6, "platters_clamp")
        #self.add_class('images', 7, "platter")
        #self.add_class('images', 8, "bay")
        #self.add_class('images', 9, "lid")
        #self.add_class('images', 10, "pcb")
        #self.add_class('images', 11, "head_contacts")
        #self.add_class('images', 12, "top_dumper")
        annotations = json.load(open(annotation))
        
        try:
           del annotations['385.png1262170']
        except:
           pass
        valist= []
        removekeys = []
        for key in annotations.keys():
           aaa = key.split('.')[0]+'.png'
           if aaa in valist:
               removekeys.append(key)
        for key in removekeys:
           del annotations[key]

        annotations = list(annotations.values())
        # Add images
        for mask in annotations:
            if type(mask['regions']) is dict:
                polygons = [r for r in mask['regions'].values()]
            else:
                polygons = [r for r in mask['regions']]
            image_name = mask['filename']
            image_path = os.path.join(dataset_dir, image_name)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "images",
                image_id=mask['filename'],  # use file name as a unique image id
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
        # If not a component dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "images":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        classes = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            classes.append(self.encode[p['region_attributes']['name']])
            rr, cc = skimage.draw.polygon(p['shape_attributes']['all_points_y'], p['shape_attributes']['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(classes)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "images":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ComponentDataset()
    dataset_train.load_data(args.dataset, args.annotation)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ComponentDataset()
    dataset_val.load_data(args.dataset, args.annotation)
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    print("Training network")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=90,
                layers='all')

# function to get the IoU (intersection over Union) metric
def get_iou(gt, pr, n_classes):
        class_wise = np.zeros(n_classes)
        for cl in range(n_classes):
                intersection = np.sum(( gt == cl )*( pr == cl ))
                union = np.sum(np.maximum( ( gt == cl ) , ( pr == cl ) ))
                iou = float(intersection)/( union + EPS )
                class_wise[ cl ] = iou
        return class_wise

def get_segmentation_arr(path , nClasses, no_reshape=False):
        img = cv2.imread(path, 1)
        height , width = img.shape[:2]
        seg_labels = np.zeros((height, width , nClasses))
        img = img[:, : , 0]

        for c in range(nClasses):
                seg_labels[: , : , c ] = (img == c ).astype(int)

        if no_reshape:
                return seg_labels

        seg_labels = np.reshape(seg_labels, (height, width , nClasses ))
        return seg_labels

def evaluate(seg_dir, gt_dir):
    ious = []
    imagenames = os.listdir(seg_dir)
    for name in imagenames:
        gt_path = gt_dir + name
        seg_path = seg_dir + name
        gt = get_segmentation_arr(gt_path, len(class_names))
        seg = get_segmentation_arr(seg_path, len(class_names))
        gt = gt.argmax(-1)
        seg = seg.argmax(-1)
        iou = get_iou( gt , seg , len(class_names) )
        ious.append(iou)

    ious = np.array( ious )
    print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
    print("Total  IoU "  ,  np.mean(ious ))

# To find the boundaries and centers of the masks
def getBoundaryPositions(mask):

    # conver to opencv type
    mask_cv = mask.astype(np.uint8)

    # Find contours
    (mask_cv, contours, hierarchy) = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # get the biggest contour, noise gets eliminated this way
    cnt = max(contours, key = cv2.contourArea)  
    
    # Calculate image moments of the detected contour
    M = cv2.moments(cnt)

    # collect pose points of the center
    pose = []

    # to prevent zero division error, do a check
    if M["m00"] != 0: 
        pose.append(round(M['m10'] / M['m00'])) #x
        pose.append(round(M['m01'] / M['m00'])) #y
        #z, put zero for now
        pose.append(0)

        outline_poses = np.array([np.append(x[0], 0)for x in cnt])
        
        # TODO: FIND A WAY TO GET THE ORIENTATION
        pose.append(0) #roll
        pose.append(0) #pitch
        pose.append(0) #yaw
    else:
        outline_poses = []

    return (mask_cv, pose, outline_poses)

# function used to classify the bearing type, launches the DNN
def classify_bearing_type(image_path):
    # read the image data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # keep the labels in an array
    label_lines = [line.rstrip() for line 
                    in tf.gfile.GFile(home + '/ownCloud/imagine_weights/bearing_classifier/retrained_labels.txt')]

    with tf.gfile.FastGFile(home + '/ownCloud/imagine_weights/bearing_classifier/retrained_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; 
        graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
        _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # run the prediction
        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # get the prediction values in array
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        # get the output
        results = {}
        for node_id in top_k:
            if (node_id == 2): # it's a binary classification so only 2 classes are needed: 0 and 1
                return results
            class_name = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (class_name, score))
            results.update({class_name:score}) 

    return results

# To find out the type of a specific component. In this case, the bearing
def getType(mask, image, BB, working_dir):
    
    mask = mask.astype(np.uint8)
    part_type = ''
    part_type_specifics_confidence = .99
      
    # get the ROI (where the bearing BB is)
    # +/- 10 pixels in each direction are enough for a ROI cut window for the bearing
    roi = image[BB[0]-10:BB[2]+10, BB[1]-10:BB[3]+10]

    # save the image under /tmp
    folder_path = working_dir
    bearing_image_path = folder_path + "/bearing.png"
    cv2.imwrite(bearing_image_path, roi)

    # use the inception network to classify the bearing
    # which side are we looking at
    results = classify_bearing_type(bearing_image_path)

    # get the best scoring side and its score
    part_type = max(results.items(), key=operator.itemgetter(1))[0]
    part_type_specifics_confidence = max(results.items(), key=operator.itemgetter(1))[1]

    return part_type, part_type_specifics_confidence

def detect(working_dir, model, image_path=None):
    # define the name of the directory to be created 
    folder_path = working_dir

    # Read image
    image = skimage.io.imread(image_path)

    # Detect objects
    predictions = model.detect([image], verbose=1)[0]
    scores = predictions['scores']
    masks = predictions['masks']
    class_ids = predictions['class_ids']
    rois = predictions['rois']

    '''
    if len(scores)>0:
        index = np.argmax(scores)
        # for convenience, use simpler notation
        rois = predictions['rois'][index]
        score = scores[index]
    '''
    # to iterate through the instance, we need an incrementable enumerator
    enumerator = 0
    
    # we'll keep the explored data in a dict
    state_estimation_data = dict()

    # uncomment to display the results
    #visualize.display_instances(image, predictions['rois'], predictions['masks'], predictions['class_ids'], class_names, predictions['scores'])
    
    # run through the instances
    for class_id in class_ids:
        id = class_names[class_id] + str(enumerator)
        part_type = ''
        part_type_specifics_confidence = 0.99 # default value for the confidence
        # if the part is bearing, then classify it
        if (class_names[class_id] == 'bearing'):
            part_type, part_type_specifics_confidence = getType(masks[:,:,enumerator], image, rois[enumerator], folder_path)
        # get the outline coordinates
        part_mask, poses, outline_poses = getBoundaryPositions(masks[:, :, enumerator])
         # skip the mask if the shape is weird. This is a rare situation, but may happen.
        if (len(poses) == 0 or len(outline_poses) == 0):
            continue
        part_type_confidence = scores[enumerator]

        # save the images to the folder, as well as the .json
        assos_img_path = folder_path + "/assos_img.png"
        part_mask_path = folder_path + "/part_mask" + "_" + id + ".png"
        cv2.imwrite(assos_img_path, image)
        part_mask = numpy2Mat(part_mask)  # convert to opencv type of mat
        cv2.imwrite(part_mask_path, part_mask)

        # Define data to be written
        component_data = {
            #'part_id': (class_names[class_id] + str(enumerator)),
            'part_bounding_box': rois[enumerator].tolist(),
            'outline_poses': outline_poses.tolist(),
            'part_type': part_type,
            'part_type_confidence': float(part_type_confidence),
            'part_type_specifics_confidence': float(part_type_specifics_confidence),
            'pose': poses
        }

        # write everything to another dict
        state_estimation_data[class_names[class_id] + str(enumerator)] = component_data

        # increment per component to form the id
        enumerator = enumerator + 1

    # save the result image with all the detections
    visualize.save_image(image, "result_image", rois, masks, class_ids, scores, class_names, state_estimation_data, filter_classs_names=None, scores_thresh=0.8, save_dir=folder_path, mode=0)

    # write JSON file
    with io.open(folder_path + '/state_estimation_data.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(state_estimation_data,
                            indent=4, sort_keys=True,
                            separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

    # to test, read the file
    with open(folder_path + '/state_estimation_data.json') as data_file:
        data_loaded = json.load(data_file)

    print("State Estimation -> .json file loaded: ", state_estimation_data == data_loaded)


def save_detect_list(model, image_dir=None):
    image_path = os.listdir(image_dir)
    print(image_path[0])
    try:
        shutil.rmtree('draw')
    except:
        pass
    try:
        os.mkdir('draw')
    except:
        pass
    for image_name in image_path:
        image = skimage.io.imread(image_dir+image_name)
        # Detect objects
        predictions = model.detect([image], verbose=1)[0] 
        scores = predictions['scores']
        
        if len(scores)>0:
            index = np.argmax(scores)
            box = predictions['rois'][index]
            score = scores[index]
        visualize.save_instances(image, predictions['rois'], predictions['masks'], predictions['class_ids'], 
                            class_names,image_name, predictions['scores'])

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect components of a HDD.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/component/dataset/",
                        help='Directory of the all component images')
    parser.add_argument('--annotation', required=False,
                        default="ucdata4.json",
                        metavar="/path/to/annotation/file/",
                        help='Directory of the all component images')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--test_dir', required=False,
                        default="test_images/",
                        metavar="/path/to/test images",
                        help="Path to test directory")
    parser.add_argument('-i','--image', required=False,
                        default="test_images/image1.png",
                        metavar="/path/to/test images",
                        help="Path to the image")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('-w', '--working_dir', required=True,
                        help='Working dir to store data')
    parser.add_argument('--seg_dir', required=False, 
                        help='Folder path to the segmentation images')
    parser.add_argument('--gt_dir', required=False, 
                        help='Folder generated for ground truth images')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    if args.command == "detect":
        assert args.image, "Argument --image is required for detection"
    if args.command == "evaluate":
        assert args.gt_dir, "Argument --gt_dir is required for evaluation"
        assert args.seg_dir, "Argument --seg_dir is required for evaluation"
        assert args.test_dir, "Argument --test_dir is required for evaluation"

    # for the time being, get it directly by the path, not by the user
    args.weights = TRAINED_MODEL_PATH

    print("Weights: ", args.weights)
    print("Image: ", args.image)
    print("Dataset: ", args.dataset)
    print("Working Dir: ", args.working_dir)
    print("Seg Dir: ", args.seg_dir)
    print("Gt Dir: ", args.gt_dir)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ComponentConfig()
    else:
        class InferenceConfig(ComponentConfig):
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
    elif args.command == "detect":
        detect(args.working_dir, model, args.image)
    elif args.command == "evaluate":
        save_detect_list(model, args.test_dir)
        evaluate(args.seg_dir, args.gt_dir)
        #os.system('python3 evaluation/evaluation.py')
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'detect' or 'evaluate'".format(args.command))
