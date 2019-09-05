import json
import glob, cv2
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from cv2 import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser(
        description='convert dataset')
parser.add_argument('--json_path', required=True,
                metavar="/path/to/json/dataset/",
                help='path of json file')
parser.add_argument('--out_dir', required=True,
                metavar="/path/to/annotation/files/",
                help='Directory of the all annotation images')
parser.add_argument('--in_dir', required=True,
                metavar="/path/to/images folder",
                help='Directory of the all images')
args = parser.parse_args()

colors=[(31,23, 176),(205,16,118),(153,50,204),(230,230,250),(198,226,255),(95,158,160),(60,179,113),(48,128,20),(107,142,35),(255,153,18),(55,226,255),(95,55,160),(60,55,113),(48,138,20),(0,142,35),(55,153,18)]

encode = {'cable':1}

annotations = json.load(open(args.json_path))
#del annotations['385.png1262170']
annotations = list(annotations.values())  # don't need the dict keys
print(len(annotations))
for mask in annotations:
  image_name = mask['filename']
  print(image_name)
  #idx = int(image_name.split('.')[0])
  #if idx>=389:
  #   continue
  #print(idx)
  #try:
  image = cv2.imread(args.in_dir+image_name)
  #print(image)
  im_h, im_w  = image.shape[:2]
  #print(im_h)
  #print(im_w)
  #except:
    #continue
  ratio=1
  ann_img = np.zeros((im_h,im_w,3)).astype('uint8')
  _polygons = []
  areas = []
  _names = []
  for idn, region in enumerate(mask['regions']):
    xs = region['shape_attributes']['all_points_x']
    ys = region['shape_attributes']['all_points_y']
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    #print(region)
    name = region['shape_attributes']['name']
    _names.append(name)

    pls = []
    area  = (xmax-xmin)*(ymax-ymin)
    for i in range(len(xs)):
        pls.append((int(xs[i]/ratio),int(ys[i]/ratio)))
    polygon = Polygon(pls)
    _polygons.append(polygon)
    areas.append(area)

  aaa = np.argsort(areas)
  #print(aaa)
  #print(len(_polygons))
  polygons = []
  labels = []
  for i in aaa:
    polygons.append(_polygons[i])
    #labels.append(encode[_names[i]])
  for i in range(im_w):
      #print("hallo1")  
      for j in range(im_h):
         #print(image[i,j])
         point = Point(i,j)
         for aa, polygon in enumerate(polygons):
           if polygon.contains(point):
             ann_img[j,i]= [255,255,255]#image[j,i]#1#labels[aa]
             break
  #print("hallo")           
  #cv2.imshow(ann_img)
  cv2.imwrite(args.out_dir+'training'+str(mask)+'.jpg' ,ann_img)
  if not cv2.imwrite(args.out_dir+image_name ,ann_img):
     raise Exception("Could not write image")

