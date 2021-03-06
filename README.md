# Cable-Detection-in-Automated-Dissassembly-Environment-using-Deep-Learning
Cables are essential components of any device in the electronic waste sector. For several years disassembly processes in this sector got automated step by step. However, most of these automation’s were device-specific and not usable on a wide range 1 . An advantage came about as neural networks improved in image recognition. The recent improvements marked the starting point for many research groups to focus on disassembly lines that can handle multiple devices. This bachelor thesis aims to address a fundamental problem in such disassembly processes: Cable-Detection . Consider the task of disassembling a DVD-Player which still includes usable components. Cables connect most of these components for information and energy exchange. Therefore one main goal before removing individual components is to cut the cables between the components to ensure their safe removal. Inevitably this task needs a precise knowledge about the position of every single cable. For a disassembly robot, a strategy that is worth pursuing is to locate the cables process- ing an RGB image. An RGB image is a lightweight solution with high standards regarding the improvement in camera technology over the last year through mobile devices. This Bachelor thesis presents a model for cable detection.  A state-of-the-art instance segmentation model Mask R-CNN 2 , published in 2018, is trained. The presented work is public on GitHub to make this global concerning topic available for the public. 

# Keywords: 
Deep Convolutional Neuronal Networks, Mask R-CNN, Cable-Detection, Disassembly, E-Waste

# Walk Trough:
Please have a look at Thesis.pdf for a extensive discription. 

# Backbone
During this thesis matterport/MaskR_CNN was used as groundwork. Nearly all code is uploaded without significant changes.
For futher informations see: https://github.com/matterport/Mask_RCNN

# Dataset
The used dataset is public on kaggle.com: https://www.kaggle.com/zeuscasio/cable-dataset-annotation

# Folder structure

The presented code was combined in one folder called "samples". "samples" was included in Mask R-CNN. Also, the folder "datasets" and "logs" were included in Mask R-CNN. "datasets " included one folder named "cable" in which the three folders from kaggle are placed (train, val, predict).
Also, a folder "logs" was included in Mask R-Cnn. In this folder, the trained network is placed for detections. 

# Trained network

The trained network is too big (~250 MB) for uploading it to Git Hub. If it's needed by someone, please let me know. 
