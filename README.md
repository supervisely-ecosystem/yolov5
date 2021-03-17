This guide explains how to use Supervisely with YOLOv5.

# Table of Contents

1. [About Supervisely](#About-Supervisely)
2. [YOLOv5 Apps Collection](#YOLOv5-Apps-Collection) 
3. [Prerequisites](#Prerequisites)
4. [Train Custom Data](#Installation)
5. [Deploy model as REST API](#Documentation)
6. [Integrate model to labeling UI](#Example-images)
7. [Apply model to images project](#Eecent-changes)
8. [For developers](#For-developers)
9. [Contact](#Contact)

# About Supervisely

You can think of Supervisely as an Operating System available via Web Browser to help you solve Computer Vision tasks. The idea is to unify all the relevant tools that may be needed to make the development process as smooth and fast as possible. 

More concretely, Supervisely includes the following functionality:
 - Data labeling for images, videos, 3D point cloud and volumetric medical images (dicom)
 - Data visualization and quality control
 - State-Of-The-Art Deep Learning models for segmentation, detection, classification and other tasks
 - Interactive tools for model performance analysis
 - Specialized Deep Learning models to speed up data labeling (aka AI-assisted labeling)
 - Synthetic data generation tools
 - Instruments to make it easier to collaborate for data scientists, data labelers, domain experts and software engineers

One challenge is to make it possible for everyone to train and apply SOTA Deep Learning models directly from the Web Browser. To address it, we introduce an open sourced Supervisely Agent. All you need to do is to execute a single command on your machine with the GPU that installs the Agent. After that, you keep working in the browser and all the GPU related computations will be performed on the connected machine(s).

# YOLO v5 apps collection

YOLOv5 is a part of [Supervisely Ecosystem](https://ecosystem.supervise.ly/) 🎉. Now Supervisely users can quickly train and use YOLOv5 with their custom data with a few clicks.

# Prerequisites

1. Add YOLOv5 [train](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fyolov5%252Fsupervisely%252Ftrain) / [serve](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fyolov5%252Fsupervisely%252Fserve) apps from Ecosystem to your team.
2. Be sure that you connected computer with GPU to Supervisely. Watch how-to video:

# YOLOv5 integration into Supervisely

- [train app](./supervisely/train/README.md)

<img src="https://i.imgur.com/YwSq29o.png"/>

- [serve app](./supervisely/serve/README.md)

<img src="https://i.imgur.com/1qXIdqs.png"/>

Original readme is [here](./README-original.md).


# For Developers
- associated release version
- sources
- contact us
- for enterprises