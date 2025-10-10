<div align="center" markdown>

<img src="https://user-images.githubusercontent.com/106374579/183667927-7b2161e7-97e3-4219-90ec-6b558992faa2.png"/>

# Train YOLOv5

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a>
  <a href="#How-To-Use-Your-Trained-Model-Outside-Supervisely">How To Use Your Trained Model Outside Supervisely</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a> •
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../supervisely-ecosystem/yolov5/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/yolov5/supervisely/train)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/yolov5/supervisely/train)](https://supervisely.com)

</div>

# Overview

Train YOLOv5 on your custom data. All annotations will be converted to the bounding boxes automatically. Configure Train / Validation splits, model and training hyperparameters. Run on any agent (with GPU) in your team. Monitor progress, metrics, logs and other visualizations withing a single dashboard.  


Major releases:
- **May 17, 2021**: [v5.0 release](https://github.com/supervisely-ecosystem/yolov5/tree/v5.0.0): merge updates from original YOLOv5 repo (including new model architectures), split data to train/val based on datasets or tags, update UI for settings, other fixes
- **March 3, 2021**: [v4.0 release](https://github.com/supervisely-ecosystem/yolov5/tree/v4.0.9): YOLOv5 is integrated to Supervisely (train / serve / inference)

# How To Use
Watch short video for more details:

<a data-key="sly-embeded-video-link" href="https://youtu.be/e47rWdgK-_M" data-video-code="e47rWdgK-_M">
    <img src="https://i.imgur.com/sJdEEkN.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

1. Add app to your team from Ecosystem
2. Be sure that you connected computer with GPU to your team by running Supervisely Agent on it 
3. Run app from context menu of images project
4. Open Training Dashboard (app UI) and follow instructions provided in the video below
5. All training artifacts (metrics, visualizations, weights, ...) are uploaded to Team Files. Link to the directory is provided in output card in UI. 
   
   Save path is the following: ```"/yolov5_train/<input project name>/<task id>```

   For example: ```/yolov5_train/lemons-train/2712```
   
6. Also in this directory you can find file `open_app.lnk`. It is a link to the finished UI session. It can be opened at any time to 
   get more details about training: options, hyperparameters, metrics and so on.

   <img src="https://i.imgur.com/fIgiKMJ.png"/>
   
   - go to `Team Files`
   - open directory with training artifacts
   - right click on file `open_app.lnk`
   - open

# How To Use Your Trained Model Outside Supervisely

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. See this [Jupyter Notebook](https://github.com/supervisely-ecosystem/yolov5/blob/master/inference_outside_supervisely.ipynb) for details.

# Screenshot

<img src="https://i.imgur.com/eiROUgb.png"/>

# Acknowledgment

This app is based on the great work `yolov5` ([github](https://github.com/ultralytics/yolov5)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/yolov5?style=social)
