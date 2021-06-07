<div align="center" markdown>
<img src=""/>

# Export YOLOv5 weights

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#For-Developers">For Developers</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5/supervisely/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App deploys YOLO v5 model (pretrained on COCO or custom one) as REST API service. Serve app is the simplest way how any model can be integrated into Supervisely. Once model is deployed, user gets the following benefits:

1. Use out of the box apps for inference
   - used directly in [labeling interface](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) (images, videos)
   - apply to [images project or dataset](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset)
   - apply to videos (coming soon)
2. Apps from Supervisely Ecosystem can use NN predictions: for visualization, for analysis, performance evaluation, etc ...
3. Communicate with NN in custom python script (see section <a href="#For-developers">for developers</a>)
4. App illustrates how to use NN weights. For example: you can train model in Supervisely, download its weights and use them the way you want.


# How To Run

**For pretrained model**: just choose weights from dropdown menu and press `Run`. 

<img src="https://i.imgur.com/SEuE2jD.png" width="400"/>


**For custom weights**: 

1. Training app saves artifacts to `Team Files`. Just copy path to weights `.pt` file. 
   Training app saves results to the directory: `/yolov5_train/<training project name>/<session id>/weights`. 
   For example: `/yolov5_train/lemons_annotated/2577/weights/best.pt`

<img src="https://i.imgur.com/VkSS58q.gif" width="800"/>

2. Paste path to modal window

<img src="https://i.imgur.com/YbnwzI7.png" width="400"/>

Then

3. Choose device (optional): for GPU just provide device id (`0` or `1` or ...), or type `cpu`. Also in advanced section you can 
change what agent should be used for deploy.

4. Press `Run` button.

5. Wait until you see following message in logs: `Model has been successfully deployed`

<img src="https://i.imgur.com/wKs7zw0.png" width="800"/>


# For Developers

This python example illustrates available methods of the deployed model. Now you can integrate network predictions to your python script. This is the way how other Supervisely Apps can communicate with NNs. And also you can use serving app as an example - how to use download NN weights outside Supervisely.

To implement serving app developer has just to define four methods:
- function [`get_session_info`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/sly_serve.py#L50) - information about deployed model (returns python dictionary with any useful information)
- function [`construct_model_meta`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/nn_utils.py#L16) - returns model output classes and tags in [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi)
- function [`load_model`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/nn_utils.py#L37) - how to load model to the device (cpu or/and gpu) - [link](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/sly_serve.py#L165)
- function [`inference`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/nn_utils.py#L62)  - how to apply model to the image and how to convert predictions to [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi)
