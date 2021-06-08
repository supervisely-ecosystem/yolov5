<div align="center" markdown>
<img src=""/>

# Export YOLOv5 weights

<p align="center">
  <a href="#Overview">Overview</a>
  <a href="#How-To-Use">How To Use</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5/supervisely/export_weights)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App exports pretrained YOLO v5 model weights to Torchscript(.torchscript.pt), ONNX(.onnx), CoreML(.mlmodel) formats. 

# How To Run

1. App saves converted weight files to the directory: . 
   For example: `/yolov5_train/lemons_annotated/2577/weights/best.pt`

<img src="https://i.imgur.com/l0p7Z8F.png"/>
<img src="https://i.imgur.com/uzMlQ2e.png"/>


2. Paste path to modal window

<img src="" width="400"/>

Then

3. Choose device (optional): for GPU just provide device id (`0` or `1` or ...), or type `cpu`. Also in advanced section you can 
change what agent should be used for deploy.

4. Press `Run` button.

5. Wait until you see following message in logs: `Model has been successfully deployed`

<img src="" width="800"/>
