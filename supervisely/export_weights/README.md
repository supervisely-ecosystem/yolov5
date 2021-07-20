<div align="center" markdown>
<img src="https://i.imgur.com/csTZRio.png"/>

# Export YOLOv5 weights

<p align="center">
  <a href="#Overview">Overview</a>
  <a href="#How-To-Use">How To Use</a>
  <a href="#Infer-models">Infer models</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5/supervisely/export_weights)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App exports pretrained YOLO v5 model weights to [Torchscript](https://pytorch.org/docs/stable/jit.html?highlight=model%20features)(.torchscript.pt), [ONNX](https://onnx.ai/index.html)(.onnx), [CoreML](https://coremltools.readme.io/docs)(.mlmodel) formats. 

# How To Run
**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/import-mot-format) if it is not there.

**Step 2**: Find your pretrained model weights file in `Team Files`, open file context menue(right click on it) -> `Run App` -> `Export YOLOv5 weights`.

<img src="https://i.imgur.com/uzMlQ2e.png" width="800px"/>

**Step 3**: Set image size in modal window. Also in advanced section you can change what agent should be used for deploy.

<img src="https://i.imgur.com/7q7wLKW.png" width="600px"/>

**Step 4**: Press `Run` button. Now application log window will be opened. You can safely close it.

<img src="https://i.imgur.com/zjXgxhg.png"/>

**Step 5**: Converted model files will be placed to source weight file folder:
 - `{source weights filename}.onnx`
 - `{source weights filename}.torchscript.pt`

<img src="https://i.imgur.com/415Ijbk.png"/>

# Infer models
**saved model loading and usage**
```
import numpy as np

import torch
import onnx
import onnxruntime as rt


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


N = 1 # batch size
C = 3 # number of channels
H = 640 # image height
W = 640 # image width

# generate random tensor for inference
# Tensor valueas have to be distributed in range [0.0, 1.0] (if tensor values distributed in range [0, 255], 
divide tensor to 255.0) and tensor spartial values must match with model input image's spartial values:

tensor = torch.randn(N,C,H,W)
```
**TorchScript**
```
torch_script_model = torch.jit.load(path_to_torch_script_saved_model)
torch_script_model_inference = torch_script_model(tensor)[0]
```
 
**ONNX**
```    
onnx_model = rt.InferenceSession(path_to_onnx_saved_model)
input_name = onnx_model.get_inputs()[0].name
label_name = onnx_model.get_outputs()[0].name
onnx_model_inference = onnx_model.run([label_name], {input_name: to_numpy(tensor).astype(np.float32)})[0]
```
Pass inference result through [non_max_suppression](https://github.com/supervisely-ecosystem/yolov5/blob/0138090cd8d6f15e088246f16ca3240854bbba12/utils/general.py#L455): ([explanation](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)) with default settings for YOLOv5: 
 - `conf_thres=0.25`
 - `iou_thres=0.45`
 - `agnostic=False`
```
torchScript_output = non_max_suppression(torch_script_model_inference, conf_thres=0.25, iou_thres=0.45, agnostic=False)
onnx_output = non_max_suppression(onnx_model_inference, conf_thres=0.25, iou_thres=0.45, agnostic=False)
```
Each row of `output` tensor will have 6 positional values, representing: `top`, `left`, `bot`, `right`, `confidence`, `label_mark`

To get fast visualization, use following code:
```
# img0: torch.Tensor([1, 3, 640, 640]) - image(tensor) for inference
```
More info about `construct_model_meta` [here](https://github.com/supervisely-ecosystem/yolov5/blob/0138090cd8d6f15e088246f16ca3240854bbba12/supervisely/serve/src/nn_utils.py#L16)
```
# metadata for YOLOv5
meta = construct_model_meta(model)

# class_names
names = model.module.names if hasattr(model, 'module') else model.names

labels = []
for i, det in enumerate(output):
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            top, left, bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
            rect = sly.Rectangle(top, left, bottom, right)
            obj_class = meta.get_obj_class(names[int(cls)])
            tag = sly.Tag(meta.get_tag_meta("confidence"), round(float(conf), 4))
            label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
            labels.append(label)

height, width = img0.shape[:2]
ann = sly.Annotation(img_size=(height, width), labels=labels)

vis = np.copy(img0)
ann.draw_contour(vis, thickness=2)
sly.image.write("vis_detection.jpg", vis)
```
