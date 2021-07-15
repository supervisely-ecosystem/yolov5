<div align="center" markdown>
<img src="https://i.imgur.com/csTZRio.png"/>

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
**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/import-mot-format) if it is not there.

**Step 2**: Find your pretrained model weights file in `Team Files`, open file context menue(right click on it) -> `Run App` -> `Export YOLOv5 weights`.

<img src="https://i.imgur.com/uzMlQ2e.png" width="800px"/>

**Step 3**: Set image size in modal window. Also in advanced section you can change what agent should be used for deploy.

<img src="https://i.imgur.com/7q7wLKW.png" width="600px"/>

**Step 4**: Press `Run` button. Now application log window will be opened. You can safely close it.

<img src="https://i.imgur.com/zjXgxhg.png"/>

5. Result files will be placed to source weight file folder:
 - `{source weights filename}.mlmodel`
 - `{source weights filename}.onnx`
 - `{source weights filename}.torchscript.pt`

<img src="https://i.imgur.com/415Ijbk.png"/>

```
import numpy as np
import torch

# N - batch size
# C - number of channels
# H - image height
# W - image width
tensor = torch.randn(N,C,H,W)
```
## TorchScript
saved model loading:
`torch_script_model = torch.jit.load(path_to_torch_script_saved_model)`
and usage:
`torch_script_model_inference = torch_script_model(tensor)`

## ONNX 
saved model loading:
```
import onnx
import onnxruntime as rt

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
onnx_model = rt.InferenceSession(path_to_onnx_saved_model)
input_name = onnx_model.get_inputs()[0].name
label_name = onnx_model.get_outputs()[0].name
```
and usage:
`onnx_model_inference = onnx_model.run([label_name], {input_name: to_numpy(tensor).astype(np.float32)})[0]`

## CoreML (converted models work only with MacOS Version > 10)
### [CoreML](https://coremltools.readme.io/docs) saved model loading:
```
import coremltools as ct

core_ml_model = ct.models.MLModel(path_to_core_ml_saved_model)
```
### and usage:
```
e = np.zeros((3,224,224)) 
d = {} 
d['data'] = e 
r = coreml_model.predict(d)
core_ml_model_inference = core_ml_model.predict({"image": to_numpy(tensor).astype(np.float32)})
```
