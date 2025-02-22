<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/106374579/183668719-361e275f-e6f8-47e5-994d-3341c0e6241f.png"/>

# Serve YOLOv5

<p align="center">
  <a href="#Overview">Overview</a> •
    <a href="#Related-Apps">Related Apps</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#For-Developers">For Developers</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/yolov5/supervisely/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/yolov5/supervisely/serve)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/yolov5/supervisely/serve)](https://supervisely.com)

</div>

# Overview

App deploys YOLO v5 model (pretrained on COCO or custom one) as REST API service. Serve app is the simplest way how any model can be integrated into Supervisely. Once model is deployed, user gets the following benefits:

1. Use out of the box apps for inference
   - used directly in [labeling interface](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) (images, videos)
   - apply to [images project or dataset](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset)
   - apply to videos (coming soon)
2. Apps from Supervisely Ecosystem can use NN predictions: for visualization, for analysis, performance evaluation, etc ...
3. Communicate with NN in custom python script (see section <a href="#For-developers">for developers</a>)
4. App illustrates how to use NN weights. For example: you can train model in Supervisely, download its weights and use them the way you want.

Model serving allows to apply model to image (URL, local file, Supervisely image id) with 3 modes (full image, image ROI, sliding window). Also app sources can be used as example how to use downloaded model weights outside Supervisely.

**Watch usage demo:**

<a data-key="sly-embeded-video-link" href="https://youtu.be/cMBhn1Erluk" data-video-code="cMBhn1Erluk">
    <img src="https://i.imgur.com/UlEMeem.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

# Related Apps

You can use served model in next Supervisely Applications ⬇️ 
  

- [Apply NN to Images Project](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" height="70px" margin-bottom="20px"/>  

- [Apply NN to Videos Project](https://ecosystem.supervisely.com/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [NN Image Labeling](https://ecosystem.supervisely.com/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>




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


## Python Example: how to communicate with deployed model 
```python
import json
import yaml
import numpy as np
import supervisely_lib as sly


def visualize(img: np.ndarray, ann: sly.Annotation, name, roi: sly.Rectangle = None):
    vis = img.copy()
    if roi is not None:
        roi.draw_contour(vis, color=[255, 0, 0], thickness=3)
    ann.draw_contour(vis, thickness=3)
    sly.image.write(f"./images/{name}", vis)


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 2723

    # get information about model
    info = api.task.send_request(task_id, "get_session_info", data={})
    print("Information about deployed model:")
    print(json.dumps(info, indent=4))

    # get model output classes and tags
    meta_json = api.task.send_request(task_id, "get_output_classes_and_tags", data={})
    model_meta = sly.ProjectMeta.from_json(meta_json)
    print("Model produces following classes and tags")
    print(model_meta)

    # get model inference settings (optional)
    resp = api.task.send_request(task_id, "get_custom_inference_settings", data={})
    settings_yaml = resp["settings"]
    settings = yaml.safe_load(settings_yaml)
    # you can change this default settings and pass them to any inference method
    print("Model inference settings:")
    print(json.dumps(settings, indent=4))

    # inference for url
    image_url = "https://i.imgur.com/tEkCb69.jpg"

    # download image for further debug visualizations
    save_path = f"./images/{sly.fs.get_file_name_with_ext(image_url)}"
    sly.fs.ensure_base_path(save_path)  # create directories if needed
    sly.fs.download(image_url, save_path)
    img = sly.image.read(save_path)  # RGB

    # apply model to image URl (full image)
    # you can pass 'settings' dictionary to any inference method
    # every model defines custom inference settings
    ann_json = api.task.send_request(task_id, "inference_image_url",
                                     data={
                                         "image_url": image_url,
                                         "settings": settings,
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "01_prediction_url.jpg")

    # apply model to image URL (only ROI - region of interest)
    height, width = img.shape[0], img.shape[1]
    top, left, bottom, right = 0, 0, height - 1, int(width/2)
    roi = sly.Rectangle(top, left, bottom, right)
    ann_json = api.task.send_request(task_id, "inference_image_url",
                                     data={
                                         "image_url": image_url,
                                         "rectangle": [top, left, bottom, right]
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "02_prediction_url_roi.jpg", roi)

    # apply model to image id (full image)
    image_id = 770730
    ann_json = api.task.send_request(task_id, "inference_image_id", data={"image_id": image_id})
    ann = sly.Annotation.from_json(ann_json, model_meta)
    img = api.image.download_np(image_id)
    visualize(img, ann, "03_prediction_id.jpg")

    # apply model to image id (only ROI - region of interest)
    image_id = 770730
    img = api.image.download_np(image_id)
    height, width = img.shape[0], img.shape[1]
    top, left, bottom, right = 0, 0, height - 1, int(width / 2)
    roi = sly.Rectangle(top, left, bottom, right)
    ann_json = api.task.send_request(task_id, "inference_image_id",
                                     data={
                                         "image_id": image_id,
                                         "rectangle": [top, left, bottom, right]
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "04_prediction_id_roi.jpg", roi)

    # apply model to several images (using id)
    batch_ids = [770730, 770727, 770729, 770720]
    resp = api.task.send_request(task_id, "inference_batch_ids", data={"batch_ids": batch_ids})
    for ind, (image_id, ann_json) in enumerate(zip(batch_ids, resp)):
        ann = sly.Annotation.from_json(ann_json, model_meta)
        img = api.image.download_np(image_id)
        visualize(img, ann, f"05_prediction_batch_{ind:03d}_{image_id}.jpg")


if __name__ == "__main__":
    main()
```

## Python Example: how to apply model to raw images

You can do inference on image bytes from request when your Serve app is started.

1. Run Serve YOLOv5 app. Open the app's log.
2. Find the string `✅ To access the app in browser, copy and paste this URL:` and copy the link after this text.
3. Open [supervisely/serve/src/demo_request_raw_image.py](https://github.com/supervisely-ecosystem/yolov5/tree/master/supervisely/serve/src/demo_request_raw_image.py) script.
4. Paste copied link to variable `APP_ADDRESS` instead of current value.
5. Run this script from `supervisely/serve/src` path by command:

`python demo_request_raw_image.py`

You will get the result annotation in json format.

## Example Output

Information about deployed model:

```json
{
    "app": "YOLOv5 serve",
    "weights": "https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt",
    "device": "cuda:0",
    "half": "True",
    "input_size": 640,
    "session_id": "2723",
    "classes_count": 80,
    "tags_count": 1
}
```

Model produces following classes and tags:
```
ProjectMeta:
Object Classes
+----------------+-----------+----------------+--------+
|      Name      |   Shape   |     Color      | Hotkey |
+----------------+-----------+----------------+--------+
|     person     | Rectangle | [36, 15, 138]  |        |
|    bicycle     | Rectangle | [113, 138, 15] |        |
|      car       | Rectangle | [138, 15, 53]  |        |
|   motorcycle   | Rectangle | [15, 138, 101] |        |
|    airplane    | Rectangle | [138, 75, 15]  |        |
|      bus       | Rectangle | [20, 138, 15]  |        |
|     train      | Rectangle | [125, 15, 138] |        |
|     truck      | Rectangle | [15, 73, 138]  |        |
|      boat      | Rectangle | [15, 127, 138] |        |
| traffic light  | Rectangle | [138, 15, 102] |        |
|  fire hydrant  | Rectangle | [15, 138, 55]  |        |
|   stop sign    | Rectangle | [138, 24, 15]  |        |
| parking meter  | Rectangle | [65, 138, 15]  |        |
|     bench      | Rectangle | [79, 15, 138]  |        |
|      bird      | Rectangle | [138, 116, 15] |        |
|      cat       | Rectangle | [15, 37, 138]  |        |
|      dog       | Rectangle | [15, 98, 138]  |        |
|     horse      | Rectangle | [138, 48, 15]  |        |
|     sheep      | Rectangle | [138, 15, 79]  |        |
|      cow       | Rectangle | [138, 15, 127] |        |
|    elephant    | Rectangle | [15, 138, 124] |        |
|      bear      | Rectangle | [89, 138, 15]  |        |
|     zebra      | Rectangle | [135, 138, 15] |        |
|    giraffe     | Rectangle | [15, 138, 77]  |        |
|    backpack    | Rectangle | [138, 15, 27]  |        |
|    umbrella    | Rectangle | [101, 15, 138] |        |
|    handbag     | Rectangle | [17, 15, 138]  |        |
|      tie       | Rectangle | [15, 138, 33]  |        |
|    suitcase    | Rectangle | [40, 138, 15]  |        |
|    frisbee     | Rectangle | [138, 96, 15]  |        |
|      skis      | Rectangle | [60, 15, 138]  |        |
|   snowboard    | Rectangle | [15, 55, 138]  |        |
|  sports ball   | Rectangle | [15, 114, 138] |        |
|      kite      | Rectangle | [138, 15, 66]  |        |
|  baseball bat  | Rectangle | [52, 138, 15]  |        |
| baseball glove | Rectangle | [138, 129, 15] |        |
|   skateboard   | Rectangle | [101, 138, 15] |        |
|   surfboard    | Rectangle | [138, 36, 15]  |        |
| tennis racket  | Rectangle | [138, 61, 15]  |        |
|     bottle     | Rectangle | [15, 138, 89]  |        |
|   wine glass   | Rectangle | [77, 138, 15]  |        |
|      cup       | Rectangle | [138, 15, 115] |        |
|      fork      | Rectangle | [15, 138, 21]  |        |
|     knife      | Rectangle | [48, 15, 138]  |        |
|     spoon      | Rectangle | [138, 15, 41]  |        |
|      bowl      | Rectangle | [15, 25, 138]  |        |
|     banana     | Rectangle | [138, 106, 15] |        |
|     apple      | Rectangle | [137, 15, 138] |        |
|    sandwich    | Rectangle | [15, 86, 138]  |        |
|     orange     | Rectangle | [114, 15, 138] |        |
|    broccoli    | Rectangle | [90, 15, 138]  |        |
|     carrot     | Rectangle | [15, 138, 136] |        |
|    hot dog     | Rectangle | [15, 138, 67]  |        |
|     pizza      | Rectangle | [138, 85, 15]  |        |
|     donut      | Rectangle | [138, 15, 17]  |        |
|      cake      | Rectangle | [15, 46, 138]  |        |
|     chair      | Rectangle | [124, 138, 15] |        |
|     couch      | Rectangle | [138, 15, 88]  |        |
|  potted plant  | Rectangle | [30, 138, 15]  |        |
|      bed       | Rectangle | [15, 138, 44]  |        |
|  dining table  | Rectangle | [69, 15, 138]  |        |
|     toilet     | Rectangle | [15, 138, 114] |        |
|       tv       | Rectangle | [27, 15, 138]  |        |
|     laptop     | Rectangle | [138, 15, 72]  |        |
|     mouse      | Rectangle | [15, 106, 138] |        |
|     remote     | Rectangle | [15, 133, 138] |        |
|    keyboard    | Rectangle | [15, 63, 138]  |        |
|   cell phone   | Rectangle | [138, 68, 15]  |        |
|   microwave    | Rectangle | [138, 15, 34]  |        |
|      oven      | Rectangle | [95, 138, 15]  |        |
|    toaster     | Rectangle | [15, 121, 138] |        |
|      sink      | Rectangle | [15, 92, 138]  |        |
|  refrigerator  | Rectangle | [58, 138, 15]  |        |
|      book      | Rectangle | [138, 15, 95]  |        |
|     clock      | Rectangle | [138, 55, 15]  |        |
|      vase      | Rectangle | [15, 79, 138]  |        |
|    scissors    | Rectangle | [15, 19, 138]  |        |
|   teddy bear   | Rectangle | [138, 15, 47]  |        |
|   hair drier   | Rectangle | [15, 138, 27]  |        |
|   toothbrush   | Rectangle | [15, 138, 83]  |        |
+----------------+-----------+----------------+--------+
Tags
+------------+------------+-----------------+--------+---------------+--------------------+
|    Name    | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
+------------+------------+-----------------+--------+---------------+--------------------+
| confidence | any_number |       None      |        |      all      |         []         |
+------------+------------+-----------------+--------+---------------+--------------------+
```

Model inference settings:
```json
{
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "augment": false,
    "debug_visualization": false
}
```

Prediction for image URL (full image):

Image URL  |  `01_prediction_url.jpg`
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/tEkCb69.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/9OOoXn3.jpg" style="max-height: 300px; width: auto;"/>

Prediction for image URL (ROI - red rectangle):

Image URL + ROI  |  `02_prediction_url_roi.jpg`
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/tEkCb69.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/iSKS17L.jpg" style="max-height: 300px; width: auto;"/>


Prediction for image id (full image):

<table>
  <tr>
    <th>03_input_id.jpg</th>
    <th>03_prediction_id.jpg</th>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/RQDrH4B.jpg" height="300"/></td>
    <td><img src="https://i.imgur.com/yYujbI0.jpg" height="300"/></td>
  </tr>
</table>


Prediction for image id (ROI - red rectangle):

<table>
  <tr>
    <th>04_input_id_roi.jpg</th>
    <th>04_prediction_id_roi.jpg</th>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/2XlEZQK.jpg" height="300"/></td>
    <td><img src="https://i.imgur.com/1U7413M.jpg" height="300"/></td>
  </tr>
</table>

Prediction for batch of images ids:

<table>
  <tr>
    <th>Image ID</th>
    <th>Prediction</th>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/4Lh9tAm.jpg" height="300"/></td>
    <td><img src="https://i.imgur.com/emsah1q.jpg" height="300"/></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/UqiV5Ka.jpg" height="300"/></td>
    <td><img src="https://i.imgur.com/GhoKKCl.jpg" height="300"/></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/8GjoNDH.jpg"/></td>
    <td><img src="https://i.imgur.com/yzinXD6.jpg"/></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/xOydF3B.jpg" height="300"/></td>
    <td><img src="https://i.imgur.com/YFNmIPY.jpg" height="300"/></td>
  </tr>
</table>

# Acknowledgment

This app is based on the great work `yolov5` ([github](https://github.com/ultralytics/yolov5)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/yolov5?style=social)
