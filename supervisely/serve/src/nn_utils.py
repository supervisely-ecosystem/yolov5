import json

import torch
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.fs import get_file_name_with_ext
import os
from pathlib import Path
import yaml

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox

import serve_globals as g

import cv2

CONFIDENCE = "confidence"
IMG_SIZE = 640


def construct_model_meta(model):
    names = model.module.names if hasattr(model, 'module') else model.names

    colors = None
    if hasattr(model, 'module') and hasattr(model.module, 'colors'):
        colors = model.module.colors
    elif hasattr(model, 'colors'):
        colors = model.colors
    else:
        colors = []
        for i in range(len(names)):
            colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [sly.ObjClass(name, sly.Rectangle, color) for name, color in zip(names, colors)]
    tags = [sly.TagMeta(CONFIDENCE, sly.TagValueType.ANY_NUMBER)]

    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                           tag_metas=sly.TagMetaCollection(tags))
    return meta


def load_model(weights_path, imgsz=640, device='cpu'):
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights_path, map_location=device)  # load FP32 model
    configs_path = os.path.join(Path(weights_path).parents[0], 'opt.yaml')

    with open(configs_path, 'r') as stream:
        cfgs_loaded = yaml.safe_load(stream)

    if hasattr(model, 'module') and hasattr(model.module, 'img_size'):
        imgsz = model.module.img_size[0]
    elif hasattr(model, 'img_size'):
        imgsz = model.img_size[0]
    elif cfgs_loaded['img_size']:
        imgsz = cfgs_loaded['img_size'][0]
    else:
        sly.logger.warning(f"Image size is not found in model checkpoint. Use default: {IMG_SIZE}")
        imgsz = IMG_SIZE
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    return model, half, device, imgsz, stride


def inference(model, half, device, imgsz, stride, image: np.ndarray, meta: sly.ProjectMeta, conf_thres=0.25, iou_thres=0.45,
              augment=False, agnostic_nms=False, debug_visualization=False) -> sly.Annotation:
    names = model.module.names if hasattr(model, 'module') else model.names

    img0 = image # RGB
    # Padded resize
    img = letterbox(img0, new_shape=imgsz, stride=stride)[0]
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    inf_out = model(img, augment=augment)[0]

    # Apply NMS
    labels = []
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=agnostic_nms)
    for i, det in enumerate(output):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                top, left, bottom, right = int(xyxy[1]), int(xyxy[0]), int(xyxy[3]), int(xyxy[2])
                rect = sly.Rectangle(top, left, bottom, right)
                obj_class = meta.get_obj_class(names[int(cls)])
                tag = sly.Tag(meta.get_tag_meta(CONFIDENCE), round(float(conf), 4))
                label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
                labels.append(label)

    height, width = img0.shape[:2]
    ann = sly.Annotation(img_size=(height, width), labels=labels)

    if debug_visualization is True:
        # visualize for debug purposes
        vis = np.copy(img0)
        ann.draw_contour(vis, thickness=2)
        sly.image.write("vis.jpg", vis)

    return ann.to_json()


def get_frame_np(api, video_id, frame_index):
    img_rgb = api.video.frame.download_np(video_id, frame_index)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def dump_annotation(ann_path, ann_json):
    with open(ann_path, 'w') as file:
        json.dump(ann_json, file, indent=2)


def inference_images_dir(images_path, ann_path, context, state, app_logger):
    sly.fs.clean_dir(ann_path)

    image_names = os.listdir(images_path)

    for image_name in image_names:
        image_path = os.path.join(images_path, image_name)
        ann_json = inference_image_path(image_path, context, state, app_logger)
        dump_annotation(os.path.join(ann_path, f'{image_name}.json'), ann_json)


def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})

    rect = None
    if "rectangle" in state:
        top, left, bottom, right = state["rectangle"]
        rect = sly.Rectangle(top, left, bottom, right)

    settings = state.get("settings", {})
    for key, value in g.default_settings.items():
        if key not in settings:
            app_logger.warn("Field {!r} not found in inference settings. Use default value {!r}".format(key, value))
    debug_visualization = settings.get("debug_visualization", g.default_settings["debug_visualization"])
    conf_thres = settings.get("conf_thres", g.default_settings["conf_thres"])
    iou_thres = settings.get("iou_thres", g.default_settings["iou_thres"])
    augment = settings.get("augment", g.default_settings["augment"])

    image = sly.image.read(image_path)  # RGB image
    if rect is not None:
        canvas_rect = sly.Rectangle.from_size(image.shape[:2])
        results = rect.crop(canvas_rect)
        if len(results) != 1:
            return {
                "message": "roi rectangle out of image bounds",
                "roi": state["rectangle"],
                "img_size": {"height": image.shape[0], "width": image.shape[1]}
            }
        rect = results[0]
        image = sly.image.crop(image, rect)
    ann_json = inference(g.model, g.half, g.device, g.imgsz, g.stride, image, g.meta,
                         conf_thres=conf_thres, iou_thres=iou_thres, augment=augment,
                         debug_visualization=debug_visualization)
    return ann_json




