import supervisely_lib as sly
from supervisely_lib.io.fs import download, file_exists, get_file_name, get_file_name_with_ext
import os
import pathlib
import sys
import torch
import torch.nn as nn
import yaml
import onnxruntime as rt

import numpy as np
from PIL import Image

import supervisely_lib as sly
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)
import models
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from models.common import Conv, DWConv

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
TASK_ID = int(os.environ['TASK_ID'])

customWeightsPath = os.environ['modal.state.slyFile']

meta: sly.ProjectMeta = None
DEVICE_STR = os.environ['modal.state.device']
final_weights = None
half = None
device = select_device(device=DEVICE_STR)
imgsz = None
IMG_SIZE = 640
stride = None

image_size = int(os.environ['modal.state.imageSize'])
ts = None
CONFIDENCE = "confidence"


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


def export_to_torch_script(weights, img, model):
    global ts
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        # ts = optimize_for_mobile(ts)  # https://pytorch.org/tutorials/recipes/script_optimized.html
        ts.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')
        raise FileNotFoundError(e)


def export_to_onnx(weights, img, model, dynamic, simplify):
    prefix = colorstr('ONNX:')
    try:
        import onnx

        print(f'{prefix} starting export with onnx {onnx.__version__}...')
        f = weights.replace('.pt', '.onnx')  # filename
        # torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
        #                   dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
        #                                 'output': {0: 'batch', 2: 'y', 3: 'x'}} if dynamic else None)
        # # Checks
        # model_onnx = onnx.load(f)  # load onnx model
        # onnx.checker.check_model(model_onnx)  # check onnx model

        opset_version = 12
        train = False
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset_version,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch',  2: 'y', 3: 'x'}  # shape(1,25200,85)
                                        } if dynamic else None) # 'output': {0: 'batch', 1: 'anchors'}

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # # Simplify
        # if simplify:
        #     try:
        #         check_requirements(['onnx-simplifier'])
        #         import onnxsim
        #
        #         print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
        #         model_onnx, check = onnxsim.simplify(model_onnx,
        #                                              dynamic_input_shape=dynamic,
        #                                              input_shapes={'images': list(img.shape)} if dynamic else None)
        #         assert check, 'assert check failed'
        #         onnx.save(model_onnx, f)
        #     except Exception as e:
        #         print(f'{prefix} simplifier failure: {e}')
        #         pass
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')
        pass


def export_to_core_ml(weights, img):
    prefix = colorstr('CoreML:')
    try:
        import coremltools as ct

        print(f'{prefix} starting export with coremltools {ct.__version__}...')
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')
        pass


def download_weights(api, customWeightsPath):
    remote_path = customWeightsPath
    weights_path = os.path.join(my_app.data_dir, get_file_name_with_ext(remote_path))
    try:
        api.file.download(team_id=TEAM_ID,
                          remote_path=remote_path,
                          local_save_path=weights_path)
        return weights_path
    except:
        raise FileNotFoundError('FileNotFoundError')
        return None


def get_image():
    image = None
    images_list = [i for i in os.listdir(my_app.data_dir) if '.jp' in i]

    image_path = os.path.join(my_app.data_dir, images_list[0])
    image_path_exists = os.path.exists(image_path)
    if image_path_exists:
        # image = sly.image.read(image_path)
        image = Image.open(image_path)
        image_orig = image
        sly.image.write("vis.jpg", np.array(image))
        image = image.resize((IMG_SIZE, IMG_SIZE))
        sly.image.write("vis640.jpg", np.array(image))
    return image, image_orig


def transform_image_to_tensor(img0: np.ndarray, stride):
    img = letterbox(img0, new_shape=imgsz, stride=stride)[0]
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = torch.tensor(img, dtype=torch.float16).to(device)
    img = img.float()  # img.half() if half else img.float()  # uint8 to fp16/32
    # img = img.to(torch.float16)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@my_app.callback("export_weights")
@sly.timeit
def export_weights(api: sly.Api, task_id, context, state, app_logger):
    batch_size = 1
    img_size = [image_size, image_size]  # [640, 640]
    grid = True

    weights_path = download_weights(api, customWeightsPath)
    img_size *= 2 if len(img_size) == 1 else 1
    # set_logging()
    model = attempt_load(weights=weights_path, map_location=device)

    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = not grid  # set Detect() layer grid export
    for _ in range(2):
        y = model(img)  # dry runs
    print(f"\n{colorstr('PyTorch:')} starting from {weights_path} ({file_size(weights_path):.1f} MB)")

    image, image_orig = get_image()
    img0 = np.array(image)  # RGB

    global imgsz
    if hasattr(model, 'module') and hasattr(model.module, 'img_size'):
        imgsz = model.module.img_size[0]
    elif hasattr(model, 'img_size'):
        imgsz = model.img_size[0]
    else:
        sly.logger.warning(f"Image size is not found in model checkpoint. Use default: {IMG_SIZE}")
        imgsz = IMG_SIZE

    img = transform_image_to_tensor(img0, stride=int(model.stride.max()))

    # @TODO: fix export_to_onnx for cuda:0
    # ========================================================================
    export_to_torch_script(weights_path, img, model)  #
    export_to_onnx(weights_path, img, model, dynamic=False, simplify=False)  #
    # export_to_core_ml(weights_path, img)  #
    # ========================================================================

    augment = False
    torch_script_model_save = os.path.join(my_app.data_dir, 'best.torchscript.pt')
    torch_script_model = torch.jit.load(torch_script_model_save)

    onnx_model_save = os.path.join(my_app.data_dir, 'best.onnx')
    onnx_model = rt.InferenceSession(onnx_model_save)
    input_name = onnx_model.get_inputs()[0].name
    label_name = onnx_model.get_outputs()[0].name

    inf_out = model(img, augment=augment)[0]
    ts_out = torch_script_model(img)[0]
    onnx_out = onnx_model.run([label_name], {input_name: to_numpy(img).astype(np.float32)})[0]

    difference1 = inf_out - ts_out
    difference2 = to_numpy(inf_out) - onnx_out
    # # Apply NMS
    # labels = []
    conf_thres = 0.1
    iou_thres = 0.45
    agnostic_nms = True
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=agnostic_nms)
    output1 = non_max_suppression(ts_out, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=agnostic_nms)
    output2 = non_max_suppression(torch.tensor(onnx_out), conf_thres=conf_thres, iou_thres=iou_thres, agnostic=agnostic_nms)

    meta = construct_model_meta(model)
    names = model.module.names if hasattr(model, 'module') else model.names
    labels = []
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

    vis = np.copy(img0)
    ann.draw_contour(vis, thickness=2)
    sly.image.write("vis_detection.jpg", vis)

    process_folder = str(pathlib.Path(weights_path).parents[0])
    remote_path_template = str(pathlib.Path(remote_path).parents[0])
    for file in os.listdir(process_folder):
        file_path = os.path.join(process_folder, file)
        remote_file_path = os.path.join(remote_path_template, file)
        if '.onnx' in file_path or '.mlmodel' in file_path or '.torchscript' in file_path:
            api.file.upload(team_id=TEAM_ID, src=file_path, dst=remote_file_path)
    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": TEAM_ID,
        "context.workspaceId": WORKSPACE_ID,
        "modal.state.weightsPath": customWeightsPath
    })

    my_app.run(initial_events=[{"command": "export_weights"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
