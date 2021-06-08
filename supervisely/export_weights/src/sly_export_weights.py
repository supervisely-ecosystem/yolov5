import supervisely_lib as sly
from supervisely_lib.io.fs import download, file_exists, get_file_name, get_file_name_with_ext
import os
import pathlib
import sys
import torch
import torch.nn as nn

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

customWeightsPath = os.environ['modal.state.slyFile']
# DEVICE_STR = os.environ['modal.state.device']
# _img_size = int(os.environ['modal.state.imageSize'])
final_weights = None
ts = None


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
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                check_requirements(['onnx-simplifier'])
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
                pass
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


@my_app.callback("export_weights")
@sly.timeit
def export_weights(api: sly.Api, task_id, context, state, app_logger):
    batch_size = 1
    img_size = [640, 640]  # [_img_size, _img_size]
    DEVICE_STR = 'cpu'
    grid = True
    remote_path = customWeightsPath
    weights_path = os.path.join(my_app.data_dir, get_file_name_with_ext(remote_path))
    try:
        api.file.download(team_id=TEAM_ID,
                          remote_path=remote_path,
                          local_save_path=weights_path)
    except:
        raise FileNotFoundError('FileNotFoundError')

    img_size *= 2 if len(img_size) == 1 else 1
    # set_logging()
    device = select_device(device=DEVICE_STR)
    model = attempt_load(weights=weights_path, map_location=device)
    model = model.train()
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
    # print(f"\n{colorstr('PyTorch:')} starting from {weights_path} ({file_size(weights_path):.1f} MB)")

    # @TODO: fix export_to_onnx for cuda:0
    # ========================================================================
    export_to_torch_script(weights_path, img, model)                         #
    export_to_onnx(weights_path, img, model, dynamic=False, simplify=False)  #
    export_to_core_ml(weights_path, img)                                     #
    # ========================================================================

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
