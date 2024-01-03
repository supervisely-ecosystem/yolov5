import os
import supervisely as sly
from utils.torch_utils import select_device
from supervisely.app.v1.app_service import AppService

my_app = AppService()

TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
TASK_ID = sly.env.task_id()
customWeightsPath = sly.env.file(raise_not_found=False)
if customWeightsPath is None:
    raise Exception("Weights path is not found. Please, specify it in the modal.")
device = select_device(device='cpu')
image_size = 640
ts = None
batch_size = 1
grid = True
args = dict(my_app=my_app, TEAM_ID=TEAM_ID)
