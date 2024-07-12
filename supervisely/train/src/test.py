import supervisely as sly
import os

api = sly.Api.from_env()
api.task_id = 62688
os.environ.setdefault("TEAM_ID", "451")
from workflow import Workflow

workflow = Workflow(api)

state = {"weightsInitialization": "custom"}

team_files_dir = "/yolov5_train/Train dataset - Eschikon Wheat Segmentation (EWS)/62688"

workflow.add_output(state, team_files_dir)