import os
from pathlib import Path
import sys
import yaml
import supervisely as sly
from supervisely.nn.artifacts.yolov5 import YOLOv5
from supervisely.app.v1.app_service import AppService
from dotenv import load_dotenv
from workflow import Workflow

root_source_dir = str(Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"Source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

# currently PYTHONPATH is ":/app/repo"
python_path = os.environ.get("PYTHONPATH", "/app/repo").replace(":", "")
sys.path.insert(0, python_path)

if not sly.is_production():
    load_dotenv(os.path.join(root_source_dir, "supervisely", "train", "debug.env"))
    load_dotenv(os.path.join(root_source_dir, "supervisely", "train", "secret_debug.env"), override=True)

my_app = AppService()
# my_app._ignore_stop_for_debug = True

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

api: sly.Api = my_app.public_api
workflow = Workflow(api)
task_id = my_app.task_id

local_artifacts_dir = None
remote_artifacts_dir = None
project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
project_stats = api.project.get_stats(project_id)

with open(os.path.join(root_source_dir, "data/hyp.scratch.yaml"), 'r') as file:
    scratch_str = file.read()  # yaml.safe_load(

with open(os.path.join(root_source_dir, "data/hyp.finetune.yaml"), 'r') as file:
    finetune_str = file.read()  # yaml.safe_load(


runs_dir = os.path.join(sly.app.get_synced_data_dir(), 'runs')
sly.fs.mkdir(runs_dir, remove_content_if_exists=True)  # for debug, does nothing in production
experiment_name = str(task_id)
local_artifacts_dir = os.path.join(runs_dir, experiment_name)
sly.logger.info(f"All training artifacts will be saved to local directory {local_artifacts_dir}")

sly_yolov5 = YOLOv5(team_id)
framework_dir = sly_yolov5.framework_folder

remote_artifacts_dir = os.path.join(framework_dir, project_info.name, experiment_name)
remote_artifacts_dir = api.file.get_free_dir_name(team_id, remote_artifacts_dir)

remote_weights_dir = sly_yolov5.get_weights_path(remote_artifacts_dir)
remote_weights_dir = api.file.get_free_dir_name(team_id, remote_artifacts_dir)

sly.logger.info(f"After training artifacts will be uploaded to Team Files: {remote_artifacts_dir}")
