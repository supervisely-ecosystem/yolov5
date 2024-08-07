# Description: This file contains versioning features and the Workflow class that is used to add input and output to the workflow.

import supervisely as sly
import os


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            try:
                self.is_compatible = self.check_instance_ver_compatibility()
            except Exception as e:
                sly.logger.error(
                    "Can not check compatibility with Supervisely instance. "
                    f"Workflow and versioning features will be disabled. Error: {repr(e)}"
                )
                self.is_compatible = False
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self._min_instance_version = (
            "6.9.31" if min_instance_version is None else min_instance_version
        )
    
    def check_instance_ver_compatibility(self):
        if not self.api.is_version_supported(self._min_instance_version):
            sly.logger.info(
                f"Supervisely instance version {self.api.instance_version} does not support workflow and versioning features."
            )
            if not sly.is_community():
                sly.logger.info(
                    f"To use them, please update your instance to version {self._min_instance_version} or higher."
                )
            return False
        return True
    
    @check_compatibility
    def add_input(self, project_info: sly.ProjectInfo, state: dict):
        try:
            project_version_id = self.api.project.version.create(
                project_info, "Train YOLO v5", f"This backup was created automatically by Supervisely before the Train YOLO task with ID: {self.api.task_id}"
            )
        except Exception as e:
            sly.logger.warning(f"Failed to create a project version: {repr(e)}")
            project_version_id = None
            
        try:
            if project_version_id is None:
                project_version_id = project_info.version.get("id", None) if project_info.version else None
            self.api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
            file_info = False
            if state["weightsInitialization"] is not None and state["weightsInitialization"] == "custom":
                file_info = self.api.file.get_info_by_path(sly.env.team_id(), state["_weightsPath"])
                self.api.app.workflow.add_input_file(file_info, model_weight=True)
            sly.logger.debug(f"Workflow Input: Project ID - {project_info.id}, Project Version ID - {project_version_id}, Input File - {True if file_info else False}")
        except Exception as e:
            sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")

    @check_compatibility
    def add_output(self, state: dict, team_files_dir: str):
        try:
            weights_dir_in_team_files = os.path.join(team_files_dir, "weights")
            files_info = self.api.file.list(sly.env.team_id(), weights_dir_in_team_files, return_type="fileinfo")
            best_filename_info = None
            for file_info in files_info:
                if "best" in file_info.name:
                    best_filename_info = file_info
                    break
            if best_filename_info:
                module_id = self.api.task.get_info_by_id(self.api.task_id).get("meta", {}).get("app", {}).get("id")                       
                if state["weightsInitialization"] is not None and state["weightsInitialization"] == "custom":
                    model_name = "Custom Model"
                else:
                    model_name = "YOLOv5"
                
                meta = {
                    "customNodeSettings": {
                    "title": f"<h4>Train {model_name}</h4>",
                    "mainLink": {
                        "url": f"/apps/{module_id}/sessions/{self.api.task_id}" if module_id else f"apps/sessions/{self.api.task_id}",
                        "title": "Show Results"
                    }
                },
                "customRelationSettings": {
                    "icon": {
                        "icon": "zmdi-folder",
                        "color": "#FFA500",
                        "backgroundColor": "#FFE8BE"
                    },
                    "title": "<h4>Checkpoints</h4>",
                    "mainLink": {"url": f"/files/{best_filename_info.id}/true", "title": "Open Folder"}
                    }
                }
                sly.logger.debug(f"Workflow Output: Team Files dir - {team_files_dir}, Best filename - {best_filename_info.name}")
                sly.logger.debug(f"Workflow Output: meta \n    {meta}")
                self.api.app.workflow.add_output_file(best_filename_info, model_weight=True, meta=meta)
            else:
                sly.logger.debug(f"File with the best weighs not found in Team Files. Cannot set workflow output.")
        except Exception as e:
            sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
