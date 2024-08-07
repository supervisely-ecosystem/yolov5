import os
import supervisely as sly
import tqdm

import sly_train_globals as g

from sly_train_globals import \
    my_app, task_id, \
    team_id, workspace_id, project_id, \
    root_source_dir, scratch_str, finetune_str

import ui as ui
from sly_project_cached import download_project
from sly_train_utils import init_script_arguments
from sly_utils import get_progress_cb, upload_artifacts
from splits import get_train_val_sets, verify_train_val_sets
import yolov5_format as yolov5_format
from architectures import prepare_weights
from artifacts import set_task_output
import train as train_yolov5


@my_app.callback("restore_hyp")
@sly.timeit
def restore_hyp(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_field(task_id, "state.hyp", {
        "scratch": scratch_str,
        "finetune": finetune_str,
    })


@my_app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        use_cache = state.get("useCache", True)
        prepare_weights(state)

        # prepare directory for original Supervisely project
        project_dir = os.path.join(my_app.data_dir, "sly_project")
        sly.fs.mkdir(project_dir, remove_content_if_exists=True)  # clean content for debug, has no effect in prod

        # -------------------------------------- Add Workflow Input -------------------------------------- #
        g.workflow.add_input(g.project_info, state)
        # ----------------------------------------------- - ---------------------------------------------- #

        # download and preprocess Sypervisely project (using cache)
        try:
            download_project(
                api=api,
                project_info=g.project_info,
                project_dir=project_dir,
                use_cache=use_cache,
            )
        except Exception as e:
            sly.logger.warn("Can not download project")
            raise Exception(
                "Can not download the project. "
                f"Check if the project is available, it is not archived, not modified and not empty. "
                f"{repr(e)}"
            )

        # preprocessing: transform labels to bboxes, filter classes, ...
        try:
            sly.Project.to_detection_task(project_dir, inplace=True)
        except RuntimeError as e:
            if not use_cache:
                raise
            sly.logger.warn("Error during project transformation to detection task. Will try to re-download the project", exc_info=True)
            download_project(
                api=api,
                project_info=g.project_info,
                project_dir=project_dir,
                use_cache=False,
            )
            sly.Project.to_detection_task(project_dir, inplace=True)

        except ValueError:
            # search for problem images and ignore them
            images_info = []
            for dataset_info in api.dataset.get_list(project_id):
                images_info.extend(api.image.get_list(dataset_info.id))
            for image_info in images_info:
                ann_json = api.annotation.download(image_info.id).annotation
                for object in ann_json["objects"]:
                    if "points" not in object:
                        sly.logger.info(f"Ignoring image with id {image_info.id} since its annotation is not deserializable")
                        dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
                        dataset_dir = os.path.join(project_dir, dataset_info.name)
                        dataset = sly.Dataset(dataset_dir, mode=sly.OpenMode.READ)
                        dataset.delete_item(image_info.name)
            # when problem images are removed, we can transform project to detection task
            sly.Project.to_detection_task(project_dir, inplace=True)
            # update train and val sets taking deleted images into account
            imgs_before = g.project_info.items_count
            project = sly.Project(project_dir, sly.OpenMode.READ)
            imgs_after = project.total_items
            if imgs_after != imgs_before:
                val_count = state["randomSplit"]["count"]["val"]
                val_part = val_count / imgs_before
                new_val = round(imgs_after * val_part)
                if new_val < 1:
                    raise ValueError("Val split length == 0 after ignoring images. Please check your data.")
                new_train = imgs_after - new_val
                state["randomSplit"]["count"]["train"] = new_train
                state["randomSplit"]["count"]["val"] = new_val
            
        train_classes = state["selectedClasses"]
        sly.Project.remove_classes_except(project_dir, classes_to_keep=train_classes, inplace=True)
        
        if state["unlabeledImages"] == "ignore":
            imgs_before = g.project_info.items_count
            sly.Project.remove_items_without_objects(project_dir, inplace=True)
            project = sly.Project(project_dir, sly.OpenMode.READ)
            imgs_after = project.total_items
            if imgs_after != imgs_before:
                val_count = state["randomSplit"]["count"]["val"]
                val_part = val_count / imgs_before
                new_val = int(imgs_after * val_part)
                if new_val < 1:
                    raise ValueError("Val split length == 0 after ignoring images. Please check your data.")
                new_train = imgs_after - new_val
                state["randomSplit"]["count"]["train"] = new_train
                state["randomSplit"]["count"]["val"] = new_val


        # split to train / validation sets (paths to images and annotations)
        train_set, val_set = get_train_val_sets(project_dir, state)
        verify_train_val_sets(train_set, val_set)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")

        # prepare directory for data in YOLOv5 format (nn will use it for training)
        train_data_dir = os.path.join(my_app.data_dir, "train_data")
        sly.fs.mkdir(train_data_dir, remove_content_if_exists=True)  # clean content for debug, has no effect in prod

        # convert Supervisely project to YOLOv5 format
        progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", len(train_set) + len(val_set))
        yolov5_format.transform(project_dir, train_data_dir, train_set, val_set, progress_cb)

        # init sys.argv for main training script
        init_script_arguments(state, train_data_dir, g.project_info.name)

        # start train script
        api.app.set_field(task_id, "state.activeNames", ["labels", "train", "pred", "metrics"])  # "logs",
        get_progress_cb("YOLOv5: Scanning data ", 1)(1)
        train_yolov5.main(stop_event_check=g.my_app.is_stopped)

        # upload artifacts directory to Team Files
        upload_artifacts(g.local_artifacts_dir, g.remote_artifacts_dir)
        set_task_output()
        g.workflow.add_output(state, g.remote_artifacts_dir)
    except Exception as e:
        msg = f"Something went wrong. Find more info in the app logs."
        my_app.show_modal_window(f"{msg} {repr(e)}", level="error", log_message=False)
        sly.logger.error(repr(e), exc_info=True, extra={ 'exc_str': str(e)})
        try:
            api.task.set_output_error(task_id, repr(e), "Find more info in the app logs.")
            api.app.set_field(task_id, "state.started", False)
        except:
            pass

    # stop application
    get_progress_cb("Finished, app is stopped automatically", 1)(1)
    my_app.stop(wait=False)


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": team_id,
        "context.workspaceId": workspace_id,
        "modal.state.slyProjectId": project_id,
    })

    data = {}
    state = {}
    data["taskId"] = task_id

    my_app.compile_template(g.root_source_dir)

    # init data for UI widgets
    try:
        ui.init(data, state)
    except Exception as e:
        raise Exception(f"UI initialization error. {str(e)}")

    my_app.run(data=data, state=state)


# New features:
# @TODO: resume training
# @TODO: save checkpoint every N-th epochs
if __name__ == "__main__":
    sly.main_wrapper("main", main)
