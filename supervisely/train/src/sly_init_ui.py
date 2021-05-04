import os
import supervisely_lib as sly

import sly_train_globals as globals
import sly_metrics as metrics
#from models_description import get_models_list


empty_gallery = {
    "content": {
        "projectMeta": sly.ProjectMeta().to_json(),
        "annotations": {},
        "layout": []
    }
}


def init_start_state(state):
    state["started"] = False
    state["activeNames"] = []


def init_galleries(data):
    data["vis"] = empty_gallery
    data["labelsVis"] = empty_gallery
    data["predVis"] = empty_gallery
    data["syncBindings"] = []


def init_progress(data):
    data["progressName"] = ""
    data["currentProgress"] = 0
    data["totalProgress"] = 0
    data["currentProgressLabel"] = ""
    data["totalProgressLabel"] = ""


def init_output(data):
    data["outputUrl"] = ""
    data["outputName"] = ""


def init(data, state):
    init_classes_stats(data, state, globals.project_meta)
    init_splits(globals.project_info, data, state)
    init_model_settings(data, state)
    init_training_hyperparameters(state)
    init_start_state(state)
    init_galleries(data)
    init_progress(data)
    init_output(data)
    metrics.init(data, state)


def set_output():
    file_info = globals.api.file.get_info_by_path(globals.team_id,
                                                  os.path.join(globals.remote_artifacts_dir, 'results.png'))
    fields = [
        {"field": "data.outputUrl", "payload": globals.api.file.get_url(file_info.id)},
        {"field": "data.outputName", "payload": globals.remote_artifacts_dir},
    ]
    globals.api.app.set_fields(globals.task_id, fields)
    globals.api.task.set_output_directory(globals.task_id, file_info.id, globals.remote_artifacts_dir)

