from functools import partial
import os
import time
import supervisely as sly
import sly_train_globals as globals
from dataclasses import asdict
from supervisely.nn.artifacts.artifacts import TrainInfo


def update_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done_report(count)
    _update_progress_ui(api, task_id, progress)


def _update_progress_ui(api: sly.Api, task_id, progress: sly.Progress, stdout_print=False):
    if progress.need_report():
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current_label},
            {"field": "data.totalProgressLabel", "payload": progress.total_label},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)
        if stdout_print is True:
            progress.report_if_needed()


def get_progress_cb(message, total, is_size=False):
    progress = sly.Progress(message, total, is_size=is_size)
    progress_cb = partial(update_progress, api=globals.api, task_id=globals.task_id, progress=progress)
    progress_cb(0)
    return progress_cb


def update_uploading_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done(count - progress.current)
    _update_progress_ui(api, task_id, progress, stdout_print=True)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    name = "open_app.lnk"
    local_path = os.path.join(local_dir, name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts(local_dir, remote_dir):
    def _gen_message(current, total):
        return f"Upload artifacts to Team Files [{current}/{total}] "

    _save_link_to_ui(local_dir, globals.my_app.app_url)
    local_files = sly.fs.list_files_recursively(local_dir)
    total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])

    progress = sly.Progress(_gen_message(0, len(local_files)), total_size, is_size=True)
    progress_cb = partial(update_uploading_progress, api=globals.api, task_id=globals.task_id, progress=progress)
    progress_cb(0)

    for idx, local_path in enumerate(local_files):
        remote_path = os.path.join(remote_dir, local_path.replace(local_dir, '').lstrip("/"))
        if globals.api.file.exists(globals.team_id, remote_path):
            progress.iters_done_report(sly.fs.get_file_size(local_path))
        else:
            progress_last = progress.current
            globals.api.file.upload(globals.team_id, local_path, remote_path,
                                    lambda monitor: progress_cb(progress_last + monitor.bytes_read))
        progress.message = _gen_message(idx + 1, len(local_files))
        time.sleep(0.5)

    # generate metadata
    globals.sly_yolov5_generated_metadata = globals.sly_yolov5.generate_metadata(
        app_name=globals.sly_yolov5.app_name,
        task_id=globals.experiment_name,
        artifacts_folder=globals.remote_artifacts_dir,
        weights_folder=globals.remote_weights_dir,
        weights_ext=globals.sly_yolov5.weights_ext,
        project_name=globals.project_info.name,
        task_type=globals.sly_yolov5.task_type,
        config_path=None,
    )

def create_experiment(
    model_name,
    remote_dir,
    local_dir,
    report_id=None,
    eval_metrics=None,
    primary_metric_name=None,
):
    train_info = TrainInfo(**globals.sly_yolov5_generated_metadata)
    experiment_info = globals.sly_yolov5.convert_train_to_experiment_info(train_info)
    experiment_info.experiment_name = (
        f"{globals.task_id} {globals.project_info.name} {model_name}"
    )
    experiment_info.model_name = model_name
    experiment_info.framework_name = f"{globals.sly_yolov5.framework_name}"
    experiment_info.train_size = globals.train_size
    experiment_info.val_size = globals.val_size
    experiment_info.evaluation_report_id = report_id
    experiment_info.experiment_report_id = None
    if report_id is not None:
        experiment_info.evaluation_report_link = f"/model-benchmark?id={str(report_id)}"
    experiment_info.evaluation_metrics = eval_metrics

    experiment_info_json = asdict(experiment_info)
    experiment_info_json["project_preview"] = globals.project_info.image_preview_url
    experiment_info_json["primary_metric"] = primary_metric_name

    globals.api.task.set_output_experiment(globals.task_id, experiment_info_json)
    experiment_info_json.pop("project_preview")
    experiment_info_json.pop("primary_metric")

    experiment_info_path = os.path.join(local_dir, "experiment_info.json")
    remote_experiment_info_path = os.path.join(remote_dir, "experiment_info.json")
    sly.json.dump_json_file(experiment_info_json, experiment_info_path)
    globals.api.file.upload(globals.team_id, experiment_info_path, remote_experiment_info_path)
