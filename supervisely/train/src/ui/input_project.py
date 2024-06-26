from supervisely.project.download import is_cached
import sly_train_globals as g


def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)
    data["isCached"] = is_cached(g.project_info.id)
    state["useCache"] = True