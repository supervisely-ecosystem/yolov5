import supervisely.train.src.sly_train_globals as g
import supervisely.train.src.ui.input_project as input_project
import supervisely.train.src.ui.training_classes as training_classes


def init(data, state):
    input_project.init(data)
    training_classes.init(g.api, data, state, g.project_id, g.project_meta)