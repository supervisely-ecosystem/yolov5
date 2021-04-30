import supervisely.train.src.sly_train_globals as g
import supervisely.train.src.ui.input_project as input_project
import supervisely.train.src.ui.training_classes as training_classes
import supervisely.train.src.ui.train_val_split as train_val_split
import supervisely.train.src.ui.model_architectures as model_architectures


def init(data, state):
    input_project.init(data)
    training_classes.init(g.api, data, state, g.project_id, g.project_meta)
    train_val_split.init(g.project_info, g.project_meta, data, state)
    model_architectures.init(data, state)