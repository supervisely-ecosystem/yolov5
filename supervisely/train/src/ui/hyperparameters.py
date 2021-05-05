import supervisely_lib as sly
import supervisely.train.src.sly_train_globals as g


def init(state):
    state["epochs"] = 10
    state["batchSize"] = 16
    state["imgSize"] = 640
    state["multiScale"] = False
    state["singleClass"] = False
    state["device"] = '0'
    state["workers"] = 8  # 0 - for debug
    state["activeTabName"] = "General"
    state["hyp"] = {
        "scratch": g.scratch_str,
        "finetune": g.finetune_str,
    }
    state["hypRadio"] = "scratch"
    state["optimizer"] = "SGD"
