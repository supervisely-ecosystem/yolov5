import supervisely as sly


def init(api: sly.Api, data, state, project_meta: sly.ProjectMeta, stats):
    class_images = {}
    for item in stats["images"]["objectClasses"]:
        class_images[item["objectClass"]["name"]] = item["total"]
    class_objects = {}
    for item in stats["objects"]["items"]:
        class_objects[item["objectClass"]["name"]] = item["total"]

    classes_json = project_meta.obj_classes.to_json()
    for obj_class in classes_json:
        cls_title = obj_class["title"]
        if cls_title not in class_images or cls_title not in class_objects:
            raise Exception(
                f"Class '{cls_title}' not found in the project. Please check your data."
            )
        obj_class["imagesCount"] = class_images[cls_title]
        obj_class["objectsCount"] = class_objects[cls_title]

    unlabeled_count = 0
    for ds_counter in stats["images"]["datasets"]:
        unlabeled_count += ds_counter["imagesNotMarked"]

    data["classes"] = classes_json
    state["selectedClasses"] = []
    state["classes"] = len(classes_json) * [True]
    data["unlabeledCount"] = unlabeled_count