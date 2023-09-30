import json
import os

import numpy as np
from num2words import num2words

ordinal = lambda x: num2words(x, ordinal=True)

EPS = 1.5


def flip(p=0.7):
    return np.random.random() < p


class SceneObject:
    def __init__(self, obj_id, position, category_name, category_id, **kwargs):
        self.obj_id = obj_id
        self.obj_name = category_name.replace("_", " ")
        self.position = numpify_vector(position)
        self.cat_id = numpify_vector(category_id)

    @classmethod
    def parse(cls, obj_id, params):
        return cls(obj_id=obj_id, **params)


class SceneHuman(SceneObject):
    def __init__(
        self, smpl_clothe, animation_fbx_name="", action_description="", **kwargs
    ):
        super().__init__(**kwargs)

        with open("resources/json/clothing.json") as f:
            clothe_description = json.load(f).get(smpl_clothe, [])

        self.clothe = clothe_description
        self.animation = animation_fbx_name
        self.action = action_description


def get_positional_relations_vertical(obj1: SceneObject, obj2: SceneObject):
    relations = []
    pos1, pos2 = obj1.position, obj2.position
    name1, name2 = obj1.obj_name, obj2.obj_name

    if abs(pos1[2] - pos2[2]) > EPS:
        n1, n2 = (name1, name2) if pos1[2] < pos2[2] else (name2, name1)
        relations.append(
            np.random.choice(
                [
                    f"the {n2} is in front of the {n1}",
                    f"the {n1} is behind the {n2}",
                ]
            )
        )
    return relations


def numpify_vector(vector, dtype=float):
    if isinstance(vector, np.ndarray):
        return vector

    if isinstance(vector, dict):
        return np.array(
            [
                vector["x"],
                vector["y"],
                vector["z"],
            ]
        )

    if isinstance(vector, list) or isinstance(vector, tuple):
        return np.array(vector)

    if vector.startswith("[") and vector.endswith("]"):
        return np.array([dtype(i) for i in vector[1:-1].split()])

    vector_eval = eval(vector)
    return numpify_vector(vector_eval)


def get_positional_relations_horizontal(img, obj1, obj2):
    name1, name2 = obj1.obj_name, obj2.obj_name
    obj1_points = np.where(np.all(img == obj1.cat_id, axis=-1))
    obj2_points = np.where(np.all(img == obj2.cat_id, axis=-1))

    if obj1_points[1].max() < obj2_points[1].min():
        return [
            f"the {name1} is to the left of the {name2}",
            f"the {name2} is to the right of the {name1}",
        ]

    if obj2_points[1].max() < obj1_points[1].min():
        return [
            f"the {name2} is to the left of the {name1}",
            f"the {name1} is to the right of the {name2}",
        ]

    return []


def get_view_matrix(camera_position, camera_look_at):
    # Up vector (assuming +y is up)
    up = np.array([0, 1, 0])
    # Compute the z-axis of the camera's coordinate system
    z = camera_position - camera_look_at
    z = z / np.linalg.norm(z)
    # Compute the x-axis of the camera's coordinate system
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    # Compute the y-axis of the camera's coordinate system
    y = np.cross(z, x)
    # Create the rotation matrix
    rotation = np.stack((x, y, z))
    # Create the translation vector
    translation = -camera_position
    # Create the view matrix
    view_matrix = np.concatenate((rotation, translation.reshape(3, 1)), axis=1)
    return view_matrix


def is_obj_in_img(img, obj_id, invalid_factor=0.003):
    img_area = img.shape[0] * img.shape[1]

    return np.all(img == obj_id, axis=-1).sum() >= invalid_factor * img_area


def is_valid_frame(category_image, human_id):
    """
    Codes:
    0: no human detected
    1: horizontal orientation
    2: vertical orientation, correct
    3: vertical orientation, flipped
    4: angled orientation, head on top
    5: angled orientation, head on bottom
    """
    if not isinstance(human_id, np.ndarray):
        human_id = np.asarray(eval(human_id))
        if not len(human_id):
            human_id = np.array([253, 214, 137])
    if not is_obj_in_img(category_image, human_id):
        # No human detected in the scene
        return 0

    y, x = np.where(np.all(category_image == human_id, axis=-1))
    x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    if abs(angle) < 30 or abs(angle - 180) < 30:
        # Human detected in horizontal position
        return 1

    height = y2 - y1
    top_width = np.where(category_image[y1 : y1 + height // 7, x1:x2] == human_id)[1]
    if len(top_width):
        top_width = top_width.max() - top_width.min()
    else:
        top_width = 0
    bottom_width = np.where(category_image[y2 - height // 7 : y2, x1:x2] == human_id)[1]
    if len(bottom_width):
        bottom_width = bottom_width.max() - bottom_width.min()
    else:
        bottom_width = 0
    flipped = top_width > bottom_width

    if abs(angle - 90) < 30 or abs(angle - 270) < 30:
        # Human detected in vertical position
        return 2 if not flipped else 3

    else:
        # Human detected in angled position
        return 4 if not flipped else 5


def transform_to_camera_coordinates(view_matrix, global_coords):
    global_coords = np.concatenate((global_coords, [1]))
    return view_matrix.dot(global_coords)


STATEMENT_PREFIX = "This scene contains"
PROMPT_PREFIX = "Q: Please describe a scene containing"
PROMPT_SUFFIX = "A: In this scene, we can see"


def sample_prompt(objects, seg_image, scene_name=None, p=0.7):
    statements = []
    # prompt
    statements.append(PROMPT_PREFIX)

    # Objects
    objects_statement = ""
    humans = set()
    for obj in objects:
        if isinstance(obj, SceneHuman):
            humans.add(obj)
            continue
        obj_name = obj.obj_name
        objects_statement += (
            "an" if obj_name[0].lower() in "aeoiu" else "a"
        ) + f" {obj_name}, "

    # objects_statement = objects_statement[:-2]
    if humans:
        if objects_statement:
            objects_statement += "and "
        objects_statement += (
            f"{num2words(len(humans))} humans." if len(humans) > 1 else "one human."
        )
    else:
        objects_statement = objects_statement[:-2] + "."
    statements.append(objects_statement)

    # Scene statement
    if scene_name:
        scene_statement = "They are in "
        with open("resources/json/scenes.json") as f:
            scene_description = json.load(f).get(scene_name, "")
        if scene_description and flip(p):
            scene_statement += "an" if scene_description[0].lower() in "aeoiu" else "a"
            scene_statement += " " + scene_description.strip() + "."
            statements.append(scene_statement)

    # Positional relations
    relations = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            relations.extend(
                get_positional_relations_horizontal(seg_image, objects[i], objects[j])
            )
            relations.extend(get_positional_relations_vertical(objects[i], objects[j]))
    if relations:
        np.random.shuffle(relations)
        for r in relations:
            if flip(p):
                statements.append(r.capitalize() + ".")

    # Clothing and action
    for h in humans:
        # Action
        if h.action:
            s_action = f"The {h.obj_name} {h.action}."
            statements.append(s_action)

        # Clothing
        if h.clothe:
            np.random.shuffle(h.clothe)
        for s in h.clothe:
            if flip(p):
                s_clothe = f"The {h.obj_name} {s.strip()}."
                statements.append(s_clothe)

    # Prompting statement
    statements.append(PROMPT_SUFFIX)
    return " ".join(statements).strip()
