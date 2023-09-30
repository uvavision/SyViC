import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.librarian import HumanoidLibrarian

"""
Add and animate a SMPL humanoid.
"""

url = "file:///tdw_example_controller_output/animations_TDW/Linux/test_teach"
smpl_url = "file:///asset_bundles/Multi_Garmentdataset/Linux/125611497641994_registered_tex"
name = "teach_walk"
# Add a camera and enable image capture.
humanoid_id = Controller.get_unique_id()
camera = ThirdPersonCamera(avatar_id="a",
                           position={"x": -3, "y": 2.5, "z": 1.6},
                           look_at=humanoid_id)
path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("smpl_teach_fromTDWBundler_clothed11")
print(f"Images will be saved to: {path}")
capture = ImageCapture(avatar_ids=["a"], path=path)
# Start the controller.
c = Controller()
c.add_ons.extend([camera, capture])
# Get the record for the SMPL humanoid and for the animation.
c.humanoid_librarian = HumanoidLibrarian("smpl_humanoids.json")
humanoid_record = c.humanoid_librarian.get_record("humanoid_smpl_f")
animation_command, animation_record = c.get_add_humanoid_animation("walking_1")
commands = [TDWUtils.create_empty_room(120, 120),
            {"$type": "add_humanoid", # "add_smpl_humanoid",
             "id": humanoid_id,
             "name": humanoid_record.name,
             "url": smpl_url, # humanoid_record.get_url(),
             "position": {"x": 0, "y": 1, "z": 0},
             "rotation": {"x": 0, "y": 0, "z": -90},
             "height": random.uniform(-1, 1),
             "weight": random.uniform(-1, 1),
             "torso_height_and_shoulder_width": random.uniform(-1, 1),
             "chest_breadth_and_neck_height": random.uniform(-1, 1),
             "upper_lower_back_ratio": random.uniform(-1, 1),
             "pelvis_width": random.uniform(-1, 1),
             "hips_curve": random.uniform(-1, 1),
             "torso_height": random.uniform(-1, 1),
             "left_right_symmetry": random.uniform(-1, 1),
             "shoulder_and_torso_width": random.uniform(-1, 1)},
            # animation_command,
            {"$type": "add_humanoid_animation",
                "name": name,
                #"url": record.get_url()},
                "url": url},
                {"$type": "set_target_framerate",
                "framerate": 30},
                {"$type": "play_humanoid_animation",
                "name": name,
                "id": humanoid_id,
                "framerate": 10}]

            # {"$type": "play_humanoid_animation",
            #  "name": animation_record.name,
            #  "id": humanoid_id},
            # {"$type": "set_target_framerate",
            #  "framerate": animation_record.framerate}]
c.communicate(commands)
frames = 30 #animation_record.get_num_frames()
for i in range(frames):
    c.communicate({"$type": "look_at",
                   "object_id": humanoid_id})
c.communicate({"$type": "terminate"})
