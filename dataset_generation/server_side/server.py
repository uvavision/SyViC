import socket

import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.librarian import HumanoidLibrarian

from pathlib import Path
from tdw.asset_bundle_creator.animation_creator import AnimationCreator
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH

import shutil

from abc import ABC
from typing import Union
from pathlib import Path
from overrides import final
from tdw.tdw_utils import TDWUtils
from tdw.asset_bundle_creator.asset_bundle_creator import AssetBundleCreator

from tdw.asset_bundle_creator.humanoid_creator_base import HumanoidCreatorBase
import os
import time
import json

##################################################################################################################
##################################################################################################################

class ExtractAnimFromTeachFbx(AssetBundleCreator, ABC):

    @final
    def source_file_to_asset_bundles(self, name: str, source_file: Union[str, Path], output_directory: Union[str, Path]) -> None:

        args = HumanoidCreatorBase._get_source_destination_args(name=name, source=source_file, destination=output_directory)
        self.call_unity(method="ExtractAnimFromTeachFbx",
                        args=args,
                        log_path=AssetBundleCreator._get_log_path(output_directory))



class CustomAnimationCreator(HumanoidCreatorBase):
    """
    Create animation asset bundles from .anim or .fbx files.
    """

    def get_creator_class_name(self) -> str:
        return "ExtractAnimFromTeachFbx"


##################################################################################################################

TDW_socket = 1071
while True:
    HOSTNAME = "0.0.0.0"
    PORT = 10801

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOSTNAME, PORT))
    s.listen()

    output_name_tmp = ""

    while 1:
        print("Waiting for connections.")
        conn, addr = s.accept()
        with open(f"result_{addr[1]}.fbx", "ab+") as f:
            print("Connected to addr:", addr)
            while 1:
                data = conn.recv(1024)
                if not data:
                    break
                f.write(data)
            conn.close()
        print(f"Saved FBX from {addr} at 'result_{addr[1]}.fbx'")
        output_name_tmp = f'result_{addr[1]}.fbx'

        break

    s.shutdown(socket.SHUT_RDWR)
    s.close()

    # time.sleep(2)

    ### reopen to get additional data:
    PORT = 10800
    data = {}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOSTNAME, PORT))
    s.listen()
    while 1:
        print("Waiting for additional data...")
        conn, addr = s.accept()

        data = conn.recv(1024)
        data = json.loads(data.decode("utf-8"))
        conn.close()

        break

        # data = conn.recv(1024)
        # data = json.loads(data.decode())

    print ("Done...")
    clothe = data.get("clothe")
    scene = data.get("scene")
    length = data.get("length")
    print (clothe, scene, length)
    s.shutdown(socket.SHUT_RDWR)
    s.close()

    shutil.copyfile(f"teach_gen/{output_name_tmp}", "asset_bundle_creator/Assets/Resources/teach_fbx_101.fbx")

    r = CustomAnimationCreator()
    r.source_file_to_asset_bundles(name="teach_fbx_101_anim", 
                                source_file="source_file",
                                output_directory="output_directory")


    # remove existing anim bundle
    shutil.rmtree(EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("animations"))

    source_file = "asset_bundle_creator/Assets/Resources/teach_fbx_101.anim" #Path.home().joinpath("walking.fbx")
    output_directory = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("animations")
    print(f"Asset bundles will be saved to: {output_directory}")
    r = AnimationCreator()
    r.source_file_to_asset_bundles(name="teach_fbx_101", 
                                source_file=source_file,
                                output_directory=output_directory)


    ##################################################################################################################



    """
    Add and animate a SMPL humanoid.
    """
    TDW_socket += 1 
    os.system(f"DISPLAY=:1.0 tdw_build/TDW/TDW.x86_64 -port={TDW_socket} &")

    url = "file:///tdw_example_controller_output/animations/Linux/teach_fbx_101"
    name = "teach_fbx_101"

    # smpl_url = "file:///tdw_example_controller_output/clothedFBX_to_humanoid_smpl/Linux/cmu_37_37_01_male"
    # smpl_url = "file:///tdw_example_controller_output/clothed_from_prefab/male_prefab_try"
    # smpl_ulr = "file:///tdw_example_controller_output/clothed_prefab_to_TDW/Linux/cmu_male"
    # smpl_ulr = "file:///tdw_example_controller_output/try_clothed_nov15_1/Linux/cmu_37_37_01_male"
    # smpl_url = "file:///tdw_example_controller_output/try_clothed_nov15_2/Linux/cmu_37_37_01_male"
    # smpl_url = "file:///tdw_example_controller_output/try_clothed_nov15_3/Linux/cmu_37_37_01_male"
    # smpl_url = "file:///tdw_example_controller_output/try_clothed_nov15_4/Linux/cmu_37_37_01_male"
    # smpl_url = "file:///asset_bundles/Multi_Garmentdataset/Linux/125611497641994_registered_tex"
    # smpl_url = "file:///asset_bundles/Multi_Garmentdataset/Linux/125611494278283_registered_tex"
    # asset_bundles/Multi_Garmentdataset/Linux 125611494278283_registered_tex
    smpl_clothe = clothe.split("/")[-1].split(".jpg")[0]
    smpl_url = "file:///asset_bundles/Multi_Garmentdataset/Linux/" + smpl_clothe
    print ("clothe location:", smpl_url, "-", smpl_clothe)

    tdw_scene_name = "_".join(scene.replace("img/img_", "").split("_")[4:]).replace(".png", "")
    # Add a camera and enable image capture.
    humanoid_id = Controller.get_unique_id()
    camera = ThirdPersonCamera(avatar_id="a",
                            position={"x": -3, "y": 2.5, "z": 1.6},
                            look_at=humanoid_id)
    # path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("smpl_teach_fromTDWBundler_clothed11")
    path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("anim_from_fbx")
    print(f"Images will be saved to: {path}")
    capture = ImageCapture(avatar_ids=["a"], path=path)
    # Start the controller.
    c = Controller(port=TDW_socket)
    c.add_ons.extend([camera, capture])
    # Get the record for the SMPL humanoid and for the animation.
    c.humanoid_librarian = HumanoidLibrarian("smpl_humanoids.json")
    humanoid_record = c.humanoid_librarian.get_record("humanoid_smpl_f")
    animation_command, animation_record = c.get_add_humanoid_animation("walking_1")
    commands = [{"$type": "set_screen_size",
                "width": 1920,
                "height": 1080},
                c.get_add_scene(scene_name=tdw_scene_name),
                # {"$type": "create_exterior_walls", "walls": TDWUtils.get_box(1920, 1080)},
                {"$type": "set_render_quality",
                "render_quality": 5}, #TDWUtils.create_empty_room(120, 120),

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
                    "framerate": 30}]

                # {"$type": "play_humanoid_animation",
                #  "name": animation_record.name,
                #  "id": humanoid_id},
                # {"$type": "set_target_framerate",
                #  "framerate": animation_record.framerate}]
    c.communicate(commands)
    frames = 60*20 #animation_record.get_num_frames()
    for i in range(frames):
        c.communicate({"$type": "look_at",
                    "object_id": humanoid_id})
    c.communicate({"$type": "terminate"})


    os.system("rm out.mp4")
    os.system("ffmpeg -framerate 30 -pattern_type glob -i 'tdw_example_controller_output/anim_from_fbx/a/*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4")
    os.system("pkill -9 TDW")

    ########################################################
    HOSTNAME = ""
    PORT = 10802

    # Send FBX data to server
    path_to_fbx = "out.mp4"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOSTNAME, PORT))

    with open(path_to_fbx, "rb") as f:
        data = f.read()
    s.sendall(data)
    s.shutdown(socket.SHUT_RDWR)
    s.close()
    print("Sent MP4 to client.")


    # time.sleep(5)



