import socket

import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.librarian import HumanoidLibrarian, ModelLibrarian

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

import numpy as np
from tdw.output_data import OutputData, Transforms, Rigidbodies, SegmentationColors, Categories, IdPassSegmentationColors
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.trigger_collision_manager import TriggerCollisionManager
from tdw.output_data import Occlusion as Occl
import csv
import time
import copy
import pickle
import math
import sys

from glob import glob
from scipy.spatial import ConvexHull
from utils import list_records_by_size

from tdw.librarian import SceneLibrarian

from pathlib import Path
EXAMPLE_CONTROLLER_OUTPUT_PATH = Path.home().joinpath("tdw_example_controller_output_v2")

"""
Add and animate a SMPL humanoid.
"""
class TDW_Humanoid_Scene_Generator():
    def __init__(self, tdw_scene_name = "tdw_room", folder_name = "test", TDW_socket=1071, folder_id=1):
        
        c = Controller(port=TDW_socket)

        # Where it will be saved
        path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath(f"anim_from_fbx_{folder_id}")
        print(f"Images will be saved to: {path}")

        look_at_position = {"x": 1, "y": 0, "z": -1.5}
        cameras_ids = [f"c{ii}" for ii in range (4)] 
        camera_positions = {}
        all_cameras = []
        for avatar_id in cameras_ids:
            # avatar_id = folder_name
            initial_position = {"x": random.uniform(-4, 4), "y": random.uniform(0.5, 3), "z": random.uniform(-4, 4)}
            camera_positions[avatar_id] = initial_position
            camera = ThirdPersonCamera(avatar_id=avatar_id,
                                        position=initial_position,
                                        look_at=look_at_position)
            all_cameras.append(camera)
        
        # capture = ImageCapture(avatar_ids=[avatar_id], path=path, pass_masks=["_img", "_id", "_mask", "_normals", "_albedo", "_category", "_depth", "_flow"]) # pass_masks=["_img", "_depth"]) #, pass_masks=["_img", "_id", "_category"]) # 
        capture = ImageCapture(avatar_ids=cameras_ids, path=path, pass_masks=["_img", "_id", "_category", "_depth"])

        # setup controller
        c.add_ons.extend(all_cameras)
        c.add_ons.append(capture)

        self.path = path
        self.tdw_scene_name = tdw_scene_name
        self.TDW_socket = TDW_socket
        
        self. camera_positions = camera_positions
        self.all_cameras = all_cameras

        self.look_at_position = look_at_position
        self.controller = c
        self.image_info_path = self.path.joinpath("info.p")

        self.humanoids_recods = {}
        self.objects_recods = {}
        self.per_frame_info = {}

    def add_humanoid(self, smpl_clothe="125611520103063_registered_tex", animation_name="teach_fbx_101", position = {"x": 0, "y": 1, "z": 0}, frame_rate=120, rotation=90, scale=1):
        humanoid_id = Controller.get_unique_id()

        # url = f"file:///tdw_example_controller_output/animations/Linux/{animation_name}"
        # smpl_url = random.choice(["file:///asset_bundles/NO_CLOTH/Linux/cmu_37_37_01_female", "file:///asset_bundles/NO_CLOTH/Linux/cmu_37_37_01_male"])
        # smpl_url = f"file:///asset_bundles/SURREAL/Linux/{smpl_clothe}" 
        smpl_url = f"file:///asset_bundles/Multi_Garmentdataset_legacyTexture/Linux/{smpl_clothe}" 
        url = f"file:///{animation_name}"

        # Add the SMPL humanoid and the animation.
        commands = [{"$type": "add_humanoid", 
                    "id": humanoid_id,
                    "name": animation_name,
                    "url": smpl_url, 
                    "position": position,
                    "rotation": {"x": 0, "y": 0, "z": rotation},
                    "scale_factor": scale,
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

                    # add gravity
                    {"$type": "set_kinematic_state",
                                "id": humanoid_id, 
                                "is_kinematic": False, 
                                "use_gravity": True},

                    {"$type": "set_mass",
                                "mass": 0.65, 
                                "id": humanoid_id},
                    {"$type": "set_physic_material",
                        "dynamic_friction": 0.45, 
                        "static_friction": 0.48, 
                        "bounciness": 0.5, 
                        "id": humanoid_id},

                    # animation_command,
                    {"$type": "add_humanoid_animation",
                        "name": animation_name,
                        "url": url},
                    {"$type": "set_target_framerate",
                        "framerate": frame_rate},
                    {"$type": "play_humanoid_animation",
                        "name": animation_name,
                        "id": humanoid_id,
                        "framerate": frame_rate}]

        self.humanoids_recods[humanoid_id] = {"smpl_clothe": smpl_clothe, "animation_fbx_name": animation_name, "position": position, "rotation": {"x": 0, "y": 0, "z": rotation}}

        return humanoid_id, commands

    def add_object(self, obj_name = "iron_box", position = {"x": -0.1, "y": 0, "z": 0.1}):
        dropped_object_id = Controller.get_unique_id()
        model_record = ModelLibrarian(library='models_full.json').get_record(obj_name)
        scale_factor = 1
        rotation = {"x": int(random.choice(np.arange(10))), "y": int(random.choice(np.arange(20))), "z": 0}
        commands = [{"$type": "add_object", 
                            "name": model_record.name, 
                            "url": model_record.get_url(), 
                            "scale_factor": model_record.scale_factor,  
                            "position": position,
                            "category": model_record.wcategory, 
                            "id": dropped_object_id,
                            "rotation": rotation}, 
                        #  {"$type": "rotate_object_to_euler_angles", 
                        #         "euler_angles": {"x": int(random.choice(np.arange(10))), "y": int(random.choice(np.arange(20))), "z": 0},
                        #         "id": dropped_object_id}, 
                         {"$type": "set_kinematic_state",
                                "id": dropped_object_id, 
                                "is_kinematic": False, 
                                "use_gravity": True},
                         {"$type": "set_mass",
                                "mass": 0.65, 
                                "id": dropped_object_id},
                         {"$type": "set_physic_material",
                                "dynamic_friction": 0.45, 
                                "static_friction": 0.48, 
                                "bounciness": 0.5, 
                                "id": dropped_object_id},
                         {"$type": "scale_object", "id": dropped_object_id, "scale_factor": {"x": scale_factor, "y": scale_factor, "z": scale_factor}},
                            
            ]

        self.objects_recods[dropped_object_id] = {"obj_name": obj_name, "position": str(position), "rotation": str(rotation)}

        return dropped_object_id, commands

    def add_objs_trackers(self):
        #  add/record rigidbody and transforms status
        commands = [{"$type": "send_rigidbodies",
                            "frequency": "always"},
                    {"$type": "send_transforms",
                            "frequency": "always"},
                    {"$type": "send_humanoids", 
                            "frequency": "always"},
                    {"$type": "send_segmentation_colors",
                            "frequency": "always"},
                    {"$type": "send_id_pass_segmentation_colors",
                            "frequency": "always"},
                    {"$type": "send_categories",
                            "frequency": "once"},]

        return commands

    def log_information(self, frame_id, frame_number):
        self.per_frame_info[frame_id] = {"frame_number": frame_number, "tdw_scene_name": self.tdw_scene_name, 
                                         "objects": copy.deepcopy(self.objects_recods), "humans": copy.deepcopy(self.humanoids_recods), 
                                         "camera_pos": copy.deepcopy(self.camera_positions), "camera_look_at": str(self.look_at_position)}
                                         # "camera_pos": str(self.initial_position), "camera_look_at": str(self.look_at_position)}


    def animate(self, commands, all_humanoids_ids, all_objects_ids, dict_amass_info):
        # breakpoint()

        execute_empty_commands = True
        
        c = self.controller

        frames = 0
        for k_smpl, v_smpl in dict_amass_info.items():
            tmp_frames = len(v_smpl["per_frame_ann"])
            if tmp_frames > frames:
                frames = tmp_frames

        action_idx = 0
        frame_animation_index = 0

        resp = c.communicate(commands)
        print ("humanoids:", all_humanoids_ids, "| objects:", all_objects_ids)
        ## always look at the humanoid
        for camera_tmp in self.all_cameras:
            camera_tmp.look_at(all_humanoids_ids[0])

        self.log_segmentation_info(resp)
        self.log_information(0, frame_animation_index)
        frame_animation_index += 1  # everytime there is a communicate, add 1 after logging the info

        for current_frame in range(frames):

            if current_frame > 1:
                # self.log_segmentation_info(resp)
                pickle.dump( self.per_frame_info, open( self.image_info_path, "wb" ) )
                with open(str(self.image_info_path).replace('.p', '.json'), 'w', encoding='utf-8') as f:
                    json.dump(self.per_frame_info, f, ensure_ascii=False, indent=4) 

            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])

                if r_id == "tran":
                    transforms = Transforms(resp[i])
                    for j in range(transforms.get_num()):
                        if transforms.get_id(j) in all_humanoids_ids:
                            humanoid_id = transforms.get_id(j)
                            humanoid_position = transforms.get_position(j)

                            try:
                                self.humanoids_recods[humanoid_id]["position"] = str(humanoid_position)
                                self.humanoids_recods[humanoid_id]["rotation"] = str(transforms.get_rotation(j))
                                self.humanoids_recods[humanoid_id]["forward"] = str(transforms.get_forward(j))
                                self.humanoids_recods[humanoid_id]["action_description"] = dict_amass_info[humanoid_id]["per_frame_ann"][current_frame]
                                self.humanoids_recods[humanoid_id]["action_categories"] = dict_amass_info[humanoid_id]["per_frame_act_cats"][current_frame]
                                self.humanoids_recods[humanoid_id]["raw_caption"] = dict_amass_info[humanoid_id]["raw_caption"]
                            except:
                                self.humanoids_recods[humanoid_id]["action_description"] = dict_amass_info[humanoid_id]["per_frame_ann"][-1]
                                self.humanoids_recods[humanoid_id]["action_categories"] = dict_amass_info[humanoid_id]["per_frame_act_cats"][-1]
                                self.humanoids_recods[humanoid_id]["raw_caption"] = dict_amass_info[humanoid_id]["raw_caption"]

                        elif transforms.get_id(j) in all_objects_ids:
                            dropped_object_id = transforms.get_id(j)
                            object_position = transforms.get_position(j)

                            self.objects_recods[dropped_object_id]["position"] = str(object_position)
                            self.objects_recods[dropped_object_id]["rotation"] = str(transforms.get_rotation(j))
                            self.objects_recods[dropped_object_id]["forward"] = str(transforms.get_forward(j))

                elif r_id == "rigi":
                    rigidbodies = Rigidbodies(resp[i])
                    for j in range(rigidbodies.get_num()):
                        if rigidbodies.get_id(j) in all_objects_ids :
                            dropped_object_id = rigidbodies.get_id(j)
                            # sleeping == False means the object moved
                            sleeping = rigidbodies.get_sleeping(j)

            ## always look at the humanoid
            for camera_tmp in self.all_cameras:
                camera_tmp.look_at(all_humanoids_ids[0])


            if execute_empty_commands:
                resp = c.communicate([])
                self.log_information(current_frame+1, frame_animation_index)
                frame_animation_index += 1
            else:
                execute_empty_commands = True

        # end gratiously 
        c.communicate({"$type": "terminate"})

    def log_segmentation_info(self, resp):

        # Get each segmentation color.
        segmentation_colors_per_object = dict()
        segmentation_colors_in_image = list()
        category_colors = dict()
        visible_object_ids = []

        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            # Get segmentation color output data.
            if r_id == "segm":
                segm = SegmentationColors(resp[i])
                for j in range(segm.get_num()):
                    try:
                        object_id = segm.get_object_id(j)
                        object_category = segm.get_object_category(j)
                        segmentation_color = segm.get_object_color(j)
                        segmentation_colors_per_object[object_id] = [segmentation_color, object_category]
                        self.objects_recods[object_id]['id'] = str(segmentation_color)
                        self.objects_recods[object_id]['category_name'] = str(object_category)
                        visible_object_ids.append(object_id)
                    except:
                        print ("object_id does not exist:", object_id)
            elif r_id == "ipsc":
                ipsc = IdPassSegmentationColors(resp[i])
                for j in range(ipsc.get_num_segmentation_colors()):
                    color_id = ipsc.get_segmentation_color(j).tolist()
                    if color_id not in segmentation_colors_in_image:
                        segmentation_colors_in_image.append(color_id)
            elif r_id == "cate":
                cate = Categories(resp[i])
                for j in range(cate.get_num_categories()):
                    category_name = cate.get_category_name(j)
                    category_color = cate.get_category_color(j)
                    category_colors[category_name] = category_color

        humanoid_cat_color = []
        for k,v in category_colors.items():
            if k != 'humanoid':
                for k2,v2 in self.objects_recods.items():
                    if v2['category_name'] == k:
                        self.objects_recods[k2]['category_id'] = str(v)
            else:
                humanoid_cat_color = str(v)
        
        for k in self.humanoids_recods.keys():
            self.humanoids_recods[k]['category_id'] = str(humanoid_cat_color)

        self.segmentation_colors_per_object = segmentation_colors_per_object
        self.segmentation_colors_in_image = segmentation_colors_in_image
        self.category_colors = category_colors

def get_good_records(lib):
    records = []
    records_dims = []
    for record in lib.records:
        if record.do_not_use:
            continue
        points = []
        for _, v in record.bounds.items():
            points.append([v['x'], v['y'], v['z']])
        records_dims.append(ConvexHull(points).volume)
        records.append({"name": record.name,
                        "wnid": record.wnid,
                        "wcategory": record.wcategory,
                        "substructure": record.substructure,
                        "bounds": record.bounds,
                        "dimension": ConvexHull(points).volume})
    records = {"records": records, "records_dims": records_dims}
    return records

if __name__ == '__main__':

    user_manual_port = int(sys.argv[1])
    user_manual_anim_idx = int(sys.argv[2])

    # randomly select smpl_clothe
    all_files_Multi_Garmentdataset = glob("asset_bundles/Multi_Garmentdataset_legacyTexture/Linux/*")
    all_files_Multi_Garmentdataset = [ff.split('/')[-1] for ff in all_files_Multi_Garmentdataset if '.' not in ff]  

    surreal_females_clothes = glob("asset_bundles/SURREAL/Linux/*_female_*")
    surreal_females_clothes = [ff.split('/')[-1] for ff in surreal_females_clothes if '.' not in ff] 
    surreal_males_clothes = glob("asset_bundles/SURREAL/Linux/*_male_*")
    surreal_males_clothes = [ff.split('/')[-1] for ff in surreal_males_clothes if '.' not in ff] 

    # randomly select small object
    lib = ModelLibrarian(library='models_full.json')
    usable_records = get_good_records(lib)
    # update info after looking at dimensions of objects
    usable_records = list_records_by_size(usable_records)

    # load valid subset from babel-teach train set -- filtered for raw actions coverage
    cmu_good_coverage_samples = pickle.load( open( "cmu_good_coverage_samples.p", "rb" ) )
    cmu_valid_full_json = cmu_good_coverage_samples['cmu_valid_full_json']
    cmu_valid_asset_paths = cmu_good_coverage_samples['cmu_valid_asset_paths']
    # precompute per-frame unique actions/annotations
    per_frame_annotations = []
    for k in range (len(cmu_valid_asset_paths)):
        for k_babel,v_babel in cmu_valid_full_json[k].items():
            # babel_current_motions[k][0]
            raw_caption = str (v_babel['seq_ann']['labels'][0]['raw_label'])

            per_frame_ann = [" "*60] * math.floor(v_babel['dur'] * 120)
            per_frame_ann = np.array(per_frame_ann)
            per_frame_act_cats = [" "*60] * math.ceil(v_babel['dur'] * 120)
            for amass_labels in v_babel['frame_ann']['labels']:
                for ii in range (math.floor(amass_labels['start_t']) * 120, math.floor(amass_labels['end_t']) * 120):
                    per_frame_ann[ii] = amass_labels['proc_label']
                    per_frame_act_cats[ii] = ",".join(amass_labels['act_cat'])

        try:
            false_frame_idx = list(per_frame_ann).index('                                                            ')
        except:
            false_frame_idx = len(per_frame_ann) - 1
        per_frame_annotations.append({"raw_caption": raw_caption, "per_frame_ann": list(per_frame_ann)[:false_frame_idx], "per_frame_act_cats": list(per_frame_act_cats)[:false_frame_idx]})
    # humanoid rotations:
    humanoid_rotations = [-90] * len(cmu_valid_asset_paths) # []
    # humanoid_rotations[1] = 90
    
    mocap_names = []
    mocap_names_v2 = []
    for anim_path_idx in range(len(cmu_valid_asset_paths)):
        valid_idx = cmu_valid_asset_paths[anim_path_idx].replace('.npz', '').replace("cmu", "CMU")
        filepath_amass = glob("asset_bundle_creator/Assets/Resources/" + valid_idx + "*")
        babel_anim_path = filepath_amass[0].replace(".meta", "")
        mocap_names.append(babel_anim_path.replace("asset_bundle_creator/Assets/Resources/", "").replace(".anim", ""))
        mocap_names_v2.append(babel_anim_path.replace("asset_bundle_creator/Assets/Resources/", "").replace(".anim", "").replace("_female", "").replace("_male", ""))
    ######
    with open('check_orientations_manual.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            tmp_mocap_name = row[0].replace("cmu", "CMU")
            humanoid_rotations[mocap_names_v2.index(tmp_mocap_name)] += int(row[1])

    lib = SceneLibrarian()
    scene_names = [rec.name for rec in lib.records]
    tdw_usable_scene_names = scene_names[:6] + scene_names[-4:]
    tdw_usable_scene_names.remove('monkey_physics_room')
    tdw_usable_scene_names.remove('archviz_house')
    tdw_usable_scene_names.remove('downtown_alleys')

    total_ii = 0
    for smpl_ii in range(user_manual_anim_idx+1, user_manual_anim_idx+4):

        valid_idx = cmu_valid_asset_paths[smpl_ii]
        valid_idx = valid_idx.replace('.npz', '')
        filepath_amass = glob("asset_bundles/Linux/" + valid_idx + "*")
        filepath_amass = filepath_amass[0].replace(".manifest", "")
        
        random.shuffle(tdw_usable_scene_names)
        for iii_scene_name in tdw_usable_scene_names[:1]:

            TDW_socket=user_manual_port+total_ii
            os.system(f"DISPLAY=:1.0 tdw_build/TDW/TDW.x86_64 -port={TDW_socket} &")

            generator = TDW_Humanoid_Scene_Generator(tdw_scene_name = iii_scene_name, TDW_socket=TDW_socket, folder_name = "", folder_id=f"{TDW_socket}_{iii_scene_name}_{valid_idx}")

            # two humans:
            animation_commands = [{"$type": "set_screen_size",
                                    "width": 1080,
                                    "height": 1080},
                                    generator.controller.get_add_scene(scene_name=generator.tdw_scene_name),
                                    {"$type": "set_render_quality",
                                    "render_quality": 5},]
            all_humanoids_ids = []
            all_objects_ids = []
            all_humanoid_annotations = {}

            # add first humanoid
            h_position = {"x": random.uniform(-0.5, 0.5), "y": 1, "z": random.uniform(-0.5, 0.5)}
            # assign clothes
            smpl_clothe=random.choice(all_files_Multi_Garmentdataset)
            humanoid_id, commands1 = generator.add_humanoid(animation_name=filepath_amass, smpl_clothe=smpl_clothe, position=h_position, rotation=humanoid_rotations[smpl_ii])  # -90) # rotation=90) # rotation=180)
            all_humanoids_ids.append(humanoid_id)
            animation_commands.extend(commands1)
            all_humanoid_annotations[humanoid_id] = per_frame_annotations[smpl_ii]
            print ("first humanoid:", humanoid_id)

            # additional humanoid
            number_of_adds_humanoids = random.choice(np.arange(1, 3))
            for _ in range (number_of_adds_humanoids):
                additional_smpl = random.choice(mocap_names)
                additional_smpl_idx = mocap_names.index(additional_smpl)
                additional_smpl = additional_smpl.lower()
                random_filepath_amass = "tdw_example_controller_output/animations/Linux/" + additional_smpl
                print ("random_filepath_amass:", random_filepath_amass)
                h_position = {"x": random.uniform(-1.9, 1.9), "y": 1, "z": random.uniform(-1.9, 1.9)}
                # assign clothes
                smpl_clothe=random.choice(all_files_Multi_Garmentdataset)
                smpl_clothe=random.choice(all_files_Multi_Garmentdataset)
                humanoid_id, commands2 = generator.add_humanoid(animation_name=random_filepath_amass, smpl_clothe=smpl_clothe, position=h_position, rotation=humanoid_rotations[additional_smpl_idx], scale=0.5)  # -90) # rotation=90) # rotation=180)
                all_humanoids_ids.append(humanoid_id)
                animation_commands.extend(commands2)
                all_humanoid_annotations[humanoid_id] = per_frame_annotations[additional_smpl_idx]
                print ("second humanoid:", humanoid_id)
                
            number_of_objects = random.choice(np.arange(0, 8))
            print ("number_of_objects", number_of_objects)
            for o_number in range(number_of_objects):
                o_position = {"x": random.uniform(-2.5, 2.5), "y": 0, "z": random.uniform(-2.5, 2.5)}
                obj_name=random.choice(usable_records['objects_small'])
                dropped_object_id, commands = generator.add_object(obj_name = obj_name, position=o_position)
                all_objects_ids.append(dropped_object_id)
                animation_commands.extend(commands)

            commands_trackers = generator.add_objs_trackers()   
            animation_commands.extend(commands_trackers)     
        
            generator.animate(animation_commands, all_humanoids_ids, all_objects_ids, all_humanoid_annotations)

            pickle.dump( generator.per_frame_info, open( generator.image_info_path, "wb" ) )
            with open(str(generator.image_info_path).replace('.p', '.json'), 'w', encoding='utf-8') as f:
                json.dump(generator.per_frame_info, f, ensure_ascii=False, indent=4) 
            
            total_ii += 1