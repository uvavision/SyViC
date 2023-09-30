from pathlib import Path
from math import radians, sin, cos
from tdw.controller import Controller
from tdw.output_data import Images
from tdw.tdw_utils import TDWUtils
from random import uniform, randint, choice
from itertools import product
import os
import time
import pdb
import csv
import json
import argparse
import numpy as np
from numpy import linalg as LA
import glob

from tdw.librarian import ModelLibrarian
from tdw.librarian import MaterialLibrarian
# to get record volume
from scipy.spatial import ConvexHull
from utils import list_records_by_size #, get_objects_by_category_with_extra_information

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Collision, EnvironmentCollision, Rigidbodies
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager


class TDW_Scene_Generator():
    def __init__(self, scene_name, port=1071, output_dir="photos"):
        c = Controller(port=port)

        # get library
        self.lib, self.lib_materials = self.get_full_lib()
        self.usable_records = self.get_good_records(self.lib)
        # update info after looking at dimensions of objects
        self.usable_records = list_records_by_size(self.usable_records)

        self.scene_name = scene_name
        self.scene_controller = c
        self.output_dir = output_dir

        # to not load again the materials already loaded 
        self.materials_already_rendered = {}

        # get all materials
        self.all_materials = [rec.name for rec in self.lib_materials.records]
        
    
    def set_camera(self):
        c = self.scene_controller

        x_rot, z_rot, y = self.rotation_displacement(self.camera_theta) # , x = 4.5
        camera = ThirdPersonCamera(position={"x": self.camera_pos_x, "y": self.camera_pos_y, "z": self.camera_pos_z},
                                   look_at={"x": x_rot + self.camera_pos_x,
                                            "y": y + self.camera_pos_y,
                                            "z": z_rot + self.camera_pos_z},
                                   field_of_view=68)

        # capture = ImageCapture(avatar_ids=["a"], path=output_dir, pass_masks=["_img", "_id"])
        capture = ImageCapture(avatar_ids=[camera.avatar_id], path=self.output_dir, pass_masks=["_img"])
        object_manager = ObjectManager(transforms=True, rigidbodies=True)
        c.add_ons.extend([camera, object_manager, capture])

        self.camera = camera
        self.object_manager = object_manager

    def set_scene(self):
        # just use the global controller
        c = self.scene_controller

        commands = [{"$type": "set_screen_size",
                    "width": 1920,
                    "height": 1080},
                    {"$type": "set_render_quality",
                    "render_quality": 5},
                    c.get_add_scene(scene_name=self.scene_name)]

        return commands

    def excecute_video_and_destroy(self, commands, objs_to_destroy, wait=2):
        c = self.scene_controller

        c.communicate(commands)
        xx = self.camera_pos_x
        yy = self.camera_pos_y
        zz = self.camera_pos_z
        tilt = 0.1
        mov = 0.0
        round_ = -1
        for i in range(wait):

            self.camera.teleport(position={"x": xx + mov, "y": yy + tilt, "z": zz})
            self.camera.look_at(choice(objs_to_destroy))
            tilt += 0.1
            if tilt >= 3.1:
                tilt = 0.0
                round_ = 0 
            if round_ == 0:
                mov += -0.1
                if mov <= 1:
                    round_ = 1
                    mov = 0.0
            if round_ == 1:
                mov += 0.1
                if mov >= 1:
                    round_ = 0
                    mov = 0.0

            c.communicate([{"$type": "rotate_object_by",
                            "id": objs_to_destroy[0],
                            "axis": "yaw",
                            "is_world": False,
                            "angle": 10}])

        # breakpoint()
        # Reset camera position and Destroy the object.
        xx = self.camera_pos_x
        yy = self.camera_pos_y
        zz = self.camera_pos_z
        self.camera.teleport(position={"x": xx, "y": yy, "z": zz})
        self.stop_look_at()
        c.communicate([])

        for object_id in objs_to_destroy:
            c.communicate({"$type": "destroy_object",
                           "id": object_id})
        # Mark the object manager as requiring re-initialization.
        self.object_manager.initialized = False

    def excecute_commands(self, commands):
        c = self.scene_controller
        c.communicate(commands)

        c.communicate({"$type": "set_shadow_strength",
                       "strength": 0.80})
    
    def end_simulation(self):
        c = self.scene_controller

        c.communicate({"$type": "terminate"})

    def get_full_lib(self):
        return ModelLibrarian(library='models_full.json'), MaterialLibrarian()

    def get_good_records(self, lib):
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

    def set_camera_position(self, cx, cy, cz, theta):
        self.camera_pos_x = cx
        self.camera_pos_y = cy
        self.camera_pos_z = cz
        self.camera_theta = theta

    def rotation_displacement(self, theta, x = 3, y = 0, z = 0, c_x = 0, c_z = 0):
        rad = radians(theta)
        x_rot = cos(rad) * (x - c_x) - sin(rad) * (z - c_z) + c_x
        z_rot = sin(rad) * (x - c_x) + cos(rad) * (z - c_z) + c_z
        return x_rot, z_rot, y

    def place_object_inside_cameraview(self, commands = [], new_object="glass_table_round", displacement=1, material_name='', add_to_y=0, material_color="", scale_factor = 1):
        c = self.scene_controller
        x_rot, z_rot, y = self.rotation_displacement(self.camera_theta + displacement)

        model_record = ModelLibrarian(library='models_full.json').get_record(new_object)
        dropped_object_id = c.get_unique_id()
        material_record_id = -1

        if material_color:
            if material_color not in self.materials_already_rendered.keys():
                url_mat = f"file:///unity_material_assets/StandaloneLinux64_MaterialsGlossColors/{material_color}"
                material_record_id = str(c.get_unique_id())

                commands.extend([{"$type": "add_material",
                                "name": material_record_id,
                                "url": url_mat}])

                self.materials_already_rendered[material_color] = material_record_id
            else:
                # material is already loaded, just use it
                material_record_id = self.materials_already_rendered[material_color]

        commands.extend([{"$type": "add_object", 
                "name": model_record.name, # "iron_box", 
                "url": model_record.get_url(), # "https://tdw-public.s3.amazonaws.com/models/linux/2018-2019.1/iron_box", 
                "scale_factor": model_record.scale_factor, # 1.0, 
                "position": {"x": x_rot + self.camera_pos_x,
                            "y": y + add_to_y,  # I want it to be in the floor already!
                            "z": z_rot + self.camera_pos_z},
                "category": model_record.wcategory, # "box", 
                "id": dropped_object_id}, 
               {"$type": "rotate_object_to_euler_angles", 
                "euler_angles": {"x": int(choice(np.arange(10))), "y": int(choice(np.arange(20))), "z": 0},
                "id": dropped_object_id}, 
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
                {"$type": "scale_object", "id": dropped_object_id, "scale_factor": {"x": scale_factor, "y": scale_factor, "z": scale_factor}},])

        
        if material_color:
            for sub_struct_tmp in model_record.substructure:
                model_material_name =  sub_struct_tmp["name"]
                number_materials = len(sub_struct_tmp["materials"])
                for idx in range(number_materials):
                    commands.extend([{"$type": "set_visual_material",
                                    "material_index": idx,
                                    "material_name": material_record_id,
                                    "object_name": model_material_name, 
                                    "id": dropped_object_id}])
        

        # # a list of materials are available at scene_generator.lib_materials.data['records']
        if material_name:
            commands.extend([c.get_add_material(material_name=material_name)])
            for sub_struct_tmp in model_record.substructure:
                model_material_name =  sub_struct_tmp["name"]
                number_materials = len(sub_struct_tmp["materials"])
                for idx in range(number_materials):
                    commands.extend([{"$type": "set_visual_material",
                                    "material_index": idx,
                                    "material_name": material_name,
                                    "object_name": model_material_name, 
                                    "id": dropped_object_id}])

        return dropped_object_id, material_record_id, commands

    def place_human_inside_cameraview(self, commands = [], displacement=1, material_name='', add_to_y=0, material_color=""):
        c = self.scene_controller
        x_rot, z_rot, y = self.rotation_displacement(self.camera_theta + displacement)

        url = "file:///unity_material_assets/StandaloneLinux64_SMPLX_Animated/cmu_02_02_01_male_125611494277906_registered_tex_smplx"
        
        c = self.scene_controller
        x_rot, z_rot, y = self.rotation_displacement(self.camera_theta + 1)
        dropped_object_id = c.get_unique_id()
        commands.extend([{"$type": "add_object",
                        "name": "cmu_02_02_01_male_125611494277906_registered_tex",
                        "url": url,
                        "position": {"x": x_rot + self.camera_pos_x,
                                    "y": y + 0,  # I want it to be in the floor already!
                                    "z": z_rot + self.camera_pos_z},
                        "rotation": {"x": -90, "y": 0, "z": 0},
                        "scale_factor": 1,
                        "id": dropped_object_id},
                        {"$type": "set_kinematic_state",
                        "id": dropped_object_id, 
                        "is_kinematic": True, 
                        "use_gravity": False},])

        print ("==>", dropped_object_id)
        return dropped_object_id, commands        

    def do_the_panorama_thing(self, tilt=0, object_id=None):
        # xx = self.camera_pos_x
        # yy = self.camera_pos_y
        # zz = self.camera_pos_z
        # tt = self.camera_theta

        ## Teleport the avatar, i.e. camera.
        # self.camera.teleport(position={"x": xx, "y": yy + tilt, "z": zz})
        self.camera.look_at(object_id)

    def stop_look_at(self):
        self.camera.look_at(None)


parser = argparse.ArgumentParser(description='YOUR PROJECT NAME HERE')
parser.add_argument('--cuda', dest = 'cuda', default=0, type=int, help='cuda device number')
parser.add_argument('--port', dest = 'port', default=1071, type=int, help='port number')
parser.add_argument('--scene_name', type = str, default = 'floorplan_1a', help='scene name folder')
parser.add_argument('--chunk', dest = 'chunk', default=0, type=int, help='number of data chunk from placeholders in folder')
parser.add_argument('--scale_stuff', action='store_true', help='Scale Objects')

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # breakpoint()

    # # get hand-picked camera positions from scene
    camera_info = glob.glob(f"placeholders/placeholders/{args.scene_name}/*.png")
    all_material_colors = glob.glob("unity_material_assets/StandaloneLinux64_MaterialsGlossColors/*")

    final_dir_path = f"photos_{args.cuda}_{args.port}/{args.scene_name}"
    if os.path.isdir(final_dir_path):
        print ("Dir exists :: Images will be saved at " + final_dir_path)
    else:
        os.system(f"mkdir {final_dir_path}")
        print ("Dir created :: Images will be saved at " + final_dir_path)

    image_info_path = f"photos_{args.cuda}_{args.port}/{args.scene_name}/image_info.csv"

    # I'm very tired, just run this forever... ## NOT!
    for camera_name in camera_info[:10]:
        # call TDW display to handle rendering
        os.system(f"DISPLAY=:1.0 tdw_build/TDW/TDW.x86_64 -port={args.port} &")
        # CREATE CONTROLLER -- new instance of class TDW_Scene_Generator with the scene name and port to be used
        scene_generator = TDW_Scene_Generator(args.scene_name, port=args.port, output_dir=final_dir_path)

        # for camera_name in camera_info[2][(total_files*args.chunk) : (total_files*(args.chunk+1))]:
        tmp_out = camera_name.split("/")[-1].replace(".png", "").split("_")
        cx = float(tmp_out[1])
        cy = float(tmp_out[2])
        cz = float(tmp_out[3])
        theta = float(tmp_out[4])
        scene_name = '_'.join(tmp_out[5:])
        print (cx, '_', cy, '_', cz, '_', theta, '_', scene_name)

        # pass camera x, y, z and theta to scene generator
        scene_generator.set_camera_position(cx, cy, cz, theta)
        # set the camera
        scene_generator.set_camera()
        # create commands array to start a scene
        commands = scene_generator.set_scene()

        print ("==> Data will be generated in folder ", scene_generator.camera.avatar_id)
        # all_object_types = list(scene_generator.extra_info_records.keys())
        
        # iterate to add and destroy objects
        for iter_number in range(100):
            print ('Image coords:', cx, cy, cz)

            first_object_name = choice(scene_generator.usable_records['objects_all']) # "b03_mesh_asian_elephant" # 
            third_object_name = choice(scene_generator.usable_records['objects_all']) # "b03_852100_giraffe" # 
            material_name_obj1, material_name_obj2 = choice(scene_generator.all_materials), choice(scene_generator.all_materials)


            if args.scale_stuff:
                new_color1 = choice(all_material_colors).split("/")[-1].replace(".meta", "").replace(".manifest", "")
                all_objs_tmp = scene_generator.usable_records['objects_huge_lv0'] + scene_generator.usable_records['objects_huge_lv1'] + \
                                scene_generator.usable_records['objects_huge_lv2'] + scene_generator.usable_records['objects_huge_lv3'] + \
                                scene_generator.usable_records['objects_huge_lv4'] + scene_generator.usable_records['objects_huge_lv5']
                first_object_name = choice(all_objs_tmp) # choice(scene_generator.usable_records['objects_huge_lv0']) # "b03_852100_giraffe" # 
                third_object_name = choice(scene_generator.usable_records['objects_small']) # first_object_name

                id_first_object, material1_id, commands = scene_generator.place_object_inside_cameraview(new_object=first_object_name, commands = commands, 
                                                                                displacement=-7.5, material_name=material_name_obj1, 
                                                                                add_to_y=1.2, material_color=new_color1, scale_factor = 0.25) # choice(np.arange(0.1, 1.0, 0.1)))

                new_color2 = choice(all_material_colors).split("/")[-1].replace(".meta", "").replace(".manifest", "")
                new_color2 = choice(all_material_colors).split("/")[-1].replace(".meta", "").replace(".manifest", "")
                # print (third_object_name, new_color2)
                # id_third_object, material2_id, commands = scene_generator.place_object_inside_cameraview(new_object=third_object_name, commands = commands, 
                #                                                                     displacement=7.5, material_name=material_name_obj2, 
                #                                                                     add_to_y=1.2, material_color=new_color2, scale_factor = choice(np.arange(1, 3, 0.5)))

            else:
                new_color1 = choice(all_material_colors).split("/")[-1].replace(".meta", "").replace(".manifest", "")
                print (first_object_name, new_color1)
                id_first_object, material1_id, commands = scene_generator.place_object_inside_cameraview(new_object=first_object_name, commands = commands, 
                                                                                    displacement=-7.5, material_name=material_name_obj1, 
                                                                                    add_to_y=1.2, material_color=new_color1)

                new_color2 = choice(all_material_colors).split("/")[-1].replace(".meta", "").replace(".manifest", "")
                new_color2 = choice(all_material_colors).split("/")[-1].replace(".meta", "").replace(".manifest", "")
                print (third_object_name, new_color2)
                id_third_object, material2_id, commands = scene_generator.place_object_inside_cameraview(new_object=third_object_name, commands = commands, 
                                                                                    displacement=7.5, material_name=material_name_obj2, 
                                                                                    add_to_y=1.2, material_color=new_color2)

            show_human_or_not = choice([True, False, False])
            if show_human_or_not:
                id_human_object, commands = scene_generator.place_human_inside_cameraview(commands = commands, 
                                                                        displacement=3,
                                                                        add_to_y=0)

            scene_generator.excecute_commands(commands)
            scene_generator.do_the_panorama_thing(tilt=0, object_id=id_first_object)
            scene_generator.excecute_commands([])
            commands = []

            scene_generator.excecute_video_and_destroy(commands, [id_first_object], wait=75)

            # reset commands to send new ones!
            commands = []

            with open(image_info_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([camera_name.split("/")[-1], scene_generator.camera.avatar_id, first_object_name, new_color1, material_name_obj1, third_object_name, new_color2, material_name_obj2])
        
        ## for now just end, kill and restart :/
        scene_generator.end_simulation()
        time.sleep(3)
        # close open socket from controller
        scene_generator.scene_controller.socket.close()

    print ("Data generated in folder ", scene_generator.camera.avatar_id)
        

