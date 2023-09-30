# Readme - Easy FBX to TDW end to end pipeline 
**Step by step instructions to import a TEACH animation to TDW using multiple SMPLx clothe models**

**Contents:**

+ **cmu_37_37_01_male.zip** contains the prefab for the SMPL model with the corresponding uv mapping, please unzip it and place it into: ```asset_bundle_creator/Assets/prefabs``` 
  + This prefab can be modified, such that a set of colliders can be attached to allow for gravity and collision detection
  + This prefab is also the baseline model to create all clothed SMPL-Xs
+ **asset_bundle_creator:Assets:Scripts:Editor:** contains all the scripts that should be placed in ```asset_bundle_creator/Assets/Scripts/Editor/```
  + To generate all materials (.mat) to clothe the SMPLs:
    1. Download all images from https://www.di.ens.fr/willow/research/surreal/data/ or https://virtualhumans.mpi-inf.mpg.de/mgn/ and place them in ```asset_bundle_creator/Assets/Resources/Textures```
    2. Open Unity, open the asset_bundle_creator project and in the Assets Menu (on top, right next to the Edit menu), click on ```Generate Materials```. This will run a script that generates a bunch of .mat that should reference to all texture images.
    3. Run the script ```assign_correct_clothe_to_material.py``` to assign the correct metadata for all the generated materials.
    4. Once all materials are correctly referenced with their corresponding image, you can automatically generate copies of the prefab (cmu_37_37_01_male) and assign the clothe/texture by running ```CLOTHE SMPLs using TDW Prefab``` under the Assets menu.
    5. Now you should move all the generated prefabs from ```asset_bundle_creator/Assets/SMPLX_base/PREFABS/``` to ```asset_bundle_creator/Assets/prefabs```. Please **NOTE** that all these prefabs should be contained inside a folder named after the prefab name, e.g., move from ```asset_bundle_creator/Assets/prefabs/this_is_name_1/this_is_name_1.prefab```, ```asset_bundle_creator/Assets/prefabs/this_is_name_2/this_is_name_2.prefab```, etc.
    6. Now, you should be able to run the following script to use TDW's humanoid creator to generate the clothed SMPL asset bundles that we will use later:
          ```python
          # TDW reference: https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/non_physics_humanoids/custom_humanoids.md 

          from pathlib import Path
          from tdw.asset_bundle_creator.humanoid_creator import HumanoidCreator
          from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
          import glob

          output_directory = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("Multi_Garmentdataset")
          print(f"Asset bundles will be saved to: {output_directory}")
          r = HumanoidCreator()

          all_files = glob.glob("asset_bundle_creator/Assets/prefabs/*")
          for ff in all_files:
          r.prefab_to_asset_bundles(name=ff.split("/")[-1], 
                                  output_directory=output_directory)   
          ```
          Now you should have a set of asset bundles ready to be used in TDW!!


+ **client_side** contains a useful step by step **Jupyter Notebook** to send the generated FBX to the server, the name of the clothed SMPL to be used, the name of the scene selected by the user and the duration of the animation. The server is where Unity and TDW is running. After the server process the whole request, it sends back to the client a video (.mp4) which is rendered inside the notebook to show the result.

+ **server_side** should be executed in the machine where TDW and Unity run. It contains all functions to recieve the FBX that is sent from the client, import the FBX as a humanoid rig to Unity, correctly exctract the animation, and create the asset bundle. This animation asset is then used along with the selected clothed SMPL model, and placed in the selected scene. After rendering for the specified length, a video is created. This video is finally sent back to the client.
  + New classes: ```ExtractAnimFromTeachFbx``` and ```CustomAnimationCreator```
  + Codeblock to add a specific clothed SMPL and apply an animation: 
    ```python
    """
    Add and animate a SMPL humanoid.
    """
    .
    .
    .
    ``` 
    + To add gravity and physics you should ```communicate``` something like these ```commands``` to the TDW ```Controller```:
    ```python
    commands.extend([{"$type": "set_kinematic_state",
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
                      "id": dropped_object_id}])
     ```

**Finally, to render scenes with multiple humanoids please run: ```render_multiple_humanoids.py```. You can also inspect ```smpl.py``` to check for sample code to include animated humanoids in a scene. To run images with different arrangement of objectsa attributes please check ```render_images_objects.py```.**