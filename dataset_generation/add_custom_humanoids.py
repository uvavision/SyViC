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
