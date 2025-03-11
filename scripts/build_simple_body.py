from dataclasses import dataclass
from pathlib import Path

import tyro

from embodied_gaussians.scene_builders.domain import save_posed_images, load_posed_images
from embodied_gaussians import SimpleBodyBuilder, SimpleBodyBuilderSettings
from embodied_gaussians.utils.utils import read_extrinsics, read_ground
from utils import get_datapoints_from_live_cameras


@dataclass
class Params:
    name: str
    extrinsics: Path
    ground: Path
    max_depth: float = 2.0
    visualize: bool = False
    offline: bool = False
    save_posed_images: bool = True


def main(params: Params):
    settings = SimpleBodyBuilderSettings()
    ground_data = read_ground(params.ground)
    extrinsics = read_extrinsics(params.extrinsics)
    settings.ground.plane = ground_data

    if not params.offline:
        datapoints = get_datapoints_from_live_cameras(extrinsics) 
        if params.save_posed_images:
            save_posed_images(f"data/posed_images/{params.name}.npz", datapoints)
    else:
        try:
            datapoints = load_posed_images(f"data/posed_images/{params.name}.npz")
        except FileNotFoundError:
            print("Posed images not found. Run with offline=False to generate them")
            return
        
    result = SimpleBodyBuilder.build(name=params.name, settings=settings, datapoints=datapoints, visualize=params.visualize)

    if result is None:
        print("Body builder failed")
        return

    with open(f"data/objects/{params.name}.json", "w") as f:
        f.write(result.model_dump_json(indent=4))
        

if __name__ == "__main__":
    params = tyro.cli(Params)
    main(params)