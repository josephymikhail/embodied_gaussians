from dataclasses import dataclass
from pathlib import Path

import tyro

from embodied_gaussians.scene_builders.domain import save_posed_images, load_posed_images
from embodied_gaussians.scene_builders.pointcloud_body_builder import (
    PointCloudBodyBuilder,
    PointCloudBodyBuilderSettings,
)
from embodied_gaussians.utils.utils import read_extrinsics, read_pointcloud
from utils import get_datapoints_from_live_cameras


@dataclass
class Params:
    name: str
    extrinsics: Path
    points: Path
    max_depth: float = 2.0
    visualize: bool = False
    offline: bool = False
    save_posed_images: bool = True


def main(params: Params):
    settings = PointCloudBodyBuilderSettings()
    extrinsics = read_extrinsics(params.extrinsics)
    points = read_pointcloud(params.points)
    settings.training_learning_rates.colors = 0.1
    settings.training_learning_rates.means = 0.0
    settings.training_learning_rates.scales = 0.004
    settings.training_learning_rates.quats = 0.0
    settings.min_scale = 0.0
    settings.max_scale = 0.03

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

    result = PointCloudBodyBuilder.build(
        name=params.name,
        points=points,
        settings=settings,
        datapoints=datapoints,
        visualize=params.visualize,
    )

    if result is None:
        print("Body builder failed")
        return

    with open(f"data/objects/{params.name}.json", "w") as f:
        f.write(result.model_dump_json(indent=4))


if __name__ == "__main__":
    params = tyro.cli(Params)
    main(params)
