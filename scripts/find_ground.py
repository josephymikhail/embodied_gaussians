import time
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import tyro

from ground_finder import GroundFinder, GroundFinderSettings
from embodied_gaussians.utils.utils import read_extrinsics
from utils import get_datapoints_from_live_cameras


@dataclass
class Params:
    output: tyro.conf.PositionalRequiredArgs[Path]
    extrinsics: Path
    max_depth: float = 2.0
    settings: GroundFinderSettings = field(
        default_factory=lambda: GroundFinderSettings()
    )
    visualize: bool = False


def main(params: Params):
    all_extrinsics = read_extrinsics(params.extrinsics)
    datapoints = get_datapoints_from_live_cameras(all_extrinsics)

    result = GroundFinder.find_ground(
        params.settings, datapoints, visualize=params.visualize
    )
    res = {"plane": result.plane.tolist()}

    with open(f"{params.output}", "w") as f:
        json.dump(res, f, indent=2)

    np.save(f"{params.output.with_suffix('.npy')}", result.points)


if __name__ == "__main__":
    params = tyro.cli(Params)
    assert params.output.suffix == ".json"
    main(params)

