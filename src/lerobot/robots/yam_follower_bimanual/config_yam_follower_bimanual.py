from dataclasses import dataclass, field

import numpy as np
from i2rt.robots.utils import GripperType

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("yam_follower_bimanual")
@dataclass
class YAMFollowerBimanualConfig(RobotConfig):
    # Port to connect to the arm
    port_left: str = "can_left"
    port_right: str = "can_right"
    gripper_type: GripperType = GripperType.LINEAR_4310
    gripper_max_force: float = 5.0
    gripper_maximum_position: float = 1.0
    gripper_minimum_position: float = 0.0

    use_almond_kp_kd: bool = True

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
