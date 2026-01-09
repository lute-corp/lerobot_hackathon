from dataclasses import dataclass, field

from i2rt.robots.utils import GripperType

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YAMFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str = "can_left"
    gripper_type: GripperType = GripperType.LINEAR_4310
    gripper_max_force: float = 50.0
    gripper_maximum_position: float = 1.0
    gripper_minimum_position: float = 0.0

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
