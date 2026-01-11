
import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
from i2rt.lerobot.helpers import (
    get_yam_robot,
)
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_yam_follower_bimanual import YAMFollowerBimanualConfig

logger = logging.getLogger(__name__)

YAM_ARM_MOTOR_NAMES_BIMANUAL = [
    "joint1_left",
    "joint2_left",
    "joint3_left",
    "joint4_left",
    "joint5_left",
    "joint6_left",
    "gripper_left",
    "joint1_right",
    "joint2_right",
    "joint3_right",
    "joint4_right",
    "joint5_right",
    "joint6_right",
    "gripper_right",
]

class YAMFollowerBimanual(Robot):
    """
    YAM follower arm designed by I2RT.
    """

    config_class = YAMFollowerBimanualConfig
    name = "yam_follower_bimanual"

    def __init__(self, config: YAMFollowerBimanualConfig):
        super().__init__(config)
        self.config = config
        self.robot_left: MotorChainRobot | None = None
        self.robot_right: MotorChainRobot | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

        # Joint info (initialized in connect)
        self.joint_limits: np.ndarray | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in YAM_ARM_MOTOR_NAMES_BIMANUAL
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        robot_left_is_connected = (
            self.robot_left is not None
            and hasattr(self.robot_left.motor_chain, "running")
            and self.robot_left.motor_chain.running
        )
        robot_right_is_connected = (
            self.robot_right is not None
            and hasattr(self.robot_right.motor_chain, "running")
            and self.robot_right.motor_chain.running
        )

        cameras_are_connected = all(cam.is_connected for cam in self.cameras.values())
        return robot_left_is_connected and robot_right_is_connected and cameras_are_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        for cam in self.cameras.values():
            cam.connect()

        self.robot_left = get_yam_robot(
            channel=self.config.port_left,
            gripper_type=self.config.gripper_type,
        )
        self.robot_right = get_yam_robot(
            channel=self.config.port_right,
            gripper_type=self.config.gripper_type,
        )

        current_kp = self.robot_left._kp.copy()
        current_kd = self.robot_left._kd.copy()

        # Reduce gripper kp to slow it down (e.g., from 20 to 5)
        if self.config.use_almond_kp_kd:
            current_kp[6] = 5.0  # Much gentler position control
            current_kd[6] = 1.0  # Increase damping to smooth it out

        self.robot_left.update_kp_kd(current_kp, current_kd)
        self.robot_right.update_kp_kd(current_kp, current_kd)

        # Cache joint limits directly from the robot object (avoids buggy get_robot_info())
        self.joint_limits = self.robot_left._joint_limits  # Shape: (6, 2) for 6 arm joints in radians

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # YAM doesn't need to calibrate
        return True

    def calibrate(self) -> None:
        raise NotImplementedError("YAM doesn't need to calibrate")

    def configure(self) -> None:
        raise NotImplementedError("YAM doesn't need to configure")

    def setup_motors(self) -> None:
        raise NotImplementedError("YAM doesn't need to setup motors")

    def _get_motor_positions(self) -> dict[str, float]:
        """Get current motor positions in radians."""
        joint_positions_left = self.robot_left.get_joint_pos()  # Returns np.ndarray in radians
        joint_positions_right = self.robot_right.get_joint_pos()  # Returns np.ndarray in radians
        joint_positions = np.concatenate([joint_positions_left, joint_positions_right])
        motor_positions = {}
        for i, motor_name in enumerate(YAM_ARM_MOTOR_NAMES_BIMANUAL):
            motor_positions[motor_name] = joint_positions[i]
        return motor_positions

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = {f"{motor}.pos": val for motor, val in self._get_motor_positions().items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Command arm to move to a target joint configuration.
        Refer to YAM_ARM_MOTOR_NAMES_BIMANUAL for features names.
        
        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.
        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        joint_positions_rad = [0.0] * len(YAM_ARM_MOTOR_NAMES_BIMANUAL) 

        for i, motor_name in enumerate(YAM_ARM_MOTOR_NAMES_BIMANUAL):
            if motor_name in goal_pos:
                if "gripper" in motor_name:
                    gripper_val_0_1 = goal_pos[motor_name]
                    gripper_val_0_1 = np.clip(
                        gripper_val_0_1, self.config.gripper_minimum_position, self.config.gripper_maximum_position
                    )
                    joint_positions_rad[i] = gripper_val_0_1
                else:
                    joint_positions_rad[i] = goal_pos[motor_name]

        # Send command to robots
        self.robot_left.command_joint_pos(np.array(joint_positions_rad[:7]))
        self.robot_right.command_joint_pos(np.array(joint_positions_rad[7:]))

        # Return the (potentially clipped) action that was sent
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.robot_left.close()
        self.robot_right.close()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def go_to_position(
        self,
        target_positions: dict[str, float],
        duration: float = 2.0,
        steps: int = 50,
    ) -> None:
        """
        Move robot to target position with linear interpolation.
        
        Args:
            target_positions: Target joint positions as dict (e.g., {"joint1_left.pos": 0.5, ...})
            duration: Time to complete the motion in seconds
            steps: Number of interpolation steps
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        current_positions = {f"{k}.pos": v for k, v in self._get_motor_positions().items()}
        
        dt = duration / steps
        for i in range(1, steps + 1):
            alpha = i / steps
            interpolated = {}
            for key in current_positions:
                start = current_positions[key]
                end = target_positions.get(key, start)
                interpolated[key] = start + alpha * (end - start)
            
            self.send_action(interpolated)
            time.sleep(dt)