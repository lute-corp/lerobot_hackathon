"""
Standalone script to send pre-gathered observation data to a policy server for inference
and visualize the predicted actions in viser.

Designed specifically for the yam_follower_bimanual robot. The script:
1. Reads images from three cameras (top, wrist_left, wrist_right) and motor positions from disk
2. Connects to a gRPC policy server and sends the observation
3. Receives predicted actions from the policy server
4. Continuously loops through the actions, sending them to a viser visualization server

Example:

```shell
python send_offline_observation.py \
    --data_dir=/path/to/observation_data \
    --server_address=localhost:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --task="pick up the object" \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --viz_joints_address=127.0.0.1:5001
```

Expected data directory structure:
    data_dir/
    ├── top.png (or .jpg, .jpeg)
    ├── wrist_left.png (or .jpg, .jpeg)
    ├── wrist_right.png (or .jpg, .jpeg)
    └── motor_positions.json

motor_positions.json format (all 14 joints required):
{
    "joint1_left": 0.1,
    "joint2_left": 0.2,
    "joint3_left": 0.3,
    "joint4_left": 0.4,
    "joint5_left": 0.5,
    "joint6_left": 0.6,
    "gripper_left": 0.0,
    "joint1_right": 0.1,
    "joint2_right": 0.2,
    "joint3_right": 0.3,
    "joint4_right": 0.4,
    "joint5_right": 0.5,
    "joint6_right": 0.6,
    "gripper_right": 0.0
}
"""

import argparse
import json
import logging
import pickle  # nosec
import time
from pathlib import Path
from typing import Any

import cv2
import grpc
import numpy as np
import requests

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Motor names for yam_follower_bimanual robot
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

# Expected camera names
CAMERA_NAMES = ["top", "wrist_left", "wrist_right"]


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk and return as numpy array (H, W, C) in BGR format."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def find_image_file(data_dir: Path, camera_name: str) -> Path:
    """Find image file for a camera, supporting both .png and .jpg extensions."""
    for ext in [".png", ".jpg", ".jpeg"]:
        image_path = data_dir / f"{camera_name}{ext}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(
        f"Could not find image for camera '{camera_name}' in {data_dir}. "
        f"Expected one of: {camera_name}.png, {camera_name}.jpg, {camera_name}.jpeg"
    )


def load_motor_positions(json_path: Path) -> dict[str, float]:
    """Load motor positions from JSON file."""
    with open(json_path) as f:
        motor_positions = json.load(f)

    # Validate that all expected motors are present
    for motor in YAM_ARM_MOTOR_NAMES_BIMANUAL:
        if motor not in motor_positions:
            raise ValueError(f"Missing motor '{motor}' in {json_path}")

    return motor_positions


def build_observation(data_dir: Path) -> dict:
    """
    Build observation dictionary from files in data_dir.

    Returns observation in the same format as yam_follower_bimanual.get_observation():
    - Motor positions: {f"{motor}.pos": value}
    - Camera images: {cam_key: np.ndarray}
    """
    data_dir = Path(data_dir)

    # Load motor positions
    motor_positions_path = data_dir / "motor_positions.json"
    if not motor_positions_path.exists():
        raise FileNotFoundError(f"motor_positions.json not found in {data_dir}")

    motor_positions = load_motor_positions(motor_positions_path)

    # Build observation dict with motor positions
    obs_dict = {f"{motor}.pos": motor_positions[motor] for motor in YAM_ARM_MOTOR_NAMES_BIMANUAL}

    # Load camera images
    for cam_name in CAMERA_NAMES:
        image_path = find_image_file(data_dir, cam_name)
        obs_dict[cam_name] = load_image(image_path)
        logger.info(f"Loaded {cam_name} image: {image_path} (shape: {obs_dict[cam_name].shape})")

    return obs_dict


def build_lerobot_features(
    observation: dict, camera_shapes: dict[str, tuple]
) -> dict[str, dict]:
    """
    Build lerobot features dictionary from observation.
    This mimics the hw_to_dataset_features function for the specific observation structure.
    """
    features = {}

    # Motor features (state)
    motor_keys = [k for k in observation if k.endswith(".pos")]
    features[f"{OBS_STR}.state"] = {
        "dtype": "float32",
        "shape": (len(motor_keys),),
        "names": motor_keys,
    }

    # Camera features
    for cam_name, shape in camera_shapes.items():
        features[f"{OBS_STR}.images.{cam_name}"] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    return features


def send_action_to_viser(action: dict[str, Any], joints_url: str) -> None:
    """Send action to viser for visualization."""
    viser_action = {k.removesuffix(".pos"): v for k, v in action.items()}
    requests.post(joints_url, json=viser_action)


class OfflineObservationSender:
    """Minimal gRPC client for sending offline observations to policy server."""

    def __init__(
        self,
        server_address: str,
        policy_type: str,
        pretrained_name_or_path: str,
        lerobot_features: dict[str, dict],
        actions_per_chunk: int,
        policy_device: str = "cpu",
    ):
        self.server_address = server_address
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.lerobot_features = lerobot_features
        self.actions_per_chunk = actions_per_chunk
        self.policy_device = policy_device

        self.channel = None
        self.stub = None
        self.running = False

    def connect(self) -> bool:
        """Connect to the policy server and send policy instructions."""
        try:
            self.channel = grpc.insecure_channel(
                self.server_address, grpc_channel_options()
            )
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            logger.info(f"Connecting to server at {self.server_address}")

            # Handshake with server
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            logger.info(f"Connected to policy server in {end_time - start_time:.4f}s")

            # Send policy instructions
            policy_config = RemotePolicyConfig(
                policy_type=self.policy_type,
                pretrained_name_or_path=self.pretrained_name_or_path,
                lerobot_features=self.lerobot_features,
                actions_per_chunk=self.actions_per_chunk,
                device=self.policy_device,
            )
            policy_config_bytes = pickle.dumps(policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            logger.info("Sending policy instructions to policy server")
            logger.info(
                f"Policy type: {self.policy_type} | "
                f"Pretrained: {self.pretrained_name_or_path} | "
                f"Device: {self.policy_device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)
            self.running = True

            return True

        except grpc.RpcError as e:
            logger.error(f"Failed to connect to policy server: {e}")
            return False

    def send_observation(self, obs: TimedObservation) -> bool:
        """
        Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise.
        """
        if not self.running:
            raise RuntimeError("Client not connected. Call connect() first.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            logger.info(f"Sent observation #{obs_timestep}")

            return True

        except grpc.RpcError as e:
            logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def receive_actions(self) -> list[TimedAction]:
        """
        Receive action chunk from the policy server.
        Returns a list of TimedAction objects.
        """
        if not self.running:
            raise RuntimeError("Client not connected. Call connect() first.")

        try:
            actions_chunk = self.stub.GetActions(services_pb2.Empty())
            if len(actions_chunk.data) == 0:
                return []  # received `Empty` from server

            # Deserialize bytes back into list[TimedAction]
            timed_actions = pickle.loads(actions_chunk.data)  # nosec
            logger.info(f"Received {len(timed_actions)} actions from server")
            return timed_actions

        except grpc.RpcError as e:
            logger.error(f"Error receiving actions: {e}")
            return []

    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            logger.info("Channel closed")
        self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="Send offline observation data to policy server for inference"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing observation data (images and motor_positions.json)",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:8080",
        help="Policy server address (default: localhost:8080)",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Type of policy to use (e.g., act, pi0, smolvla)",
    )
    parser.add_argument(
        "--pretrained_name_or_path",
        type=str,
        required=True,
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Task description string",
    )
    parser.add_argument(
        "--policy_device",
        type=str,
        default="cpu",
        help="Device for policy inference (default: cpu)",
    )
    parser.add_argument(
        "--actions_per_chunk",
        type=int,
        default=50,
        help="Number of actions per chunk (default: 50)",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=0,
        help="Timestep to assign to the observation (default: 0)",
    )
    parser.add_argument(
        "--viz_joints_address",
        type=str,
        default="127.0.0.1:5001",
        help="Address for the viser joints endpoint (default: 127.0.0.1:5001)",
    )

    args = parser.parse_args()

    # Build observation from data directory
    logger.info(f"Loading observation data from: {args.data_dir}")
    observation = build_observation(args.data_dir)

    # Add task to observation if provided
    if args.task:
        observation["task"] = args.task
        logger.info(f"Task: {args.task}")

    # Get camera shapes for feature building
    camera_shapes = {
        cam_name: observation[cam_name].shape for cam_name in CAMERA_NAMES
    }

    # Build lerobot features
    lerobot_features = build_lerobot_features(observation, camera_shapes)
    logger.info(f"LeRobot features: {list(lerobot_features.keys())}")

    # Create TimedObservation
    timed_obs = TimedObservation(
        timestamp=time.time(),
        observation=observation,
        timestep=args.timestep,
        must_go=True,  # Force processing since this is the only observation
    )

    # Create sender and connect
    sender = OfflineObservationSender(
        server_address=args.server_address,
        policy_type=args.policy_type,
        pretrained_name_or_path=args.pretrained_name_or_path,
        lerobot_features=lerobot_features,
        actions_per_chunk=args.actions_per_chunk,
        policy_device=args.policy_device,
    )

    # Build viser joints URL
    joints_url = f"http://{args.viz_joints_address}/set_joints"
    logger.info(f"Viser joints URL: {joints_url}")

    try:
        if sender.connect():
            logger.info("Successfully connected to policy server")

            # Send the observation
            success = sender.send_observation(timed_obs)
            if success:
                logger.info("Observation sent successfully!")

                # Receive actions from the server
                logger.info("Waiting for actions from policy server...")
                timed_actions = sender.receive_actions()

                if timed_actions:
                    logger.info(f"Received {len(timed_actions)} actions, looping through action chunk...")
                    logger.info("Press Ctrl+C to stop")

                    # Loop sending actions to viser until interrupted
                    try:
                        while True:
                            for i, timed_action in enumerate(timed_actions):
                                action_tensor = timed_action.get_action()
                                # Convert tensor to dict using motor names
                                action_dict = {
                                    f"{motor}.pos": action_tensor[j].item()
                                    for j, motor in enumerate(YAM_ARM_MOTOR_NAMES_BIMANUAL)
                                }
                                send_action_to_viser(action_dict, joints_url)
                                logger.info(f"Sent action {i + 1}/{len(timed_actions)} to viser")
                                # Small delay between actions for visualization
                                time.sleep(0.02)
                            time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Interrupted by user, stopping...")
                else:
                    logger.warning("No actions received from server")
            else:
                logger.error("Failed to send observation")

        else:
            logger.error("Failed to connect to policy server")

    finally:
        sender.close()


if __name__ == "__main__":
    main()
