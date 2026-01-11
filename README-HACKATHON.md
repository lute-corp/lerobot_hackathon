# Policy Deployment Guide

This guide walks you through deploying a trained policy on the YAM robot hardware.

---

## Assumptions

We provide a server machine that you can access via SSH. The server is already configured with:

- Robot cameras and motors ready to use
- The robot motors driver (`i2rt` Python package) installed and available in `PYTHONPATH`

The `robot_client.py` script from the `lerobot_hackathon` repository runs on our server. Policy inference is also handled by the server by default, but can be executed on a separate machine if needed—just like the visualization code, which you run on your own machine.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/lute-corp/lerobot_hackathon
cd lerobot_hackathon
```

### Step 2: Create Conda Environment

```bash
conda create -n lerobot python=3.10
conda activate lerobot
```

### Step 3: Install Dependencies

Install the package with your policy's requirements:

```bash
pip install -e '.[all]'
```

> **Note:** Despite that we are using the requirements as provided in the original lerobot repository, we ran into issues with installation for '[all]'. The workaround is to install the requirements for the specific policy (e.g. '[act]') and then manually install the packages that turn out at runtime to be missing.

### Step 4: Install Visualizer Dependencies

```bash
cd experimental/yam_visualization/
pip install -r requirements.txt
```

## Running the Policy

You'll need **three terminal sessions** to run the full deployment:

---

### Terminal 1: Start the Visualizer (Your Machine)

Launch the web-browser-based robot arm visualizer:

```bash
python experimental/yam_visualization/viz.py
```

---

### Terminal 2: Start the Policy Server (Server)

Start the policy inference service:

```bash
python -m lerobot.async_inference.policy_server --host 127.0.0.1 --port 8080
```

---

### Terminal 3: Run the Robot Client (Server)

Run the robot client with your policy configuration:

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<POLICY_SERVER_IP>:8080 \
    --robot.type=yam_follower_bimanual \
    --robot.cameras='{
        "top": {
            "type": "intelrealsense",
            "serial_number_or_name": "409122273659",
            "width": 640, "height": 480, "fps": 30
        },
        "wrist_left": {
            "type": "intelrealsense",
            "serial_number_or_name": "352122273510",
            "width": 640, "height": 480, "fps": 30
        },
        "wrist_right": {
            "type": "intelrealsense",
            "serial_number_or_name": "352122271362",
            "width": 640, "height": 480, "fps": 30
        }
    }' \
    --task "fold the white cloth and put it on the left side of the table" \
    --policy_type=act \
    --actions_per_chunk=50 \
    --pretrained_name_or_path=<PATH_TO_MODEL_WEIGHTS> \
    --chunk_size_threshold=0.0 \
    --aggregate_fn_name=weighted_average \
    --policy_device=cuda \
    --viz_only=true \
    --viz_joints_endpoint=<VISUALIZER_IP>:5001
```

> **Note on `--viz_only` mode:** When set to `true`, the predicted action chunks are displayed in the visualizer but **no commands are sent to the motors**. This allows you to sanity-check your policy before running on the real robot. However, the robot must still be connected—observations (camera images and joint positions) are read from the real hardware to feed into the policy.
