# Minimal YAM Visualizer

Standalone bimanual robot arm visualizer with REST API for joint control.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Start the visualizer:

```bash
python viz.py
```

With custom ports:

```bash
python viz.py --viser-port 8081 --joints-port 5001 --host 0.0.0.0
```

### Options

| Argument        | Default   | Description                     |
| --------------- | --------- | ------------------------------- |
| `--host`        | `0.0.0.0` | Host to bind to                 |
| `--viser-port`  | `8081`    | Viser visualization server port |
| `--joints-port` | `5001`    | REST API port for joint control |

## Visualization

Open http://localhost:8081 in your browser to view the robot arms.

## REST API

### POST `/set_joints`

Send joint positions for both arms (6 joints each):

```bash
curl -X POST http://localhost:5001/set_joints \
  -H "Content-Type: application/json" \
  -d '{
    "joint1_left": 0.0, "joint2_left": 0.0, "joint3_left": 0.0,
    "joint4_left": 0.0, "joint5_left": 0.0, "joint6_left": 0.0,
    "joint1_right": 0.0, "joint2_right": 0.0, "joint3_right": 0.0,
    "joint4_right": 0.0, "joint5_right": 0.0, "joint6_right": 0.0
  }'
```

All 12 joint values (`joint1_left` through `joint6_left` and `joint1_right` through `joint6_right`) are required.

> **Note:** Gripper state is not visualized.
