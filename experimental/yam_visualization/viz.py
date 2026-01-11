"""Minimal bimanual YAM robot visualizer with REST API for joint control."""

import argparse
import os
import threading

import numpy as np
import viser
import viser.extras
import yourdfpy
from flask import Flask, request, jsonify

NUM_JOINTS = 6

# Global state
joints_left = np.zeros(NUM_JOINTS)
joints_right = np.zeros(NUM_JOINTS)
urdf_vis_left = None
urdf_vis_right = None

# Flask app for REST API
app = Flask(__name__)

@app.route('/set_joints', methods=['POST'])
def set_joints():
    global joints_left, joints_right
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Missing JSON data'}), 400
    
    # Extract joints for both arms
    new_joints_left = np.zeros(NUM_JOINTS, dtype=np.float64)
    new_joints_right = np.zeros(NUM_JOINTS, dtype=np.float64)
    
    for i in range(1, NUM_JOINTS + 1):
        key_left = f'joint{i}_left'
        key_right = f'joint{i}_right'
        
        if key_left not in data:
            return jsonify({'error': f'Missing joint: {key_left}'}), 400
        if key_right not in data:
            return jsonify({'error': f'Missing joint: {key_right}'}), 400
        
        new_joints_left[i - 1] = data[key_left]
        new_joints_right[i - 1] = data[key_right]
    
    joints_left = new_joints_left
    joints_right = new_joints_right
    return jsonify({'status': 'ok'})

def run_flask(host, port):
    app.run(host=host, port=port, threaded=True)

def main():
    global urdf_vis_left, urdf_vis_right
    
    parser = argparse.ArgumentParser(description="Bimanual YAM robot visualizer with REST API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--viser-port", type=int, default=8081, help="Viser server port (default: 8081)")
    parser.add_argument("--joints-port", type=int, default=5001, help="REST API port for receiving joints (default: 5001)")
    args = parser.parse_args()
    
    # Load URDF
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, 'yam.urdf')
    mesh_dir = script_dir
    
    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    
    # Create viser server
    server = viser.ViserServer(host=args.host, port=args.viser_port)
    print(f"Viser running at http://{args.host}:{args.viser_port}")
    
    # Add ground grid
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Arms are offset from each other by 61cm
    arms_offset = 0.61

    # Add left arm visualization
    base_left = server.scene.add_frame("/base_left", show_axes=False)
    base_left.position = (0.0, arms_offset / 2, 0.0)
    urdf_vis_left = viser.extras.ViserUrdf(server, urdf, root_node_name="/base_left")
    
    # Add right arm visualization 
    base_right = server.scene.add_frame("/base_right", show_axes=False)
    base_right.position = (0.0, -arms_offset / 2, 0.0)
    urdf_vis_right = viser.extras.ViserUrdf(server, urdf, root_node_name="/base_right")
    
    # Start REST API in background thread
    flask_thread = threading.Thread(target=run_flask, args=(args.host, args.joints_port), daemon=True)
    flask_thread.start()
    print(f"REST API running at http://{args.host}:{args.joints_port}")
    print("  POST /set_joints with {'joint1_left': val, ..., 'joint6_left': val, 'joint1_right': val, ..., 'joint6_right': val}")
    
    # Main visualization loop
    while True:
        # IMPORTANT: Flip the joint order to match URDF's internal joint ordering
        # The URDF joints are processed in reverse order (joint6, joint5, ..., joint1)
        # compared to the motor order (joint1, joint2, ..., joint6)
        urdf_vis_left.update_cfg(np.flip(joints_left))
        urdf_vis_right.update_cfg(np.flip(joints_right))
        server.flush()

if __name__ == '__main__':
    main()
