# LISN: Language-Instructed Social Navigation with VLM-based Controller Modulating

This repository hosts the code for the LISN project (project page: https://social-nav.github.io/LISN-project/) and contains the `tvss_nav` ROS package plus an installation script that provisions a ready-to-use Catkin workspace under `~/lisn_ws`. It is intended for external users and customers who want to deploy or evaluate the navigation stack with minimal manual setup.

The software targets **ROS Noetic on Ubuntu 20.04** (see [Section 1 — System Requirements](#1-system-requirements)). The project supports both **local installation** (see [Section 2 — One-Click Workspace Installation](#2-one-click-workspace-installation)) and **Docker-based installation** (see [Section 6 — Docker Containerization](#6-docker-containerization)); choose whichever fits your workflow.

### TODO List

- [x] Unify local and Docker installation via `install_lisn_ws.sh`.
- [x] Publish project page and citation info.
- [ ] Migrating this project to Arena 5.0 with more diverse tasks and environments, also for better rendering powered by Isaac Sim.


## 1. System Requirements

- **Operating system**
  - Ubuntu 20.04 (64-bit)
- **ROS**
  - ROS Noetic, properly initialized:
    - `sudo rosdep init` (once per machine)
    - `rosdep update`
    - Able to run `roscore`
- **System packages** (You **DO NOT** need these dependencies if you want a docker installation):
  ```bash
  sudo apt install \
    ros-noetic-nav-core \
    ros-noetic-move-base \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-costmap-2d \
    ros-noetic-pcl-ros \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-realsense2-camera \
    tmux tmuxinator gnome-terminal
  ```
- **Python / ML stack**
  - Python 3.11 (recommended via Conda)
  - CUDA-compatible PyTorch build (version depends on your GPU / driver)


## 2. One-Click Workspace Installation

This repository includes a convenience script that bootstraps a complete workspace at `~/lisn_ws`, including:
- `tvss_nav` (navigation package)
- `tvsn_msgs` (message definitions)
- `dynamic_obstacle_detector` (forked)
- `sfm_local_controller` (forked)
- `lightsfm` (non-catkin dependency)
- Arena-Rosnav stack (with Social-Nav simulation/evaluation replacements; can be skipped via `LISN_SKIP_ARENA=1`)

```bash
mkdir -p ~/lisn_ws/src 
cd ~/lisn_ws/src
git clone <this-repo-url> tvss_nav
cd tvss_nav
bash install_lisn_ws.sh

# The script will:
# - Create a Catkin workspace (src + dependencies) under ~/lisn_ws
# - Clone tvss_nav and the forked tvsn_msgs/dynamic_obstacle_detector/sfm_local_controller/lightsfm
# - Build and install lightsfm (path: dependencies/sfm/lightsfm)
# - Clone Arena-Rosnav (pinned commit) and replace simulation-setup and arena_evaluation with Social-Nav forks
# - Run rosdep to install ROS dependencies
# - Build the workspace with catkin (catkin build or catkin_make)

# After completion:
source ~/lisn_ws/devel/setup.bash
```

Notes:
- The same `install_lisn_ws.sh` script is used for both local setups and the Docker image build. Pass `LISN_SKIP_FETCH=1` when reusing already-cloned sources (for example when mounting a host workspace into the container), and `LISN_SKIP_ROSDEP=1` if you prefer to handle apt dependencies yourself.
- To speed up rebuilds, you can skip lightsfm or Arena-Rosnav with `LISN_SKIP_LIGHTSFM_BUILD=1` or `LISN_SKIP_ARENA=1` respectively.

Configurable parameters (environment variables):
- `LISN_WS_DIR` (default `~/lisn_ws`)
- `LISN_REMOTE`
- `LISN_MSGS_REMOTE`
- `LISN_DOD_REMOTE`
- `LISN_SFM_REMOTE`
- `LISN_LIGHTSFM_REMOTE`

If the target directory is not empty, set `LISN_FORCE=1` to reuse it.

### 2.1 Arena-Rosnav Environment (Deprecated)

The installer now provisions the Arena-Rosnav simulation stack automatically (unless you set `LISN_SKIP_ARENA=1`). It clones Arena-Rosnav, pins commit `6ad00193b17cccf160753b97da950b49ca0371c7`, imports its `.repos` (if `vcstool` is available), and replaces the default simulation/evaluation modules with the Social-Nav versions:

- `simulation-setup` → `https://github.com/Social-Nav/simulation-setup.git`
- `arena_evaluation` → `https://github.com/Social-Nav/arena_evaluation.git`

If you prefer to manage Arena-Rosnav yourself, run with `LISN_SKIP_ARENA=1` and follow the [Arena-Rosnav docs](https://arena-rosnav.readthedocs.io/en/latest/) manually. Note that in this project we only use Gazebo simulation introduced in Arena v3.0.


## 3. Python Environment

The installation script does not create a Python environment. We recommend using Conda:

```bash
conda create -n lisn python=3.11
conda activate lisn

# Install a PyTorch build compatible with your CUDA toolchain (example for CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Install remaining Python dependencies
pip install -r requirements.txt
```

If you plan to use large language / vision–language models, configure the relevant API keys (for example in `~/.bashrc`):

```bash
export OPENAI_API_KEY=...
export ARK_API_KEY=...
export GEMINI_API_KEY=...
```

## 4. Running the System

The following commands assume:
- `source ~/lisn_ws/devel/setup.bash` has been executed (inside Docker the workspace lives at `/root/lisn_ws`).
- If required, `conda activate lisn` is active in the current shell.

### 4.1 Simulation Navigation (Arena + Pedsim)

Launch the full navigation stack in a simulated environment:

```bash
roslaunch --wait tvss_nav start_arena_sfm.launch \
  simulator:=gazebo \
  model:=jackal \
  map_file:=small_warehouse \
  tm_obstacles:=scenario \
  tm_robots:=scenario \
  scenario_file:=default.json \
  entity_manager:=pedsim
```

For online tuning of parameters, you may additionally run:

```bash
rosrun rqt_reconfigure rqt_reconfigure
```

### 4.2 Initialize TVSS Core Services

After setting up the simulation environment, launch the core backend services for the TVSS system. Set `show_rviz:=false` to prevent opening a duplicate visualization window, as RViz is already running from the simulation step (Section 4.1).

```bash
roslaunch tvss_nav tvss_nav.launch show_rviz:=false
```
What this command does:

 - Loads Configurations: Reads static parameters from common.yaml into the parameter server.

 - Starts Communication Bridge: Launches the rosbridge_server (WebSocket), which is essential for exchanging data with the Python VLM pipeline in the next step.

 - Initializes Perception Utilities: Starts the pointcloud_seg (segmentation) and goal_projector nodes to process sensor data and handle goal coordinates.

### 4.3 Visual–Semantic Pipeline

To enable the visual–semantic navigation components (requires the Python environment and API keys):

```bash
# In a separate terminal (with `conda activate lisn`):
cd tvss_nav/scripts/tvss_nav/tools/grounded_sam2
python gsam2_ros.py

# In another terminal:
cd tvss_nav/scripts/tvss_nav
python -m vlm.vlm
```

## 5. Reference Files and Utilities
`install_lisn_ws.sh`  
  One-click installation script for the Catkin workspace.
- `requirements.txt`  
  Python dependency list (excluding PyTorch and its companion packages, which should be installed according to your CUDA configuration).

## 6. Docker Containerization

For easy deployment and development, Docker containers are provided with GPU support and volume mounting for live development. The Docker setup automatically includes Arena-Rosnav simulation environment dependencies required for running simulations.

### 6.1 Building the Docker Image

```bash
cd docker
docker build -f Dockerfile.ros-torch -t lisn:latest .
```

The Docker image includes:
- Arena-Rosnav simulation environment (Arena + Pedsim)
- All required ROS packages for navigation and simulation
- Foxglove bridge for real-time data visualization
- GPU acceleration support

### 6.2 Running Simulations with Docker

Use the provided simulation script for automated setup:

```bash
cd docker
./run_simulation.sh
```

This will:
- Start the container with GPU support (if available)
- Mount the workspace for live development
- Launch the simulation with Gazebo and RViz
- Start Foxglove bridge for data visualization

Inside the container, build the workspace by running `install_lisn_ws.sh`:

```bash
source /opt/ros/noetic/setup.bash
cd /root/lisn_ws/src/tvss_nav
bash install_lisn_ws.sh
```

### 6.3 Foxglove Visualization

The Docker setup includes Foxglove bridge for real-time ROS data visualization:

1. **Connect from host machine:**
   - Open Foxglove Studio
   - Select "Open Connection" → "Foxglove WebSocket"
   - Enter URL: `ws://localhost:8765`
   - Click "Open" to explore robot data

2. **Available data includes:**
   - Robot pose and odometry
   - Sensor data (LiDAR, camera)
   - Navigation goals and paths
   - Simulation state and diagnostics

### 6.4 Development Workflow

The container uses volume mounting, so changes to source code are reflected immediately without rebuilding the image. For development:

```bash
# Edit files in your workspace
# Changes are automatically available in the running container
# Rebuild only when dependencies change
```

## 7. Troubleshooting

- **`catkin` / `catkin_make` not found**  
  Ensure that ROS Noetic is sourced:
  ```bash
  source /opt/ros/noetic/setup.bash
  ```
- **RealSense topics not available**  
  Verify that `librealsense2` and `realsense2_camera` are installed, and that `rs_rgbd.launch` is running.
- **API‑key–dependent features not working**  
  Confirm that `OPENAI_API_KEY`, `ARK_API_KEY`, and `GEMINI_API_KEY` are defined in the environment. Core navigation will still operate without them, but advanced LLM/VLM features may be unavailable.
- **Missing ROS dependencies during build**  
  Re‑run:
  ```bash
  rosdep install --from-paths src --ignore-src -r -y
  ```
  and install any reported missing system packages.

## 8. **Cite LISN**

```
@misc{chen2025lisnlanguageinstructedsocialnavigation,
  title={LISN: Language-Instructed Social Navigation with VLM-based Controller Modulating}, 
  author={Junting Chen and Yunchuan Li and Panfeng Jiang and Jiacheng Du and Zixuan Chen and Chenrui Tie and Jiajun Deng and Lin Shao},
  year={2025},
  eprint={2512.09920},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2512.09920}, 
}
```