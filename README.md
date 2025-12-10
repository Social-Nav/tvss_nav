# Task-Oriented Visual-Semantic Social Navigation (`tvss_nav`)

This repository contains the `tvss_nav` ROS package and an installation script that provisions a ready-to-use Catkin workspace under `~/lisn_ws`. It is intended for external users and customers who want to deploy or evaluate the navigation stack with minimal manual setup.

The software targets **ROS Noetic on Ubuntu 20.04**.

## 1. System Requirements

- **Operating system**
  - Ubuntu 20.04 (64-bit)
- **ROS**
  - ROS Noetic, properly initialized:
    - `sudo rosdep init` (once per machine)
    - `rosdep update`
    - Able to run `roscore`
- **System packages** (install missing components as required):
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
    tmux tmuxinator gnome-terminal
  ```
- **Optional – RealSense support**
  - Intel `librealsense2` (from the official distribution)
  - ROS driver: `ros-noetic-realsense2-camera`
- **Python / ML stack**
  - Python 3.11 (recommended via Conda)
  - CUDA-compatible PyTorch build (version depends on your GPU / driver)

### 1.1 Arena-Rosnav Environment (Required for Simulation)

This project relies on the Arena-Rosnav simulation environment (Arena + Pedsim). Install it before running the simulation launch files:

```bash
# In your Catkin workspace, e.g. ~/tvsn_ws/src
git clone git@github.com:Arena-Rosnav/arena-rosnav.git
cd arena-rosnav
git checkout 6ad00193b17cccf160753b97da950b49ca0371c7

# Clone the Arena-Rosnav docs / meta repository (arena3 branch)
cd ..
git clone --branch arena3 https://github.com/Arena-Rosnav/.github.git
```

Then follow the installation and setup instructions for Arena-Rosnav described in the `.github` repository (branch `arena3`) to finish configuring the simulation environment.

After you have a working original Arena-Rosnav setup, you can replace the default simulation configuration with the one used in this `tvss_nav` project:

```bash
cd ~/arena_ws/src/arena
rm -rf simulation-setup
git clone git@github.com:Social-Nav/simulation-setup.git
```

The `Social-Nav/simulation-setup` repository contains the simulation configuration used during the development of `tvss_nav`.

## 2. One-Click Workspace Installation

This repository includes a convenience script that bootstraps a complete workspace at `~/lisn_ws`, including:
- `tvss_nav` (navigation package)
- `tvsn_msgs` (message definitions)
- `dynamic_obstacle_detector` (forked)
- `sfm_local_controller` (forked)
- `lightsfm` (non-catkin dependency)

```bash
git clone <this-repo-url> tvss_nav
cd tvss_nav
bash install_lisn_ws.sh

# The script will:
# - Create a Catkin workspace (src + dependencies) under ~/lisn_ws
# - Clone tvss_nav and the forked tvsn_msgs/dynamic_obstacle_detector/sfm_local_controller/lightsfm
# - Build and install lightsfm (path: dependencies/sfm/lightsfm)
# - Run rosdep to install ROS dependencies
# - Build the workspace with catkin (catkin build or catkin_make)

# After completion:
source ~/lisn_ws/devel/setup.bash
```

Configurable parameters (environment variables):
- `LISN_WS_DIR` (default `~/lisn_ws`)
- `LISN_REMOTE`
- `LISN_MSGS_REMOTE`
- `LISN_DOD_REMOTE`
- `LISN_SFM_REMOTE`
- `LISN_LIGHTSFM_REMOTE`

If the target directory is not empty, set `LISN_FORCE=1` to reuse it.

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
- `source ~/tvsn_ws/devel/setup.bash` has been executed.
- If required, `conda activate tvsn` is active in the current shell.

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

### 4.2 Basic Navigation (Robot or Bag Playback)

For navigation without a simulator GUI (for example on a physical robot or with rosbag playback):

```bash
roslaunch tvss_nav tvss_nav.launch show_rviz:=false
```

If using an Intel RealSense camera, start the RGB‑D node separately:

```bash
roslaunch realsense2_camera rs_rgbd.launch \
  enable_pointcloud:=true \
  filters:=spatial,temporal,hole_filling \
  spatial_filter.enable:=true \
  temporal_filter.enable:=true \
  hole_filling.enable:=true
```

### 4.3 Visual–Semantic Pipeline

To enable the visual–semantic navigation components (requires the Python environment and API keys):

```bash
roslaunch tvss_nav tvss_nav.launch rviz_file:=visual_semantic

# In a separate terminal (with `conda activate tvsn`):
cd tvss_nav/scripts/tvss_nav/tools/grounded_sam2
python gsam2_ros.py

# In another terminal:
cd tvss_nav/scripts/tvss_nav
python -m vlm.vlm
```

Additional tools (for example samplers and goal projectors) are described in `tmux/vlm_tools/.tmuxinator.yml`.

## 5. Reference Files and Utilities
- `install_tvsn_ws.sh`  
  One-click installation script for the Catkin workspace.
- `requirements.txt`  
  Python dependency list (excluding PyTorch and its companion packages, which should be installed according to your CUDA configuration).

## 6. Troubleshooting

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

## 7. Licensing and Support
