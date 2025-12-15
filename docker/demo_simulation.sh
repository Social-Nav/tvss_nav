#!/usr/bin/env bash
set -euo pipefail

# Script to run LISN simulation (in Docker container)

# Run the installation script
cd /root/lisn_ws/src/tvss_nav && bash install_lisn_ws.sh

# Source the workspace
source /root/lisn_ws/devel/setup.bash

# run roscore in background
roscore &

# Launch Foxglove bridge in background
roslaunch foxglove_bridge foxglove_bridge.launch &

# Give Foxglove bridge time to start
sleep 3

# Run the simulation
roslaunch --wait tvss_nav start_arena_sfm.launch simulator:=gazebo model:=jackal map_file:=small_warehouse tm_obstacles:=scenario tm_robots:=scenario scenario_file:=default.json entity_manager:=pedsim