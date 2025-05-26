#!/bin/bash

# -------- Preparation --------
# Function to get current pts (pseudo terminals)
get_current_pts_set() {
    ps -ef | grep "bash -c" | grep -v grep | awk '{print $6}' | sort
}

# Initialize tracking variables
BEFORE_PTS=$(get_current_pts_set)
PTS_LIST=()

# Function to detect newly spawned terminals
get_new_pts() {
    sleep 1  # Give time for terminal to initialize
    local AFTER_PTS=$(get_current_pts_set)
    local NEW_PTS=$(comm -13 <(echo "$BEFORE_PTS") <(echo "$AFTER_PTS"))
    BEFORE_PTS="$AFTER_PTS"
    for pts in $NEW_PTS; do
        PTS_LIST+=("$pts")
    done
}

# -------- Cleanup Function (for q or Ctrl+C) --------
cleanup() {
    echo -e "\n🚨 Shutting down all spawned terminals..."

    for pts in "${PTS_LIST[@]}"; do
        echo "Killing terminal $pts..."
        pkill -t "$pts"
    done

    # Kill tmux session if exists
    # if tmux has-session -t sfm-arena 2>/dev/null; then
    #     tmux kill-session -t sfm-arena
    #     echo "✅ tmux session 'sfm-arena' killed."
    # else
    #     echo "⚠️ No tmux session found."
    # fi

    tmux kill-server
    echo "✅ All tmux sessions killed."

    echo "✅ All cleaned up. Bye!"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# -------- Launch terminal tasks --------

gnome-terminal -- bash -c "roslaunch tvss_nav tvss_nav.launch; exec bash" & get_new_pts
gnome-terminal -- bash -c "roslaunch realsense2_camera rs_rgbd.launch \
    enable_pointcloud:=true \
    filters:=spatial,temporal,hole_filling \
    spatial_filter.enable:=true \
    temporal_filter.enable:=true \
    hole_filling.enable:=true; exec bash" & get_new_pts
gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate gsam2 && cd ./tvss_nav/scripts/tvss_nav/tools/grounded_sam2 && python gsam2_ros_topic.py; exec bash" & get_new_pts
# gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate gsam2 && cd ./tvss_nav/scripts/tvss_nav/tools/grounded_sam2 && python gsam2_ros.py; exec bash" & get_new_pts
gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate gsam2 && cd ./tvss_nav/scripts/tvss_nav && python -m vlm.vlm; exec bash" & get_new_pts
gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate gsam2 && cd ./tvss_nav/scripts/tvss_nav && python -m sampler.subgoal_sampler; exec bash" & get_new_pts
gnome-terminal -- bash -c "rviz -d ~/visual_semantic.rviz; exec bash" & get_new_pts

# -------- Exit handling --------

echo "Press 'q' to quit, close all spawned terminals, and kill tmux session if running (or press Ctrl+C)..."

while true; do
    read -n 1 key
    if [[ $key == "q" ]]; then
        cleanup
    fi
done
