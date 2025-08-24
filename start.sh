#!/bin/bash
(
  cd ~/arena_ws/src/thirdparty/sfm/dep/lightsfm || exit 1
  echo "📦 Cleaning previous lightsfm build..."
  make clean
  echo "📦 Building lightsfm..."
  make -j$(nproc)
  echo "📦 Installing lightsfm to /usr/local/include/lightsfm..."
  sudo -E make install
)
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
    #     tmux kill-session -t sfm-arena_p`. This was detected when initializing the generation config instance, which 
    tmux kill-server
    echo "✅ All tmux sessions killed."

    echo "✅ All cleaned up. Bye!"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# -------- Launch terminal tasks --------
gnome-terminal -- bash -c "\
  cd tvss_nav/scripts/tvss_nav/vlm && \
  python3 query_publisher.py; \
  exec bash" & get_new_pts
sleep 1
gnome-terminal -- bash -c "\
  cd tvss_nav/scripts/tvss_nav/sampler/robopoint_sampler && \
  python3 local_inference_ros.py; \
  exec bash" & get_new_pts
sleep 1


gnome-terminal -- bash -c "cd ./tmux/arena_sfm/ && tmuxinator; exec bash" & get_new_pts

sleep 2  # Give time for roscore in tmux to start

# gnome-terminal -- bash -c "cd ./tmux/semantic_tool/ && tmuxinator; exec bash" & get_new_pts



gnome-terminal -- bash -c "roslaunch tvss_nav tvss_nav.launch show_rviz:=false; exec bash" & get_new_pts
# gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav/tools/grounded_sam2 && python gsam2_ros_terminal.py; exec bash" & get_new_pts
gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav/tools/grounded_sam2 && python gsam2_ros.py; exec bash" & get_new_pts
gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav && python -m vlm.vlm; exec bash" & get_new_pts
# gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav && python -m sampler.subgoal_sampler; exec bash" & get_new_pts



# gnome-terminal -- bash -c "roslaunch tvss_nav tvss_nav.launch show_rviz:=true; exec bash" & get_new_pts
# gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav/tools/grounded_sam2 && python gsam2_ros_terminal.py; exec bash" & get_new_pts
gnome-terminal -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav/tools/grounded_sam2 && python gsam2_ros.py; exec bash" & get_new_pts
gnome-terminal -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav && python -m vlm.vlm; exec bash" & get_new_pts
# gnome-terminal -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate tvsn && cd ./tvss_nav/scripts/tvss_nav && python -m sampler.subgoal_sampler; exec bash" & get_new_pts
gnome-terminal -- bash -c "\
  rostopic echo /rosout --filter \"m.level <= 16\" | sed -n 's/^[[:space:]]*msg: //p'; \
  exec bash" & get_new_pts
# -------- Exit handling --------

echo "Press 'q' to quit, close all spawned terminals, and kill tmux session if running (or press Ctrl+C)..."

while true; do
    read -n 1 key
    if [[ $key == "q" ]]; then
        cleanup
    fi
done
