#!/usr/bin/env bash
# Entrypoint script for the lisn container.
# Sets up the conda environment and workspace.

ROS_DISTRO=${ROS_DISTRO:-noetic}
LISN_WS_DIR=${LISN_WS_DIR:-/root/lisn_ws}
CONDA_DIR=${CONDA_DIR:-/opt/conda}
CONDA_ENV=${CONDA_ENV:-lisn}

# Source ROS
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

# Source Catkin workspace, if built
if [ -f "${LISN_WS_DIR}/devel/setup.bash" ]; then
  source "${LISN_WS_DIR}/devel/setup.bash"
fi

# Optionally activate conda env
if [ -d "${CONDA_DIR}" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_DIR}/etc/profile.d/conda.sh" || true
  if conda info --envs >/dev/null 2>&1; then
    conda activate "${CONDA_ENV}" || true
  fi
fi

exec "$@"
