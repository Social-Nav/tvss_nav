#!/usr/bin/env bash

# Install / bootstrap a tvss_nav workspace under ~/tvsn_ws
# - Only the tvss_nav repo needs to be cloned manually.
# - This script will clone tvss_nav + dynamic_obstacle_detector + sfm_local_controller + lightsfm
#   into a fresh catkin workspace and build everything.
#
# Configuration (can be overridden via env vars before running):
#   TVSN_WS_DIR           : workspace root, default "$HOME/tvsn_ws"
#   TVSN_REMOTE           : tvss_nav git URL
#   TVSN_DOD_REMOTE       : dynamic_obstacle_detector fork URL
#   TVSN_SFM_REMOTE       : sfm_local_controller fork URL
#   TVSN_MSGS_REMOTE      : tvsn_msgs git URL
#   TVSN_LIGHTSFM_REMOTE  : lightsfm fork URL
#   TVSN_FORCE=1          : allow using a non-empty workspace directory

set -euo pipefail

log()  { printf '[tvsn] %s\n' "$*" >&2; }
err()  { printf '[tvsn][ERROR] %s\n' "$*" >&2; }
warn() { printf '[tvsn][WARN] %s\n' "$*" >&2; }

# Workspace root
WS_DIR="${TVSN_WS_DIR:-$HOME/tvsn_ws}"

# Default remotes (override via env if needed)
TVSN_REMOTE="${TVSN_REMOTE:-git@github.com:Social-Nav/tvss_nav.git}"
TVSN_DOD_REMOTE="${TVSN_DOD_REMOTE:-git@github.com:Social-Nav/dynamic_obstacle_detector.git}"
TVSN_SFM_REMOTE="${TVSN_SFM_REMOTE:-git@github.com:Social-Nav/sfm_local_controller.git}"
TVSN_MSGS_REMOTE="${TVSN_MSGS_REMOTE:-git@github.com:Social-Nav/tvsn_msgs.git}"
TVSN_LIGHTSFM_REMOTE="${TVSN_LIGHTSFM_REMOTE:-git@github.com:Social-Nav/lightsfm.git}"

if [[ -z "${TVSN_REMOTE}" ]]; then
  err "TVSN_REMOTE is not set. Please set a git URL for the tvss_nav package."
  exit 1
fi

# Basic environment checks
if ! command -v git >/dev/null 2>&1; then
  err "git not found. Please install git first."
  exit 1
fi

if ! command -v rosversion >/dev/null 2>&1; then
  err "ROS environment not detected. Please 'source /opt/ros/noetic/setup.bash' and run again."
  exit 1
fi

if ! command -v rosdep >/dev/null 2>&1; then
  warn "rosdep not found. Will skip 'rosdep install'. Please install dependencies manually if needed."
fi

if ! command -v make >/dev/null 2>&1; then
  err "make not found. Cannot build lightsfm."
  exit 1
fi

if ! command -v catkin >/dev/null 2>&1 && ! command -v catkin_make >/dev/null 2>&1; then
  err "Neither 'catkin' nor 'catkin_make' found. Please install catkin tools."
  exit 1
fi

# Prepare workspace directory
log "Using workspace directory: ${WS_DIR}"
mkdir -p "${WS_DIR}"

if [[ -n "$(ls -A "${WS_DIR}" 2>/dev/null || true)" ]] && [[ "${TVSN_FORCE:-0}" != "1" ]]; then
  err "Workspace directory ${WS_DIR} is not empty. Set TVSN_FORCE=1 to reuse this directory."
  exit 1
fi

cd "${WS_DIR}"

# Helper: clone or update git repo
clone_or_update() {
  local target="$1"
  local url="$2"

  mkdir -p "$(dirname "${target}")"

  if [[ -d "${target}/.git" ]]; then
    log "Updating existing repo: ${target}"
    git -C "${target}" fetch --all --prune
    git -C "${target}" pull --ff-only
  elif [[ -d "${target}" ]]; then
    warn "Directory ${target} exists but is not a git repo. Skipping clone."
  else
    log "Cloning ${url} -> ${target}"
    git clone "${url}" "${target}"
  fi
}

log "Creating catkin workspace structure (src + dependencies)..."
mkdir -p src
mkdir -p dependencies/sfm

# Clone tvss_nav into src
clone_or_update "src/tvss_nav" "${TVSN_REMOTE}"

# Clone dependent catkin packages into src
clone_or_update "src/tvsn_msgs" "${TVSN_MSGS_REMOTE}"
clone_or_update "src/dynamic_obstacle_detector" "${TVSN_DOD_REMOTE}"
clone_or_update "src/sfm_local_controller" "${TVSN_SFM_REMOTE}"

# Clone lightsfm into a non-catkin dependencies folder
clone_or_update "dependencies/sfm/lightsfm" "${TVSN_LIGHTSFM_REMOTE}"

# Build and install lightsfm
log "Building and installing lightsfm..."
pushd dependencies/sfm/lightsfm >/dev/null
MAKE_JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"
make -j"${MAKE_JOBS}"
if command -v sudo >/dev/null 2>&1; then
  sudo make install
else
  warn "sudo not found. Will try plain 'make install' (may require write permission to system dirs)."
  make install
fi
popd >/dev/null

# Install ROS dependencies
if command -v rosdep >/dev/null 2>&1; then
  log "Running rosdep to install ROS dependencies..."
  if ! rosdep install --from-paths src --ignore-src -r -y; then
    warn "rosdep failed to install some dependencies. Please check errors and install them manually."
  fi
fi

# Build catkin workspace
log "Building catkin workspace..."

if command -v catkin >/dev/null 2>&1; then
  # catkin tools
  log "Found 'catkin' (catkin tools). Using 'catkin build'."
  catkin config --source-space src || true
  catkin build
elif command -v catkin_make >/dev/null 2>&1; then
  # catkin_make
  log "Found 'catkin_make'. Using 'catkin_make'."
  if [[ ! -f src/CMakeLists.txt ]]; then
    log "Initializing src as a catkin workspace..."
    (cd src && catkin_init_workspace)
  fi
  catkin_make
fi

log "Build finished."
echo
echo "Next steps:"
echo "  source \"${WS_DIR}/devel/setup.bash\""
echo "  roslaunch tvss_nav tvss_nav.launch"
echo
echo "To rebuild/reuse the same workspace directory, set TVSN_FORCE=1 and run this script again."
